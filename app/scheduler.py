"""流水线调度器。

并发模型：
- 全局 ``asyncio.Semaphore(2)`` 即"两个名额"，名额在整个步骤生命周期内持有，
  包含进程组清理阶段 —— 组未清空前不释放名额、不写步骤终态。
- 慢清理只占住自己的名额，不阻塞另一名额上的独立步骤/构建。
- 每个构建一个 asyncio 任务，轮询 DAG 状态推进；每个运行中的步骤一个子任务，
  通过 per-build cancel event 实现整次取消的即时响应。
- 人工放行门禁：依赖全成功后进入 ``awaiting_approval``，此时不启动进程、
  不占用名额；签核 API 写库后由主循环推进 —— 批准则后代按原依赖规则启动，
  拒绝则门禁落 ``rejected``、后代跳过。取消优先于签核。
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import config, proc
from .db import Database
from .models import BUILD_TERMINAL, BuildCreate, validate_dag

# 这些状态都视为"上游未成功"：后代跳过
_STEP_FAIL_STATES = {"failed", "skipped", "cancelled", "interrupted", "rejected"}
# 未结束但也不占执行名额的门禁状态
_AWAITING = "awaiting_approval"


class ApprovalError(Exception):
    """签核请求失败，携带建议的 HTTP 状态码。"""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Scheduler:
    def __init__(self, db: Database) -> None:
        self.db = db
        self._slots = asyncio.Semaphore(config.MAX_PARALLEL)
        self._build_events: dict[int, asyncio.Event] = {}
        self._build_tasks: dict[int, asyncio.Task[None]] = {}
        # 每构建一把启动锁：保证"检查取消 -> 启动进程"与"受理取消"互斥，
        # 签核与取消交错时不会启动已取消构建的命令。
        self._start_locks: dict[int, asyncio.Lock] = {}

    # ------------------------------------------------------------------ API

    async def submit(self, payload: BuildCreate) -> int:
        validate_dag(payload.steps)  # ValueError 由路由层转 400
        build_id = await self.db.create_build(payload.name, _now())
        log_dir = config.LOG_ROOT / str(build_id)
        log_dir.mkdir(parents=True, exist_ok=True)
        for step in payload.steps:
            log_path = log_dir / f"{step.id}.log"
            log_path.touch()
            await self.db.create_step(
                build_id,
                step.id,
                step.command or "",
                step.depends_on,
                step.timeout_seconds,
                str(log_path),
                approval_required=step.approval_required,
            )
        task = asyncio.create_task(self._run_build(build_id), name=f"build-{build_id}")
        self._build_tasks[build_id] = task
        task.add_done_callback(self._log_task_exception)
        return build_id

    async def decide_approval(
        self,
        build_id: int,
        step_uid: str,
        decision: str,
        decided_by: Optional[str],
        comment: Optional[str],
    ) -> dict[str, Any]:
        """提交门禁签核结论。

        受理条件（任一不满足即抛 ApprovalError）：
        - 构建与步骤存在；
        - 步骤是门禁步骤且当前正处于待签核；
        - 构建未进入终态，也没有已受理的取消请求（取消优先）。
        批准后由构建主循环重新核对取消事件，再决定是否放行后代。
        """
        build = await self.db.get_build(build_id)
        if build is None:
            raise ApprovalError(404, "构建不存在")
        step = await self.db.get_step(build_id, step_uid)
        if step is None:
            raise ApprovalError(404, "构建或步骤不存在")
        if not step["approval_required"]:
            raise ApprovalError(409, f"步骤 {step_uid} 不是人工放行门禁")
        if build["status"] in BUILD_TERMINAL:
            raise ApprovalError(
                409, f"构建已处于终态 {build['status']}，签核不再受理"
            )
        event = self._build_events.get(build_id)
        if build["cancel_requested"] or (event is not None and event.is_set()):
            raise ApprovalError(409, "构建已请求取消，门禁签核不再受理")
        if step["status"] != _AWAITING or step["approval_status"] != "pending":
            raise ApprovalError(
                409,
                f"门禁当前状态为 {step['status']}/{step['approval_status']}，"
                "仅待签核门禁可受理一次结论",
            )

        next_status = "succeeded" if decision == "approve" else "rejected"
        # 与取消受理共用构建级启动锁：取消一旦落库，此处必然看到标记并拒绝；
        # 批准先落库时，取消随后置事件，主循环不会为已取消构建启动后代。
        async with self._start_lock(build_id):
            build_now = await self.db.get_build(build_id)
            event_now = self._build_events.get(build_id)
            if (
                build_now is None
                or build_now["cancel_requested"]
                or (event_now is not None and event_now.is_set())
            ):
                raise ApprovalError(409, "构建已请求取消，门禁签核不再受理")
            ok = await self.db.decide_approval(
                step["id"],
                decision,
                _now(),
                decided_by,
                comment,
                next_status,
            )
        if not ok:
            # 条件更新失败：并发下已被取消或已有结论
            raise ApprovalError(409, "门禁已被处理，仅待签核门禁可受理一次结论")

        if decision == "reject":
            await self._append_decision_log(step, f"拒绝签核 by={decided_by or '-'}")
        else:
            await self._append_decision_log(
                step, f"批准签核 by={decided_by or '-'}，等待调度器核对取消状态"
            )
        fresh = await self.db.get_step(build_id, step_uid)
        assert fresh is not None
        return dict(fresh)

    @staticmethod
    async def _append_decision_log(step: Any, message: str) -> None:
        try:
            await asyncio.to_thread(
                Scheduler._append_log,
                step["log_path"],
                [f"[runner] {message}"],
            )
        except OSError:
            pass

    async def awaiting_approvals(self) -> list[dict[str, Any]]:
        """当前待签核门禁（含所属构建信息），供查询接口与 /health 使用。"""
        steps = await self.db.list_awaiting_approval_steps()
        result: list[dict[str, Any]] = []
        for step in steps:
            build = await self.db.get_build(step["build_id"])
            if build is None:
                continue
            result.append({**dict(step), "build_name": build["name"],
                           "build_status": build["status"]})
        return result

    async def cancel(self, build_id: int) -> tuple[Optional[dict[str, Any]], str]:
        """返回 (构建行或 None, 说明)。重复取消幂等，不改变已定终态。"""
        row = await self.db.get_build(build_id)
        if row is None:
            return None, "构建不存在"
        if row["status"] in BUILD_TERMINAL:
            return dict(row), f"构建已处于终态 {row['status']}，取消请求忽略"
        # 取消落库/置事件与"取消检查 + spawn"共用一把构建级锁：
        # 取消一旦受理，后续启动路径在锁内必然看到事件，绝不再启动新命令。
        # 已在锁内启动的进程会在 _watch 中立即感知事件并整组终止。
        async with self._start_lock(build_id):
            await self.db.request_cancel(build_id)
            event = self._build_events.get(build_id)
            if event is not None:
                event.set()
        return dict(row), "取消请求已受理，运行中的步骤将终止整个进程组"

    # ------------------------------------------------------------- 重启恢复

    async def recover(self) -> dict[str, int]:
        """API 启动时执行：先清理旧构建残留进程，再统一标记 interrupted。

        旧构建之外的进程（PID 已被复用、starttime 对不上）绝不动。
        恢复期间不恢复调度，未跑步骤一律不再启动。
        """
        stale = await self.db.list_nonterminal_builds()
        if not stale:
            # 没有未完成构建时也要清理历史遗留的待签核标记（理论上不会出现，
            # 但旧版本库可能残留，避免 /approvals 显示幽灵门禁）。
            for step in await self.db.list_awaiting_approval_steps():
                await self.db.expire_approval(step["id"], _now())
            return {"builds": 0, "killed_groups": 0}

        rows = await self.db.list_steps_for_builds(tuple(b["id"] for b in stale))
        grouped: dict[int, list[Any]] = {}
        for row in rows:
            grouped.setdefault(row["build_id"], []).append(row)

        killed_groups = 0
        for build in stale:
            build_id = build["id"]
            for step in grouped.get(build_id, []):
                if step["status"] not in ("running", "cleaning_up"):
                    continue
                pgid = step["pgid"]
                pid = step["pid"]
                starttime = step["proc_starttime"]
                if not pgid:
                    continue
                note: Optional[str] = None
                identity_ok = bool(
                    pid
                    and proc.is_same_process(pid, starttime)
                    and _safe_getpgid(pid) == pgid
                )
                members = await asyncio.to_thread(proc.list_pgid_members, pgid)
                if identity_ok:
                    note = "API 重启恢复：进程身份确认，终止旧进程组"
                elif members:
                    # 组长已死但组内仍有成员：组未空，pgid 不可能被复用，成员即残留
                    note = "API 重启恢复：组长已退出但组内仍有残留成员，强制清理"
                if note is None:
                    continue
                log_lines = [f"[runner] {note} build={build_id} step={step['step_uid']}"]
                cleared = await proc.terminate_process_group(
                    None, pgid, "重启清理", log_lines
                )
                killed_groups += 1
                self._append_log(step["log_path"], log_lines)
                if not cleared:
                    self._append_log(
                        step["log_path"], ["[runner] 警告: 重启清理超过硬等上限仍有残留"]
                    )

            for step in grouped.get(build_id, []):
                if step["status"] not in (
                    "succeeded", "failed", "skipped", "cancelled", "rejected"
                ):
                    await self.db.set_step_status(
                        step["id"],
                        "interrupted",
                        _now(),
                        detail="API 进程重启，步骤被标记中断，不再启动",
                    )
                    self._append_log(
                        step["log_path"], ["[runner] API 重启：该步骤标记为 interrupted"]
                    )
                if step["approval_status"] == "pending":
                    # 旧签核随重启作废：待签核门禁被中断，不得因旧请求补跑
                    await self.db.expire_approval(step["id"], _now())
                    self._append_log(
                        step["log_path"],
                        ["[runner] API 重启：待签核请求已作废，需重新提交构建"],
                    )
            await self.db.set_build_status(
                build_id,
                "interrupted",
                _now(),
                detail="API 进程重启，旧构建中断，未启动步骤不再执行",
                started=build["started_at"] is None,
                finished=True,
            )
        return {"builds": len(stale), "killed_groups": killed_groups}

    # ------------------------------------------------------------- 构建主循环

    async def _run_build(self, build_id: int) -> None:
        event = asyncio.Event()
        self._build_events[build_id] = event
        # (步骤 uid, 步骤任务) 成对保存，避免用任务名反解 uid
        step_tasks: list[tuple[str, asyncio.Task[str]]] = []
        try:
            await self.db.set_build_status(
                build_id, "running", _now(), started=True
            )
            rows = await self.db.get_steps(build_id)
            statuses = {row["step_uid"]: row["status"] for row in rows}
            by_id = {row["step_uid"]: row for row in rows}

            while True:
                # 先对账：已完成的步骤任务把终态写回内存视图，否则主循环
                # 会一直看到 "running" 而无法推进/退出
                for uid, task in step_tasks:
                    if task.done() and statuses[uid] not in _STEP_FAIL_STATES | {"succeeded"}:
                        try:
                            statuses[uid] = task.result()
                        except Exception as exc:  # 任务异常按失败处理
                            statuses[uid] = "failed"
                            self._append_log(
                                by_id[uid]["log_path"],
                                [f"[runner] 步骤任务异常: {exc!r}"],
                            )

                # 门禁结论由签核 API 写库，主循环每轮刷新待签核步骤的真实状态，
                # 批准/拒绝后才能推进后代或跳过。
                for uid, row in by_id.items():
                    if statuses[uid] != _AWAITING:
                        continue
                    fresh = await self.db.get_step(build_id, uid)
                    if fresh is not None:
                        statuses[uid] = fresh["status"]

                # 取消优先：未启动/待签核步骤不再启动，运行中的交给步骤任务自行终止
                if event.is_set():
                    for uid, st in statuses.items():
                        if st == _AWAITING:
                            statuses[uid] = "cancelled"
                            await self.db.cancel_awaiting_step(
                                by_id[uid]["id"],
                                _now(),
                                "构建已取消，待签核门禁作废",
                            )
                        elif st == "pending":
                            statuses[uid] = "cancelled"
                            await self.db.set_step_status(
                                by_id[uid]["id"],
                                "cancelled",
                                _now(),
                                detail="构建已取消，步骤未启动",
                            )
                    break

                progressed = False
                for uid, row in by_id.items():
                    if statuses[uid] != "pending":
                        continue
                    deps = json.loads(row["depends_on"])
                    dep_st = [statuses[d] for d in deps]
                    if any(s in _STEP_FAIL_STATES for s in dep_st):
                        statuses[uid] = "skipped"
                        await self.db.set_step_status(
                            row["id"],
                            "skipped",
                            _now(),
                            detail=f"上游步骤未成功，后代跳过: {deps}",
                        )
                        progressed = True
                        continue
                    # 依赖未全部结束则等下一轮；全部 succeeded 才启动
                    if any(s != "succeeded" for s in dep_st):
                        continue
                    # 放行前再核对取消：批准与取消交错时不得启动已取消构建的命令
                    fresh_build = await self.db.get_build(build_id)
                    if (
                        event.is_set()
                        or fresh_build is None
                        or fresh_build["cancel_requested"]
                    ):
                        continue
                    if row["approval_required"] and not row["command"]:
                        # 纯门禁步骤无需执行命令，直接进入待签核，不占名额
                        ok = await self.db.mark_awaiting_approval(
                            row["id"], _now(), from_statuses=("pending",)
                        )
                        fresh = await self.db.get_step_by_pk(row["id"])
                        statuses[uid] = (
                            fresh["status"] if fresh is not None else "cancelled"
                        )
                        if ok:
                            self._append_log(
                                row["log_path"],
                                [
                                    f"[runner] build={build_id} step={uid} "
                                    "进入待签核（未执行命令）"
                                ],
                            )
                        progressed = True
                        continue
                    task = asyncio.create_task(
                        self._run_step_with_slot(build_id, row),
                        name=f"build-{build_id}-step-{uid}",
                    )
                    step_tasks.append((uid, task))
                    statuses[uid] = "running"
                    progressed = True

                if all(
                    st in _STEP_FAIL_STATES or st == "succeeded"
                    for st in statuses.values()
                ):
                    # 待签核既非终态也非 succeeded，因此有门禁时不会提前退出，
                    # 主循环会持续等待签核结论。
                    break
                # 没有新推进时睡眠等待；有推进也短暂让出，避免忙等
                await asyncio.sleep(config.SCHEDULER_TICK if not progressed else 0)

            # 等待所有在途步骤（含进程组清理）落终态后再确认构建终态
            results = await asyncio.gather(
                *(task for _, task in step_tasks), return_exceptions=True
            )
            for (uid, _), result in zip(step_tasks, results):
                if isinstance(result, Exception):
                    statuses[uid] = "failed"
                else:
                    statuses[uid] = result

            final_rows = await self.db.get_steps(build_id)
            final_statuses = [r["status"] for r in final_rows]
            if "rejected" in final_statuses:
                final = "failed"
                detail = "人工门禁被拒绝，后代跳过"
            elif "failed" in final_statuses or "interrupted" in final_statuses:
                final = "failed"
                detail = "存在失败步骤"
            elif "cancelled" in final_statuses:
                final = "cancelled"
                detail = "构建被取消"
            else:
                final = "succeeded"
                detail = "全部步骤成功"
            await self.db.set_build_status(
                build_id, final, _now(), detail=detail, finished=True
            )
        finally:
            self._build_events.pop(build_id, None)
            self._build_tasks.pop(build_id, None)
            self._start_locks.pop(build_id, None)

    # ------------------------------------------------------------- 单步执行

    async def _run_step_with_slot(self, build_id: int, row: Any) -> str:
        """名额覆盖"启动 -> 运行 -> 进程组完全清空 -> 终态落库"全过程。"""
        async with self._slots:
            return await self._run_step(build_id, row)

    def _start_lock(self, build_id: int) -> asyncio.Lock:
        return self._start_locks.setdefault(build_id, asyncio.Lock())

    async def _cancel_requested(self, build_id: int) -> bool:
        event = self._build_events.get(build_id)
        if event is not None and event.is_set():
            return True
        build = await self.db.get_build(build_id)
        return build is None or bool(build["cancel_requested"])

    async def _run_step(self, build_id: int, row: Any) -> str:
        step_pk = row["id"]
        step_uid = row["step_uid"]
        command = row["command"]
        timeout = float(row["timeout_seconds"])
        workdir = config.WORKDIR_ROOT / str(build_id)
        workdir.mkdir(parents=True, exist_ok=True)
        log_path = Path(row["log_path"])
        log_lines = [
            f"[runner] build={build_id} step={step_uid} 开始",
            f"[runner] workdir={workdir}",
            f"[runner] command={command}",
            f"[runner] timeout={timeout}s",
        ]
        self._append_log(log_path, log_lines)

        env = {
            **os.environ,
            "PIPELINE_BUILD_ID": str(build_id),
            "PIPELINE_STEP_ID": step_uid,
        }
        log_f = open(log_path, "ab", buffering=0)
        try:
            try:
                # 取消检查与 spawn 必须在同一把锁内完成：取消受理后绝不再启动
                # 新进程，签核放行与取消交错时也不会启动已取消构建的命令。
                async with self._start_lock(build_id):
                    if await self._cancel_requested(build_id):
                        await self.db.set_step_status(
                            step_pk,
                            "cancelled",
                            _now(),
                            detail="构建已取消，步骤未启动",
                        )
                        self._append_log(
                            log_path, ["[runner] 构建已取消，步骤未启动"]
                        )
                        return "cancelled"
                    process, pid, pgid, starttime = await proc.spawn(
                        command, workdir, env, log_f, log_f
                    )
            except Exception as exc:  # 命令本身都无法启动
                await self.db.set_step_status(
                    step_pk, "failed", _now(), detail=f"进程启动失败: {exc}"
                )
                self._append_log(log_path, [f"[runner] 进程启动失败: {exc}"])
                return "failed"

            await self.db.set_step_process(step_pk, pid, pgid, starttime, _now())
            cancel_event = self._build_events[build_id]
            outcome, detail = await self._watch(process, timeout, cancel_event)

            # 进入清理阶段即对外可见：终态尚未确认、名额仍占用
            await self.db.set_step_status(
                step_pk, "cleaning_up", _now(), detail=detail
            )
            leftovers_before = await asyncio.to_thread(
                proc.list_pgid_members, pgid
            )
            had_leftovers = bool(leftovers_before)
            cleanup_log: list[str] = []
            if outcome == "timeout":
                reason = "步骤超时"
            elif outcome == "cancelled":
                reason = "构建取消"
            else:
                reason = "主命令退出后检查残留子进程"
            cleared = await proc.terminate_process_group(
                process, pgid, reason, cleanup_log
            )
            self._append_log(log_path, cleanup_log)
            exit_code = process.returncode

            is_gate = bool(row["approval_required"])
            if not cleared:
                status, final_detail = (
                    "failed",
                    f"{detail or ''}; 进程组在硬等上限内未清空".strip("; "),
                )
            elif outcome == "timeout":
                status, final_detail = "failed", detail
            elif outcome == "cancelled":
                status, final_detail = "cancelled", detail
            elif had_leftovers:
                status = "failed"
                final_detail = (
                    f"主命令已退出(exit={exit_code})，但检测到遗留子进程，"
                    "进程组清理后按失败处理"
                )
            elif exit_code == 0:
                status, final_detail = "succeeded", None
            else:
                status, final_detail = "failed", f"命令以非零状态退出: {exit_code}"

            # 门禁步骤命令成功后不直接判成功，而是转入待签核；条件更新保证
            # 与取消/签核交错时只有仍处于运行/清理收尾的步骤能进入。
            if is_gate and status == "succeeded":
                if await self._cancel_requested(build_id):
                    # 命令跑完的同一拍收到取消：不得进入待签核，直接落取消终态
                    await self.db.set_step_status(
                        step_pk,
                        "cancelled",
                        _now(),
                        exit_code=exit_code if isinstance(exit_code, int) else None,
                        detail="构建已取消，门禁不进入待签核",
                    )
                    self._append_log(
                        log_path, ["[runner] 构建已取消，门禁不进入待签核"]
                    )
                    return "cancelled"
                transitioned = await self.db.mark_awaiting_approval(
                    step_pk, _now(), from_statuses=("running", "cleaning_up")
                )
                if transitioned:
                    self._append_log(
                        log_path,
                        [
                            f"[runner] 步骤终态=awaiting_approval exit={exit_code} "
                            "detail=人工放行门禁等待签核"
                        ],
                    )
                    return _AWAITING
                # 条件更新失败说明步骤已被并发改写（如取消），以库中状态为准
                fresh = await self.db.get_step_by_pk(step_pk)
                return fresh["status"] if fresh is not None else "failed"

            await self.db.set_step_status(
                step_pk,
                status,
                _now(),
                exit_code=exit_code if isinstance(exit_code, int) else None,
                detail=final_detail,
            )
            self._append_log(
                log_path,
                [
                    f"[runner] 步骤终态={status} exit={exit_code} "
                    f"遗留子进程={had_leftovers} detail={final_detail or '-'}"
                ],
            )
            return status
        finally:
            log_f.close()

    async def _watch(
        self, process: Any, timeout: float, cancel_event: asyncio.Event
    ) -> tuple[str, Optional[str]]:
        """三者竞速：进程退出 / 取消 / 超时。取消与超时只负责发信号，
        实际的进程组终止统一由调用方在清理阶段完成。"""
        wait_task = asyncio.create_task(process.wait())
        cancel_task = asyncio.create_task(cancel_event.wait())
        timeout_task = asyncio.create_task(asyncio.sleep(timeout))
        try:
            await asyncio.wait(
                {wait_task, cancel_task, timeout_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wait_task.done():
                return "exited", None
            if cancel_task.done():
                return "cancelled", "构建已取消，终止整个进程组"
            return "timeout", f"步骤超过 {timeout}s 超时，终止整个进程组"
        finally:
            for task in (cancel_task, timeout_task):
                task.cancel()
            await asyncio.gather(cancel_task, timeout_task, return_exceptions=True)

    # ------------------------------------------------------------- 工具方法

    @staticmethod
    def _append_log(log_path: str, lines: list[str]) -> None:
        try:
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
        except OSError:
            pass

    @staticmethod
    def _log_task_exception(task: asyncio.Task[Any]) -> None:
        if task.cancelled() or task.exception() is None:
            return
        # 构建任务内部已兜住大部分异常；这里仅留痕，避免静默吞掉
        import logging

        logging.getLogger("pipeline").exception(
            "构建任务异常: %s", task.exception()
        )


def _safe_getpgid(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None
