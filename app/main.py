"""FastAPI 应用：路由、序列化与启动恢复。"""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query

from . import __version__, config
from .db import Database
from .models import (
    ApprovalIn,
    ApprovalOut,
    AwaitingApprovalOut,
    BuildCreate,
    BuildOut,
    CancelOut,
    LogOut,
    StepOut,
)
from .scheduler import ApprovalError, Scheduler

db = Database()
scheduler = Scheduler(db)


def _step_out(row) -> StepOut:
    return StepOut(
        id=row["step_uid"],
        command=row["command"] or None,
        depends_on=json.loads(row["depends_on"]),
        timeout_seconds=row["timeout_seconds"],
        status=row["status"],
        exit_code=row["exit_code"],
        detail=row["detail"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        approval_required=bool(row["approval_required"]),
        approval_status=row["approval_status"],
        approval_requested_at=row["approval_requested_at"],
        approval_decided_at=row["approval_decided_at"],
        approval_decided_by=row["approval_decided_by"],
        approval_comment=row["approval_comment"],
    )


async def _build_out(build_id: int) -> BuildOut:
    row = await db.get_build(build_id)
    if row is None:
        raise HTTPException(status_code=404, detail="构建不存在")
    steps = [_step_out(r) for r in await db.get_steps(build_id)]
    return BuildOut(
        id=row["id"],
        name=row["name"],
        status=row["status"],
        cancel_requested=bool(row["cancel_requested"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        detail=row["detail"],
        steps=steps,
    )


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # 顺序不可颠倒：先落库连接，再清空旧构建残留并标记中断，最后才对外服务
    await db.connect()
    app.state.recovery = await scheduler.recover()
    try:
        yield
    finally:
        await db.close()


app = FastAPI(
    title="固件构建流水线 API",
    version=__version__,
    description=(
        "接收最多 12 步的 JSON DAG，在挂载工作目录中执行真实命令；"
        "全局最多两步并行，依赖全成功才启动，失败后代跳过，"
        "超时/取消终止整个进程组。步骤可标记为人工放行门禁：依赖全部成功后"
        "进入待签核且不占执行名额，经签核 API 批准后才放行后代、拒绝则跳过后代。"
        "支持逐步状态、签核结论与日志查询。"
    ),
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "max_steps": config.MAX_STEPS,
        "max_parallel": config.MAX_PARALLEL,
        "awaiting_approval": len(await scheduler.awaiting_approvals()),
        "recovery": getattr(app.state, "recovery", None),
    }


@app.post("/builds", response_model=BuildOut, status_code=201)
async def create_build(payload: BuildCreate) -> BuildOut:
    try:
        build_id = await scheduler.submit(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return await _build_out(build_id)


@app.get("/builds", response_model=list[BuildOut])
async def list_builds(limit: int = Query(100, ge=1, le=500)) -> list[BuildOut]:
    rows = await db.list_builds(limit)
    return [await _build_out(r["id"]) for r in rows]


@app.get("/approvals", response_model=list[AwaitingApprovalOut])
async def list_awaiting_approvals(
    limit: int = Query(100, ge=1, le=500),
) -> list[AwaitingApprovalOut]:
    """列出当前待签核的门禁：包含所属构建与门禁步骤。"""
    steps = await scheduler.awaiting_approvals()
    return [
        AwaitingApprovalOut(
            build_id=step["build_id"],
            build_name=step["build_name"],
            build_status=step["build_status"],
            step_id=step["step_uid"],
            command=step["command"] or None,
            depends_on=json.loads(step["depends_on"]),
            status=step["status"],
            approval_status=step["approval_status"],
            approval_requested_at=step["approval_requested_at"],
            started_at=step["started_at"],
        )
        for step in steps[:limit]
    ]


@app.post(
    "/builds/{build_id}/steps/{step_id}/approval",
    response_model=ApprovalOut,
)
async def decide_step_approval(
    build_id: int, step_id: str, payload: ApprovalIn
) -> ApprovalOut:
    """对待签核门禁提交一次批准/拒绝结论。

    仅当前待签核门禁可受理一次结论（重复提交 409）；已取消或终态构建不受理。
    批准后调度器重新核对取消状态才放行后代，拒绝则后代跳过。
    """
    try:
        step = await scheduler.decide_approval(
            build_id,
            step_id,
            payload.decision,
            payload.decided_by,
            payload.comment,
        )
    except ApprovalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    approved = payload.decision == "approve"
    return ApprovalOut(
        build_id=build_id,
        step_id=step["step_uid"],
        decision=payload.decision,
        decision_result="approved" if approved else "rejected",
        status=step["status"],
        approval_status=step["approval_status"],
        approval_decided_at=step["approval_decided_at"],
        approval_decided_by=step["approval_decided_by"],
        approval_comment=step["approval_comment"],
        detail=(
            "已批准，调度器将核对构建未取消后放行后代"
            if approved
            else "已拒绝，后代步骤将跳过"
        ),
    )


@app.get("/builds/{build_id}", response_model=BuildOut)
async def get_build(build_id: int) -> BuildOut:
    return await _build_out(build_id)


@app.post("/builds/{build_id}/cancel", response_model=CancelOut)
async def cancel_build(build_id: int) -> CancelOut:
    row, message = await scheduler.cancel(build_id)
    if row is None:
        raise HTTPException(status_code=404, detail="构建不存在")
    # 重新读取以拿到 cancel_requested 的真实落库值（终态上的重复取消不会置位）
    fresh = await db.get_build(build_id)
    return CancelOut(
        id=row["id"],
        status=fresh["status"],
        cancel_requested=bool(fresh["cancel_requested"]),
        detail=message,
    )


@app.get("/builds/{build_id}/steps/{step_id}/logs", response_model=LogOut)
async def get_step_log(
    build_id: int,
    step_id: str,
    tail: int = Query(0, ge=0, description="只返回最后 N 行，0 表示返回全部（受字节上限约束）"),
) -> LogOut:
    step = await db.get_step(build_id, step_id)
    if step is None:
        raise HTTPException(status_code=404, detail="构建或步骤不存在")
    path = Path(step["log_path"])
    if not path.exists():
        return LogOut(build_id=build_id, step_id=step_id, status=step["status"],
                      truncated=False, content="")
    # 在线程池读文件，避免大日志阻塞事件循环
    raw = await asyncio.to_thread(path.read_bytes)
    truncated = False
    if len(raw) > config.LOG_MAX_BYTES:
        raw = raw[-config.LOG_MAX_BYTES:]
        truncated = True
    content = raw.decode("utf-8", errors="replace")
    if tail > 0:
        lines = content.splitlines()[-tail:]
        content = "\n".join(lines)
    return LogOut(
        build_id=build_id,
        step_id=step_id,
        status=step["status"],
        truncated=truncated,
        content=content,
    )
