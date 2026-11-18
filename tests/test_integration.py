"""端到端集成测试：以真实 uvicorn 子进程启动 API，覆盖全部关键链路。

运行:
  .venv/bin/pytest tests/test_integration.py -v
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerHarness:
    __test__ = False  # 非测试类，避免 pytest 误收集

    def __init__(self, data_dir: Path, port: int):
        self.data_dir = data_dir
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.log_file: Path | None = None
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30)

    def start(self) -> None:
        env = {
            **os.environ,
            "PIPELINE_DATA_DIR": str(self.data_dir),
            "PIPELINE_TERM_GRACE_SECONDS": "0.5",
            "PIPELINE_KILL_GRACE_SECONDS": "10",
            "PIPELINE_PROC_POLL_INTERVAL": "0.05",
            "PIPELINE_SCHEDULER_TICK": "0.05",
        }
        # 输出落文件而不是 PIPE：管道缓冲写满会冻结 uvicorn 事件循环
        self.log_file = self.data_dir / "server.log"
        log_fh = open(self.log_file, "ab")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--workers",
                "1",
            ],
            cwd=ROOT,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                r = httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1)
                if r.status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if self.proc.poll() is not None:
                out = self.log_file.read_text() if self.log_file else ""
                raise RuntimeError(f"服务器提前退出:\n{out}")
            time.sleep(0.1)
        raise RuntimeError(f"服务器启动超时，日志:\n{self.log_file.read_text()}")

    def stop(self, sig: int = signal.SIGTERM) -> None:
        if self.proc is None:
            return
        try:
            self.proc.send_signal(sig)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        finally:
            self.client.close()
            self.proc = None


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    srv = ServerHarness(tmp_path_factory.mktemp("data"), _free_port())
    srv.start()
    yield srv
    srv.stop()


def step(uid, command, deps=None, timeout=30.0):
    return {"id": uid, "command": command, "depends_on": deps or [],
            "timeout_seconds": timeout}


def submit(client: httpx.Client, name: str, steps: list[dict]) -> int:
    r = client.post("/builds", json={"name": name, "steps": steps})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def wait_build(client: httpx.Client, build_id: int, timeout: float = 40) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        detail = client.get(f"/builds/{build_id}").json()
        if detail["status"] in TERMINAL:
            return detail
        time.sleep(0.1)
    raise AssertionError(f"构建 {build_id} 未在 {timeout}s 内结束")


def step_map(detail: dict) -> dict:
    return {s["id"]: s for s in detail["steps"]}


def assert_pid_dead(pid: int) -> None:
    for _ in range(30):
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.1)
    raise AssertionError(f"进程 {pid} 在清理后仍然存活")


def command_output(log_payload: dict) -> str:
    """剔除 [runner] 元数据行，只保留命令自身的 stdout/stderr。"""
    return "\n".join(
        line for line in log_payload["content"].splitlines()
        if not line.startswith("[runner]")
    )


# --------------------------------------------------------------- 1. 成功构建

def test_success_build_runs_all_steps(server):
    bid = submit(server.client, "success", [
        step("prepare", "mkdir -p out && echo v1 > out/fw.txt"),
        step("compile", "test -f out/fw.txt && echo compiled", ["prepare"]),
        step("package", "echo packed", ["compile"]),
    ])
    detail = wait_build(server.client, bid)
    assert detail["status"] == "succeeded"
    assert all(s["status"] == "succeeded" for s in detail["steps"])
    artifact = server.data_dir / "workdir" / str(bid) / "out" / "fw.txt"
    assert artifact.read_text().strip() == "v1"
    log = server.client.get(f"/builds/{bid}/steps/compile/logs").json()
    assert "compiled" in log["content"]
    assert log["status"] == "succeeded"


# ----------------------------------------------------- 2. 失败跳过 + 独立分支

def test_failure_skips_descendants_but_independent_branch_continues(server):
    bid = submit(server.client, "failure", [
        step("good", "echo ok"),
        step("bad", "echo boom; exit 7"),
        step("child", "echo should-not-run", ["bad"]),
        step("grand", "echo should-not-run", ["child"]),
    ])
    detail = wait_build(server.client, bid)
    assert detail["status"] == "failed"
    sm = step_map(detail)
    assert sm["good"]["status"] == "succeeded"
    assert sm["bad"]["status"] == "failed" and sm["bad"]["exit_code"] == 7
    assert sm["child"]["status"] == "skipped"
    assert sm["grand"]["status"] == "skipped"
    log = server.client.get(f"/builds/{bid}/steps/child/logs").json()
    assert "should-not-run" not in command_output(log)  # 跳过的步骤确实没执行


# ------------------------------------------------- 3. 慢子进程：主退出后判失败

def test_lingering_child_is_killed_and_build_fails(server):
    bid = submit(server.client, "slow-child", [
        step(
            "spawn",
            # 不用子 shell 包装，$! 才能拿到真实后台子进程 PID
            "sleep 300 & echo $! > slow.pid && echo main-done && exit 0",
            timeout=10,
        ),
    ])
    detail = wait_build(server.client, bid)
    assert detail["status"] == "failed"
    sm = step_map(detail)
    assert sm["spawn"]["status"] == "failed"
    assert "遗留子进程" in (sm["spawn"]["detail"] or "")
    workdir = server.data_dir / "workdir" / str(bid)
    child_pid = int((workdir / "slow.pid").read_text().strip())
    assert_pid_dead(child_pid)  # 进程组清空前不确认终态：此时必须已死
    log = server.client.get(f"/builds/{bid}/steps/spawn/logs").json()
    assert "进程组" in log["content"] and "清空" in log["content"]


# ------------------------------------------------------------------- 4. 超时

def test_timeout_kills_group_and_marks_failed(server):
    bid = submit(server.client, "timeout", [
        step("slow", "echo start; sleep 30; echo end", timeout=1),
    ])
    detail = wait_build(server.client, bid, timeout=20)
    assert detail["status"] == "failed"
    assert step_map(detail)["slow"]["status"] == "failed"
    assert "超时" in (step_map(detail)["slow"]["detail"] or "")
    log = server.client.get(f"/builds/{bid}/steps/slow/logs").json()
    out = command_output(log)
    assert "start" in out
    assert "end" not in out  # 超时点在 sleep，后续输出不会产生


# ------------------------------------------------------ 5. 全局最多两步并行

def test_global_parallelism_capped_at_two(server):
    # 四个各 1.2s 的独立步骤，2 名额 -> 至少两批，总耗时应 >= 2.0s 且 < 4.5s
    steps = [step(f"s{i}", "echo running-$PIPELINE_STEP_ID; sleep 1.2") for i in range(4)]
    t0 = time.time()
    bid = submit(server.client, "parallel-cap", steps)
    detail = wait_build(server.client, bid, timeout=30)
    elapsed = time.time() - t0
    assert detail["status"] == "succeeded"
    assert elapsed >= 2.0, f"并行度疑似超过 2，总耗时仅 {elapsed:.2f}s"
    assert elapsed < 4.5


# ------------------------------------------------------------- 6. 整次取消

def test_cancel_stops_all_running_steps_and_skips_pending(server):
    bid = submit(server.client, "cancel", [
        step("a", "echo a-start; sleep 20; echo a-end"),
        step("b", "echo b-start; sleep 20; echo b-end"),
        step("c", "echo c-should-not-run; sleep 20", ["a"]),
    ])
    # 等两个名额占满
    deadline = time.time() + 10
    running = []
    while time.time() < deadline:
        sm = step_map(server.client.get(f"/builds/{bid}").json())
        running = [k for k, v in sm.items() if v["status"] == "running"]
        if len(running) == 2:
            break
        time.sleep(0.1)
    assert sorted(running) == ["a", "b"]

    r = server.client.post(f"/builds/{bid}/cancel")
    assert r.status_code == 200
    detail = wait_build(server.client, bid, timeout=20)
    sm = step_map(detail)
    assert detail["status"] == "cancelled"
    assert sm["a"]["status"] == "cancelled" and sm["b"]["status"] == "cancelled"
    assert sm["c"]["status"] == "cancelled"
    for uid in ("a", "b"):
        log = server.client.get(f"/builds/{bid}/steps/{uid}/logs").json()
        out = command_output(log)
        assert f"{uid}-start" in out
        assert f"{uid}-end" not in out  # 取消点在 sleep，尾部输出不会产生


def test_repeated_cancel_is_idempotent(server):
    bid = submit(server.client, "cancel-idem", [step("a", "sleep 20")])
    time.sleep(0.5)
    first = server.client.post(f"/builds/{bid}/cancel")
    assert first.status_code == 200
    detail1 = wait_build(server.client, bid, timeout=20)
    second = server.client.post(f"/builds/{bid}/cancel")
    assert second.status_code == 200
    assert "终态" in second.json()["detail"]
    third = server.client.get(f"/builds/{bid}").json()
    assert third["status"] == detail1["status"] == "cancelled"
    assert third["finished_at"] == detail1["finished_at"]  # 终态未被改写


# --------------------------------------- 7. 慢清理不阻塞另一名额（跨构建）

def test_slow_cleanup_does_not_block_other_slot(server):
    # 顽固子进程忽略 SIGTERM（忽略型信号处置会被子进程继承），
    # 主命令退出后清理必须升级到 SIGKILL 流程
    stubborn = (
        "trap '' TERM; sleep 300 & echo $! > stubborn.pid; "
        "echo main-exit; exit 0"
    )
    t0 = time.time()
    bid_busy = submit(server.client, "slow-cleanup", [step("x", stubborn, timeout=10)])
    time.sleep(0.6)  # 确保 busy 已占住一个名额
    bid_fast = submit(server.client, "other-slot", [step("y", "echo fast-ok")])
    detail = wait_build(server.client, bid_fast, timeout=15)
    fast_elapsed = time.time() - t0
    assert detail["status"] == "succeeded"
    # 另一名额可用：快构建在 5s 内完成（被慢清理拖住则要等 KILL 流程更久）
    assert fast_elapsed < 5, f"另一名额疑似被慢清理阻塞: {fast_elapsed:.2f}s"
    busy = wait_build(server.client, bid_busy, timeout=30)
    assert busy["status"] == "failed"  # 顽固子进程最终被 SIGKILL，仍判失败
    workdir = server.data_dir / "workdir" / str(bid_busy)
    pid = int((workdir / "stubborn.pid").read_text().strip())
    assert_pid_dead(pid)


# --------------------------------------------------------- 8. 重启恢复全链路

def test_restart_recovers_and_does_not_kill_new_build(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("restart-data")
    port = _free_port()
    srv = ServerHarness(data_dir, port)
    srv.start()
    try:
        bid = submit(srv.client, "will-be-interrupted", [
            step("a", "sleep 30"),
            step("b", "sleep 30"),
            step("c", "echo never", ["a"]),
        ])
        deadline = time.time() + 10
        while time.time() < deadline:
            sm = step_map(srv.client.get(f"/builds/{bid}").json())
            if sm["a"]["status"] == "running":
                running_pid = _step_pid(data_dir, bid, "a")
                break
            time.sleep(0.1)
        else:
            raise AssertionError("步骤未进入 running")
        assert Path(f"/proc/{running_pid}").exists()
        srv.stop()
        # 进程组随 API 被 tini 之外的场景保留（同会话 SIGTERM 只停 uvicorn）
        assert Path(f"/proc/{running_pid}").exists()
    finally:
        pass

    # 同数据目录重启：先清理残留，再标记 interrupted，然后才对外服务
    srv2 = ServerHarness(data_dir, port)
    srv2.start()
    try:
        assert_pid_dead(running_pid)  # 旧构建残留必须被清掉
        detail = srv2.client.get(f"/builds/{bid}").json()
        assert detail["status"] == "interrupted"
        sm = step_map(detail)
        assert sm["a"]["status"] == "interrupted"
        assert sm["b"]["status"] == "interrupted"
        assert sm["c"]["status"] == "interrupted"  # 未跑步骤不再启动

        # 后来的构建必须正常工作，不得被恢复逻辑误杀
        new_id = submit(srv2.client, "after-restart", [
            step("z", "echo alive && test $PIPELINE_BUILD_ID -ge 1"),
        ])
        new_detail = wait_build(srv2.client, new_id)
        assert new_detail["status"] == "succeeded"
    finally:
        srv2.stop()


def _step_pid(data_dir: Path, build_id: int, step_uid: str) -> int:
    import sqlite3

    conn = sqlite3.connect(data_dir / "pipeline.db")
    try:
        row = conn.execute(
            "SELECT pid FROM steps WHERE build_id=? AND step_uid=?",
            (build_id, step_uid),
        ).fetchone()
        return int(row[0])
    finally:
        conn.close()


# --------------------------------------------------------------- 9. 其他接口

def test_404_and_bad_payloads(server):
    assert server.client.get("/builds/999999").status_code == 404
    assert server.client.post("/builds/999999/cancel").status_code == 404

    bad_cycle = {"name": "cycle", "steps": [
        step("a", "true", ["b"]), step("b", "true", ["a"])
    ]}
    r = server.client.post("/builds", json=bad_cycle)
    assert r.status_code == 400 and "循环" in r.json()["detail"]

    missing_dep = {"name": "ghost", "steps": [step("a", "true", ["x"])]}
    r = server.client.post("/builds", json=missing_dep)
    assert r.status_code == 400 and "不存在" in r.json()["detail"]

    empty = {"name": "empty", "steps": []}
    assert server.client.post("/builds", json=empty).status_code == 422


def test_health_and_docs(server):
    health = server.client.get("/health").json()
    assert health["status"] == "ok" and health["max_parallel"] == 2
    assert server.client.get("/docs").status_code == 200
    assert server.client.get("/openapi.json").status_code == 200


# ------------------------------------------------------- 10. 人工放行门禁

def _wait_awaiting(client: httpx.Client, build_id: int, step_uid: str,
                   timeout: float = 20) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        detail = client.get(f"/builds/{build_id}").json()
        sm = step_map(detail)
        if sm.get(step_uid, {}).get("status") == "awaiting_approval":
            return detail
        time.sleep(0.05)
    raise AssertionError(f"门禁 {step_uid} 未进入待签核")


def _approve(client: httpx.Client, build_id: int, step_uid: str,
             decision: str, **extra) -> httpx.Response:
    return client.post(
        f"/builds/{build_id}/steps/{step_uid}/approval",
        json={"decision": decision, **extra},
    )


def test_gate_blocks_descendants_until_approved(server):
    bid = submit(server.client, "gate-approve", [
        step("prepare", "echo prepared > artifact.txt"),
        {**step("signoff", "test -f artifact.txt && echo report-ok", ["prepare"]),
         "approval_required": True},
        step("package", "test -f artifact.txt && echo packaged", ["signoff"]),
    ])
    detail = _wait_awaiting(server.client, bid, "signoff")
    sm = step_map(detail)
    assert detail["status"] == "running"
    assert sm["signoff"]["approval_required"] is True
    assert sm["signoff"]["approval_status"] == "pending"
    assert sm["prepare"]["status"] == "succeeded"
    assert sm["package"]["status"] == "pending"  # 批准前后代不得启动

    listing = server.client.get("/approvals").json()
    entry = next(e for e in listing if e["build_id"] == bid)
    assert entry["step_id"] == "signoff" and entry["build_status"] == "running"
    assert entry["command"].startswith("test -f artifact.txt")

    r = _approve(server.client, bid, "signoff", "approve",
                 decided_by="qa-1", comment="测试通过")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["approval_status"] == "approved"
    assert body["approval_decided_by"] == "qa-1"

    detail = wait_build(server.client, bid, timeout=30)
    sm = step_map(detail)
    assert detail["status"] == "succeeded"
    assert sm["signoff"]["status"] == "succeeded"
    assert sm["package"]["status"] == "succeeded"
    log = server.client.get(f"/builds/{bid}/steps/signoff/logs").json()
    assert "report-ok" in log["content"]
    assert "批准签核" in log["content"]


def test_gate_reject_skips_descendants(server):
    bid = submit(server.client, "gate-reject", [
        step("build", "echo built"),
        {**step("signoff", "true", ["build"]), "approval_required": True},
        step("package", "echo should-not-run", ["signoff"]),
    ])
    _wait_awaiting(server.client, bid, "signoff")
    r = _approve(server.client, bid, "signoff", "reject",
                 decided_by="qa-2", comment="质检不合格")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "rejected"
    detail = wait_build(server.client, bid, timeout=30)
    sm = step_map(detail)
    assert detail["status"] == "failed"
    assert sm["signoff"]["status"] == "rejected"
    assert sm["signoff"]["approval_status"] == "rejected"
    assert sm["package"]["status"] == "skipped"
    log = server.client.get(f"/builds/{bid}/steps/package/logs").json()
    assert "should-not-run" not in command_output(log)


def test_gate_only_one_decision_accepted(server):
    bid = submit(server.client, "gate-once", [
        {**step("gate", "true"), "approval_required": True},
        step("after", "echo done", ["gate"]),
    ])
    _wait_awaiting(server.client, bid, "gate")
    first = _approve(server.client, bid, "gate", "approve")
    assert first.status_code == 200
    second = _approve(server.client, bid, "gate", "reject")
    assert second.status_code == 409
    detail = wait_build(server.client, bid, timeout=30)
    assert detail["status"] == "succeeded"
    assert step_map(detail)["gate"]["approval_status"] == "approved"


def test_gate_rejects_bad_requests(server):
    bid = submit(server.client, "gate-errors", [
        {**step("gate", "true"), "approval_required": True},
    ])
    _wait_awaiting(server.client, bid, "gate")
    assert _approve(server.client, 999999, "gate", "approve").status_code == 404
    assert _approve(server.client, bid, "ghost", "approve").status_code == 404
    r = server.client.post(
        f"/builds/{bid}/steps/gate/approval", json={"decision": "maybe"}
    )
    assert r.status_code == 422
    # 非门禁步骤上的签核不受理
    bid2 = submit(server.client, "gate-none", [step("plain", "true")])
    wait_build(server.client, bid2)
    assert _approve(server.client, bid2, "plain", "approve").status_code == 409


def test_cancel_while_awaiting_approval_blocks_later_approval(server):
    bid = submit(server.client, "gate-cancel", [
        {**step("gate", "true"), "approval_required": True},
        step("after", "echo should-not-run", ["gate"]),
    ])
    _wait_awaiting(server.client, bid, "gate")
    r = server.client.post(f"/builds/{bid}/cancel")
    assert r.status_code == 200
    detail = wait_build(server.client, bid, timeout=30)
    assert detail["status"] == "cancelled"
    sm = step_map(detail)
    assert sm["gate"]["status"] == "cancelled"
    assert sm["gate"]["approval_status"] == "cancelled"
    assert sm["after"]["status"] == "cancelled"
    # 取消后签核不再受理，且不得因迟到的批准补跑后代
    blocked = _approve(server.client, bid, "gate", "approve")
    assert blocked.status_code == 409
    log = server.client.get(f"/builds/{bid}/steps/after/logs").json()
    assert "should-not-run" not in command_output(log)
    detail = server.client.get(f"/builds/{bid}").json()
    assert step_map(detail)["gate"]["status"] == "cancelled"


def test_cancel_right_after_approval_does_not_start_released_descendant(server):
    bid = submit(server.client, "gate-approve-cancel", [
        {**step("gate", "true"), "approval_required": True},
        # 先睡再写标记：若后代被启动，取消必须在其写标记前终止整个进程组
        step("after", "sleep 0.5; echo released > released.txt; sleep 30", ["gate"]),
    ])
    _wait_awaiting(server.client, bid, "gate")
    assert _approve(server.client, bid, "gate", "approve").status_code == 200
    # 批准后立刻取消：已取消构建的后代命令不得真正产出，已启动的必须被终止
    assert server.client.post(f"/builds/{bid}/cancel").status_code == 200
    detail = wait_build(server.client, bid, timeout=30)
    assert detail["status"] == "cancelled"
    assert step_map(detail)["after"]["status"] == "cancelled"
    marker = server.data_dir / "workdir" / str(bid) / "released.txt"
    assert not marker.exists(), "已取消构建的后代命令被启动了"


def test_gate_waiting_does_not_occupy_execution_slots(server):
    # 门禁等待签核时不得占用两个名额：两个独立构建仍应并行执行并完成
    gate_bid = submit(server.client, "gate-holding", [
        {**step("gate", "true"), "approval_required": True},
    ])
    _wait_awaiting(server.client, gate_bid, "gate")
    t0 = time.time()
    two = [
        submit(server.client, f"slot-{i}", [step("s", "sleep 1")])
        for i in range(2)
    ]
    for bid in two:
        wait_build(server.client, bid, timeout=15)
    elapsed = time.time() - t0
    assert elapsed < 3.5, f"门禁占用名额，两个独立构建耗时 {elapsed:.2f}s"
    # 独立分支照常运行；门禁构建仍待签核
    listing = server.client.get("/approvals").json()
    assert any(e["build_id"] == gate_bid for e in listing)
    assert _approve(server.client, gate_bid, "gate", "reject").status_code == 200
    assert wait_build(server.client, gate_bid)["status"] == "failed"


def test_restart_expires_pending_approval_without_rerun(tmp_path_factory):
    """旧签核结论已在库中但后代尚未放行时，重启不得让后代补跑。"""
    data_dir = tmp_path_factory.mktemp("gate-restart-data")
    port = _free_port()
    srv = ServerHarness(data_dir, port)
    srv.start()
    try:
        bid = submit(srv.client, "gate-restart", [
            step("prep", "echo prepared > prep.txt"),
            {**step("gate", "true", ["prep"]), "approval_required": True},
            step("package", "echo should-not-run > package-ran.txt", ["gate"]),
        ])
        _wait_awaiting(srv.client, bid, "gate")
    finally:
        srv.stop()

    # 在停机状态下模拟"旧签核已受理但后代尚未启动"：直接改写库中门禁为已批准。
    import sqlite3
    now = "2026-01-01T00:00:00+00:00"
    conn = sqlite3.connect(data_dir / "pipeline.db")
    try:
        conn.execute(
            "UPDATE steps SET status='succeeded', approval_status='approved', "
            "approval_decided_at=?, approval_decided_by='old-qa' "
            "WHERE build_id=? AND step_uid='gate'",
            (now, bid),
        )
        conn.commit()
    finally:
        conn.close()

    srv2 = ServerHarness(data_dir, port)
    srv2.start()
    try:
        detail = srv2.client.get(f"/builds/{bid}").json()
        assert detail["status"] == "interrupted"
        sm = step_map(detail)
        # 门禁自身已有结论（批准），重启不改写其终态；关键是后代不得补跑
        assert sm["gate"]["status"] == "succeeded"
        assert sm["gate"]["approval_status"] == "approved"
        assert sm["package"]["status"] == "interrupted"
        # 旧签核不得使步骤补跑；重启后签核不再受理
        assert _approve(srv2.client, bid, "gate", "approve").status_code == 409
        marker = data_dir / "workdir" / str(bid) / "package-ran.txt"
        assert not marker.exists()
        log = srv2.client.get(f"/builds/{bid}/steps/package/logs").json()
        assert "should-not-run" not in command_output(log)
    finally:
        srv2.stop()

    # 恢复后的新构建仍可正常走门禁
    srv3 = ServerHarness(data_dir, port)
    srv3.start()
    try:
        new_id = submit(srv3.client, "gate-after-restart", [
            {**step("g2", "true"), "approval_required": True},
            step("tail", "echo tail-ran", ["g2"]),
        ])
        _wait_awaiting(srv3.client, new_id, "g2")
        assert _approve(srv3.client, new_id, "g2", "approve").status_code == 200
        assert wait_build(srv3.client, new_id)["status"] == "succeeded"
    finally:
        srv3.stop()


def test_restart_marks_unresolved_pending_approval_expired(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("gate-expire-data")
    port = _free_port()
    srv = ServerHarness(data_dir, port)
    srv.start()
    try:
        bid = submit(srv.client, "gate-expire", [
            {**step("gate", "true"), "approval_required": True},
        ])
        _wait_awaiting(srv.client, bid, "gate")
    finally:
        srv.stop()

    srv2 = ServerHarness(data_dir, port)
    srv2.start()
    try:
        detail = srv2.client.get(f"/builds/{bid}").json()
        sm = step_map(detail)
        assert sm["gate"]["status"] == "interrupted"
        assert sm["gate"]["approval_status"] == "expired"
        assert srv2.client.get("/approvals").json() == []
    finally:
        srv2.stop()


def test_old_database_is_migrated_with_approval_columns(tmp_path_factory):
    """升级前创建的库缺少签核列，启动时必须自动补齐且不丢旧数据。"""
    import sqlite3

    data_dir = tmp_path_factory.mktemp("legacy-data")
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_dir / "pipeline.db")
    try:
        conn.executescript(
            """
            CREATE TABLE builds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL, status TEXT NOT NULL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                detail TEXT, created_at TEXT NOT NULL,
                started_at TEXT, finished_at TEXT
            );
            CREATE TABLE steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                build_id INTEGER NOT NULL, step_uid TEXT NOT NULL,
                command TEXT NOT NULL, depends_on TEXT NOT NULL,
                timeout_seconds REAL NOT NULL, status TEXT NOT NULL,
                exit_code INTEGER, detail TEXT, log_path TEXT,
                pid INTEGER, pgid INTEGER, proc_starttime REAL,
                started_at TEXT, finished_at TEXT,
                UNIQUE(build_id, step_uid),
                FOREIGN KEY(build_id) REFERENCES builds(id)
            );
            INSERT INTO builds (name, status, created_at)
                VALUES ('legacy', 'succeeded', '2026-01-01T00:00:00+00:00');
            INSERT INTO steps
                (build_id, step_uid, command, depends_on, timeout_seconds,
                 status, log_path)
                VALUES (1, 'old', 'true', '[]', 5, 'succeeded', '/tmp/old.log');
            """
        )
        conn.commit()
    finally:
        conn.close()

    srv = ServerHarness(data_dir, _free_port())
    srv.start()
    try:
        detail = srv.client.get("/builds/1").json()
        assert detail["status"] == "succeeded"
        step_out = step_map(detail)["old"]
        assert step_out["approval_required"] is False
        assert step_out["approval_status"] == "none"
        new_id = submit(srv.client, "post-migration", [
            {**step("gate", "true"), "approval_required": True},
        ])
        _wait_awaiting(srv.client, new_id, "gate")
        assert _approve(srv.client, new_id, "gate", "approve").status_code == 200
        assert wait_build(srv.client, new_id)["status"] == "succeeded"
    finally:
        srv.stop()


def test_openapi_documents_approval_endpoints(server):
    spec = server.client.get("/openapi.json").json()
    assert "/approvals" in spec["paths"]
    assert "/builds/{build_id}/steps/{step_id}/approval" in spec["paths"]
    step_schema = spec["components"]["schemas"]["StepIn"]
    assert "approval_required" in step_schema["properties"]
    approval_schema = spec["components"]["schemas"]["ApprovalIn"]
    assert set(approval_schema["properties"]) >= {"decision", "decided_by", "comment"}
