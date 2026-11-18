"""SQLite 持久化层（aiosqlite，单连接 + WAL）。"""
from __future__ import annotations

import json
from typing import Any, Optional

import aiosqlite

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS builds (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    name             TEXT NOT NULL,
    status           TEXT NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    detail           TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    build_id        INTEGER NOT NULL,
    step_uid        TEXT NOT NULL,
    command         TEXT NOT NULL,
    depends_on      TEXT NOT NULL,
    timeout_seconds REAL NOT NULL,
    status          TEXT NOT NULL,
    exit_code       INTEGER,
    detail          TEXT,
    log_path        TEXT,
    pid             INTEGER,
    pgid            INTEGER,
    proc_starttime  REAL,
    started_at      TEXT,
    finished_at     TEXT,
    approval_required   INTEGER NOT NULL DEFAULT 0,
    approval_status     TEXT NOT NULL DEFAULT 'none',
    approval_requested_at TEXT,
    approval_decided_at   TEXT,
    approval_decided_by   TEXT,
    approval_comment      TEXT,
    UNIQUE(build_id, step_uid),
    FOREIGN KEY(build_id) REFERENCES builds(id)
);
"""

# 旧库升级：CREATE TABLE IF NOT EXISTS 不会补列，启动时按需 ALTER
_MIGRATIONS = {
    "approval_required": "INTEGER NOT NULL DEFAULT 0",
    "approval_status": "TEXT NOT NULL DEFAULT 'none'",
    "approval_requested_at": "TEXT",
    "approval_decided_at": "TEXT",
    "approval_decided_by": "TEXT",
    "approval_comment": "TEXT",
}


class Database:
    def __init__(self) -> None:
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        config.ensure_dirs()
        self._conn = await aiosqlite.connect(
            config.DB_PATH, timeout=30, isolation_level=None
        )
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=30000")
        await self._conn.executescript(_SCHEMA)
        await self._migrate()

    async def _migrate(self) -> None:
        """旧库缺少签核列时补齐，保证升级后可直接启动。"""
        assert self._conn is not None
        cur = await self._conn.execute("PRAGMA table_info(steps)")
        existing = {row[1] for row in await cur.fetchall()}
        for column, ddl in _MIGRATIONS.items():
            if column not in existing:
                await self._conn.execute(
                    f"ALTER TABLE steps ADD COLUMN {column} {ddl}"
                )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_steps_approval "
            "ON steps (approval_required, approval_status, status)"
        )

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        assert self._conn is not None
        cur = await self._conn.execute(sql, params)
        return cur.lastrowid

    async def one(self, sql: str, params: tuple[Any, ...] = ()) -> Optional[aiosqlite.Row]:
        assert self._conn is not None
        cur = await self._conn.execute(sql, params)
        return await cur.fetchone()

    async def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        assert self._conn is not None
        cur = await self._conn.execute(sql, params)
        return list(await cur.fetchall())

    # ---- 业务封装 -------------------------------------------------------

    async def create_build(self, name: str, now: str) -> int:
        return await self.execute(
            "INSERT INTO builds (name, status, created_at) VALUES (?, 'pending', ?)",
            (name, now),
        )

    async def create_step(
        self,
        build_id: int,
        step_uid: str,
        command: str,
        depends_on: list[str],
        timeout_seconds: float,
        log_path: str,
        approval_required: bool = False,
    ) -> None:
        await self.execute(
            """
            INSERT INTO steps
                (build_id, step_uid, command, depends_on, timeout_seconds,
                 status, log_path, approval_required, approval_status)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, 'none')
            """,
            (
                build_id,
                step_uid,
                command,
                json.dumps(depends_on),
                timeout_seconds,
                log_path,
                int(approval_required),
            ),
        )

    async def get_build(self, build_id: int) -> Optional[aiosqlite.Row]:
        return await self.one("SELECT * FROM builds WHERE id=?", (build_id,))

    async def list_builds(self, limit: int = 100) -> list[aiosqlite.Row]:
        return await self.all(
            "SELECT * FROM builds ORDER BY id DESC LIMIT ?", (limit,)
        )

    async def get_steps(self, build_id: int) -> list[aiosqlite.Row]:
        return await self.all(
            "SELECT * FROM steps WHERE build_id=? ORDER BY id", (build_id,)
        )

    async def get_step(self, build_id: int, step_uid: str) -> Optional[aiosqlite.Row]:
        return await self.one(
            "SELECT * FROM steps WHERE build_id=? AND step_uid=?",
            (build_id, step_uid),
        )

    async def set_build_status(
        self,
        build_id: int,
        status: str,
        now: str,
        detail: Optional[str] = None,
        started: bool = False,
        finished: bool = False,
    ) -> None:
        sets = ["status=?"]
        params: list[Any] = [status]
        if detail is not None:
            sets.append("detail=?")
            params.append(detail)
        if started:
            sets.append("started_at=COALESCE(started_at, ?)")
            params.append(now)
        if finished:
            sets.append("finished_at=?")
            params.append(now)
        params.append(build_id)
        await self.execute(f"UPDATE builds SET {', '.join(sets)} WHERE id=?", tuple(params))

    async def request_cancel(self, build_id: int) -> None:
        await self.execute(
            "UPDATE builds SET cancel_requested=1 WHERE id=?", (build_id,)
        )

    async def set_step_process(
        self, step_pk: int, pid: int, pgid: int, starttime: float, now: str
    ) -> None:
        await self.execute(
            "UPDATE steps SET pid=?, pgid=?, proc_starttime=?, status='running', "
            "started_at=COALESCE(started_at, ?) WHERE id=?",
            (pid, pgid, starttime, now, step_pk),
        )

    async def mark_awaiting_approval(
        self, step_pk: int, now: str, from_statuses: tuple[str, ...] = ("running",)
    ) -> bool:
        """把步骤推进到待签核。

        用条件更新保证只有仍处于调用方预期状态的步骤能进入待签核：并发的取消
        已把步骤置为 cancelled 时返回 False，调用方不得再写 awaiting_approval。
        纯门禁步骤没有命令，直接由 pending 进入待签核。
        """
        assert self._conn is not None
        placeholders = ",".join("?" * len(from_statuses))
        cur = await self._conn.execute(
            "UPDATE steps SET status='awaiting_approval', approval_status='pending', "
            "approval_requested_at=?, finished_at=NULL, exit_code=0, "
            "detail='人工放行门禁等待签核' "
            f"WHERE id=? AND status IN ({placeholders})",
            (now, step_pk, *from_statuses),
        )
        return cur.rowcount == 1

    async def get_step_by_pk(self, step_pk: int) -> Optional[aiosqlite.Row]:
        return await self.one("SELECT * FROM steps WHERE id=?", (step_pk,))

    async def list_awaiting_approval_steps(
        self, build_id: Optional[int] = None
    ) -> list[aiosqlite.Row]:
        sql = (
            "SELECT * FROM steps WHERE approval_required=1 "
            "AND approval_status='pending' AND status='awaiting_approval'"
        )
        params: tuple[Any, ...] = ()
        if build_id is not None:
            sql += " AND build_id=?"
            params = (build_id,)
        return await self.all(sql + " ORDER BY build_id, id", params)

    async def decide_approval(
        self,
        step_pk: int,
        decision: str,
        now: str,
        decided_by: Optional[str],
        comment: Optional[str],
        next_status: str,
    ) -> bool:
        """原子地提交一次签核结论。

        仅当步骤仍为待签核（approval_status='pending' 且 status='awaiting_approval'）
        时生效，天然实现"只受理一次结论"；重复请求返回 False 由路由层转 409。
        批准把步骤落为 succeeded、拒绝落为 rejected，均写 finished_at。
        """
        approval_status = "approved" if decision == "approve" else "rejected"
        detail = (
            f"人工门禁已批准 by={decided_by or '-'}"
            if decision == "approve"
            else f"人工门禁已拒绝 by={decided_by or '-'}，后代跳过"
        )
        cur = await self._conn.execute(
            """
            UPDATE steps
            SET approval_status=?, approval_decided_at=?, approval_decided_by=?,
                approval_comment=?, status=?, finished_at=?, detail=?
            WHERE id=? AND approval_status='pending' AND status='awaiting_approval'
            """,
            (
                approval_status,
                now,
                decided_by,
                comment,
                next_status,
                now,
                detail,
                step_pk,
            ),
        )
        return cur.rowcount == 1

    async def set_step_status(
        self,
        step_pk: int,
        status: str,
        now: str,
        exit_code: Optional[int] = None,
        detail: Optional[str] = None,
    ) -> None:
        from .models import STEP_TERMINAL

        sets = ["status=?"]
        params: list[Any] = [status]
        # finished_at 只在进入终态时落笔；cleaning_up 等中间态不写
        if status in STEP_TERMINAL:
            sets.append("finished_at=?")
            params.append(now)
        if exit_code is not None:
            sets.append("exit_code=?")
            params.append(exit_code)
        if detail is not None:
            sets.append("detail=?")
            params.append(detail)
        params.append(step_pk)
        await self.execute(f"UPDATE steps SET {', '.join(sets)} WHERE id=?", tuple(params))

    async def cancel_awaiting_step(
        self, step_pk: int, now: str, detail: str
    ) -> None:
        """取消待签核门禁：状态落 cancelled，未结论的签核请求一并作废。"""
        await self.execute(
            "UPDATE steps SET status='cancelled', "
            "approval_status=CASE WHEN approval_status='pending' "
            "THEN 'cancelled' ELSE approval_status END, "
            "detail=?, finished_at=? WHERE id=? AND status='awaiting_approval'",
            (detail, now, step_pk),
        )

    async def expire_approval(self, step_pk: int, now: str) -> None:
        """重启恢复时把待签核请求作废；旧结论不得在恢复后使步骤补跑。"""
        await self.execute(
            "UPDATE steps SET approval_status='expired', approval_decided_at=?, "
            "approval_comment=COALESCE(approval_comment, 'API 重启，签核请求作废') "
            "WHERE id=? AND approval_status='pending'",
            (now, step_pk),
        )

    async def list_nonterminal_builds(self) -> list[aiosqlite.Row]:
        return await self.all(
            "SELECT * FROM builds WHERE status NOT IN ('succeeded','failed','cancelled','interrupted')"
        )

    async def list_steps_for_builds(self, build_ids: tuple[int, ...]) -> list[aiosqlite.Row]:
        if not build_ids:
            return []
        placeholders = ",".join("?" * len(build_ids))
        return await self.all(
            f"SELECT * FROM steps WHERE build_id IN ({placeholders}) ORDER BY id",
            build_ids,
        )
