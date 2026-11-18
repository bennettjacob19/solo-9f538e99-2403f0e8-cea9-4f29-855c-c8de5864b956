"""请求/响应模型与 DAG 校验。"""
from __future__ import annotations

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from . import config

# 步骤与构建的状态机
# pending 未启动 / running 运行中 / cleaning_up 主进程已退出，正在清空进程组
# awaiting_approval 人工门禁等待签核，不占用执行名额
# rejected 人工门禁被拒绝，后代按失败向上游规则跳过
# succeeded / failed / skipped / cancelled / interrupted / rejected 均为终态
STEP_TERMINAL = {
    "succeeded", "failed", "skipped", "cancelled", "interrupted", "rejected"
}
BUILD_TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}

# 取消/超时后清理时采用的终态
CANCEL_TERMINAL = "cancelled"
TIMEOUT_TERMINAL = "failed"


class StepIn(BaseModel):
    id: Annotated[str, Field(min_length=1, max_length=128)]
    command: Optional[Annotated[str, Field(min_length=1, max_length=8192)]] = None
    depends_on: list[str] = Field(default_factory=list)
    timeout_seconds: Annotated[float, Field(gt=0, le=config.TIMEOUT_CAP_SECONDS)]
    # 人工放行门禁：命令跑完后不直接判成功，而是进入 awaiting_approval
    # 等待签核；只有批准后后代才会被调度。
    approval_required: bool = False

    @field_validator("id", "command")
    @classmethod
    def _no_surrounding_whitespace(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != v.strip():
            raise ValueError("字符串首尾不允许空白字符")
        return v

    @field_validator("depends_on")
    @classmethod
    def _unique_deps(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("依赖列表存在重复")
        return v

    @model_validator(mode="after")
    def _command_consistency(self) -> "StepIn":
        # 普通步骤必须提供命令；门禁步骤（approval_required）命令可省略，
        # 省略时仅做签核门禁，不执行任何进程。
        if self.command is None and not self.approval_required:
            raise ValueError("非门禁步骤必须提供 command")
        return self


class BuildCreate(BaseModel):
    name: Annotated[str, Field(min_length=1, max_length=128)]
    steps: Annotated[list[StepIn], Field(min_length=1, max_length=config.MAX_STEPS)]


class StepOut(BaseModel):
    id: str
    command: Optional[str]
    depends_on: list[str]
    timeout_seconds: float
    status: str
    exit_code: Optional[int]
    detail: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    approval_required: bool = False
    approval_status: str = "none"
    approval_requested_at: Optional[str] = None
    approval_decided_at: Optional[str] = None
    approval_decided_by: Optional[str] = None
    approval_comment: Optional[str] = None


class ApprovalIn(BaseModel):
    decision: Literal["approve", "reject"]
    decided_by: Optional[Annotated[str, Field(min_length=1, max_length=128)]] = None
    comment: Optional[Annotated[str, Field(max_length=2000)]] = None

    @field_validator("decided_by", "comment")
    @classmethod
    def _strip_optional(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        return v or None


class ApprovalOut(BaseModel):
    build_id: int
    step_id: str
    decision: Literal["approve", "reject"]
    decision_result: str
    status: str
    approval_status: str
    approval_decided_at: str
    approval_decided_by: Optional[str]
    approval_comment: Optional[str]
    detail: str


class AwaitingApprovalOut(BaseModel):
    """待签核门禁视图：显式列出构建与门禁步骤，便于质检员轮询。"""

    build_id: int
    build_name: str
    build_status: str
    step_id: str
    command: Optional[str]
    depends_on: list[str]
    status: str
    approval_status: str
    approval_requested_at: Optional[str]
    started_at: Optional[str]


class BuildOut(BaseModel):
    id: int
    name: str
    status: str
    cancel_requested: bool
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    detail: Optional[str]
    steps: list[StepOut]


class LogOut(BaseModel):
    build_id: int
    step_id: str
    status: str
    truncated: bool
    content: str


class CancelOut(BaseModel):
    id: int
    status: str
    cancel_requested: bool
    detail: str


def validate_dag(steps: list[StepIn]) -> None:
    """结构之外的 DAG 语义校验：重复 id、悬空依赖、自依赖、有环。"""
    ids = [s.id for s in steps]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"步骤 id 重复: {sorted(duplicates)}")

    id_set = set(ids)
    for step in steps:
        for dep in step.depends_on:
            if dep == step.id:
                raise ValueError(f"步骤 {step.id} 不能依赖自身")
            if dep not in id_set:
                raise ValueError(f"步骤 {step.id} 依赖了不存在的步骤 {dep}")

    # Kahn 拓扑排序检测有向环
    indegree = {s.id: len(set(s.depends_on)) for s in steps}
    adjacency: dict[str, list[str]] = {s.id: [] for s in steps}
    for step in steps:
        for dep in step.depends_on:
            adjacency[dep].append(step.id)

    ready = [sid for sid, deg in indegree.items() if deg == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if visited != len(steps):
        raise ValueError("DAG 存在循环依赖")
