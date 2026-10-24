"""작업 상태 머신 (State 패턴의 전이 테이블 버전).

REQUIREMENTS 11.1:
created → validating → queued → running → succeeded | failed | cancelled
학습은 running → exporting → validating_adapter → succeeded 추가.
"""

from __future__ import annotations

from app.core.errors import InvalidStateTransition
from app.domain.enums import JobStatus as S

_TRANSITIONS: dict[S, set[S]] = {
    S.CREATED: {S.VALIDATING, S.QUEUED, S.FAILED, S.CANCELLED},
    S.VALIDATING: {S.QUEUED, S.FAILED, S.CANCELLED},
    S.QUEUED: {S.RUNNING, S.FAILED, S.CANCELLED},
    S.RUNNING: {S.SUCCEEDED, S.FAILED, S.CANCELLED, S.EXPORTING},
    S.EXPORTING: {S.VALIDATING_ADAPTER, S.FAILED, S.CANCELLED},
    S.VALIDATING_ADAPTER: {S.SUCCEEDED, S.FAILED, S.CANCELLED},
    S.SUCCEEDED: set(),
    S.FAILED: set(),
    S.CANCELLED: set(),
}


def can_transition(current: S, target: S) -> bool:
    return target in _TRANSITIONS[current]


def ensure_transition(current: S, target: S) -> None:
    if not can_transition(current, target):
        raise InvalidStateTransition(detail=f"{current.value} → {target.value}")
