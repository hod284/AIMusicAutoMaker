"""잡 매니저 — 모든 비동기 작업의 단일 진입점.

패턴:
- **Command**: ``JobHandler`` 하나가 작업 종류 하나를 캡슐화. 새 작업 = 핸들러 클래스 추가 + register.
- **Template Method**: ``JobManager._execute`` 가 validating → queued → (GPU 락) → running → 종료
  골격을 고정하고, 핸들러는 ``validate`` / ``run`` 만 구현한다.
- **Observer**: 상태/진행률/로그 변화를 ``EventBus`` 로 발행 → ViewModel 이 구독.
- **State**: 상태 전이는 ``job_state.ensure_transition`` 으로만 수행.

장시간 연산 자체는 외부 프로세스(ACE 서버, MLX 서브프로세스)에서 일어나므로
NiceGUI 이벤트 루프를 막지 않는다.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.core.errors import StudioError, ValidationError
from app.core.observer import EventBus, Topics
from app.domain.entities import Job, JobEvent, utcnow
from app.domain.enums import JobKind, JobStatus
from app.domain.job_state import ensure_transition
from app.repositories.job_repo import JobRepository
from app.services.resource_lock import ResourceLock

log = logging.getLogger(__name__)


class JobCancelled(Exception):
    pass


class JobContext:
    """핸들러에 전달되는 실행 컨텍스트. 로그·진행률·취소 확인을 제공한다."""

    def __init__(self, manager: "JobManager", job: Job) -> None:
        self._manager = manager
        self.job = job
        self.cancel_event = asyncio.Event()

    @property
    def params(self) -> dict[str, Any]:
        return self.job.params

    def log(self, message: str, level: str = "info", **details: Any) -> None:
        self._manager.log_event(self.job, message, level, details)

    def progress(self, value: float | None) -> None:
        """0.0~1.0. 알 수 없으면 None (인위적 100% 금지)."""
        if value is not None:
            value = max(0.0, min(1.0, value))
            if value >= 1.0:
                value = 0.99  # 100% 는 '검증 완료 후 성공' 시에만 매니저가 설정
        self.job.progress = value
        self.job.heartbeat_at = utcnow()
        self._manager.persist(self.job)

    def heartbeat(self) -> None:
        self.job.heartbeat_at = utcnow()
        self._manager.persist(self.job, publish=False)

    def set_status(self, status: JobStatus) -> None:
        """학습 전용 중간 상태(exporting/validating_adapter) 전환."""
        self._manager.transition(self.job, status)

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise JobCancelled()


class JobHandler(ABC):
    kind: JobKind

    def validate(self, params: dict[str, Any]) -> None:
        """제출 시점 동기 검증. 실패 시 ValidationError."""

    @abstractmethod
    async def run(self, ctx: JobContext) -> dict[str, Any]:
        """작업 본문. 반환값은 job.result 로 저장된다."""

    def describe(self, params: dict[str, Any]) -> str:
        return self.kind.value


class JobManager:
    def __init__(self, repo: JobRepository, lock: ResourceLock, bus: EventBus | None = None) -> None:
        self.repo = repo
        self.lock = lock
        self.bus = bus or EventBus()
        self._handlers: dict[JobKind, JobHandler] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._contexts: dict[str, JobContext] = {}

    # --- 등록 ---
    def register(self, handler: JobHandler) -> None:
        self._handlers[handler.kind] = handler

    def supports(self, kind: JobKind) -> bool:
        return kind in self._handlers

    # --- 영속화/이벤트 ---
    def persist(self, job: Job, publish: bool = True) -> None:
        self.repo.save(job)
        if publish:
            self.bus.publish(Topics.JOB_UPDATED, job=job.model_copy())

    def transition(self, job: Job, status: JobStatus) -> None:
        ensure_transition(job.status, status)
        job.status = status
        if status == JobStatus.RUNNING and job.started_at is None:
            job.started_at = utcnow()
        if status.is_terminal:
            job.ended_at = utcnow()
        self.persist(job)

    def log_event(self, job: Job, message: str, level: str = "info", details: dict[str, Any] | None = None) -> None:
        event = self.repo.add_event(JobEvent(job_id=job.id, level=level, message=message, details=details or {}))
        getattr(log, level if level in ("debug", "info", "warning", "error") else "info")("[job %s] %s", job.id[:8], message)
        self.bus.publish(Topics.JOB_LOG, job_id=job.id, event=event)

    # --- 제출/실행 ---
    def submit(self, kind: JobKind, params: dict[str, Any] | None = None, *, category_id: str | None = None,
               dataset_id: str | None = None, model_id: str | None = None) -> Job:
        handler = self._handlers.get(kind)
        if handler is None:
            raise ValidationError(f"지원하지 않는 작업 종류: {kind.value}")
        job = Job(kind=kind, params=params or {}, category_id=category_id, dataset_id=dataset_id, model_id=model_id)
        self.repo.add(job)
        self.bus.publish(Topics.JOB_CREATED, job=job.model_copy())
        ctx = JobContext(self, job)
        self._contexts[job.id] = ctx
        self._tasks[job.id] = asyncio.create_task(self._execute(handler, ctx), name=f"job-{job.id}")
        return job

    async def _execute(self, handler: JobHandler, ctx: JobContext) -> None:
        job = ctx.job
        try:
            self.transition(job, JobStatus.VALIDATING)
            handler.validate(job.params)
            self.transition(job, JobStatus.QUEUED)
            if job.kind.is_gpu_heavy:
                if self.lock.gpu_in_use:
                    ctx.log(f"GPU 작업 대기 중 (실행 중: {len(self.lock.holders)}개)")
                async with self.lock.gpu(job.id, handler.describe(job.params)):
                    await self._run_body(handler, ctx)
            else:
                await self._run_body(handler, ctx)
        except (JobCancelled, asyncio.CancelledError):
            self._finish(job, JobStatus.CANCELLED, error="사용자가 취소했습니다.")
        except StudioError as exc:
            detail = str(exc.detail)[:800] if exc.detail else ""
            message = f"{exc.user_message} — {detail}" if detail else exc.user_message
            ctx.log(message, "error")
            self._finish(job, JobStatus.FAILED, error=message)
        except Exception as exc:  # noqa: BLE001
            log.exception("작업 %s 실패", job.id)
            ctx.log(f"예상하지 못한 오류: {exc}", "error")
            self._finish(job, JobStatus.FAILED, error=f"예상하지 못한 오류: {exc}")
        finally:
            self._tasks.pop(job.id, None)
            self._contexts.pop(job.id, None)

    async def _run_body(self, handler: JobHandler, ctx: JobContext) -> None:
        ctx.check_cancelled()
        self.transition(ctx.job, JobStatus.RUNNING)
        result = await handler.run(ctx)
        ctx.job.result = result or {}
        ctx.job.progress = 1.0
        self._finish(ctx.job, JobStatus.SUCCEEDED)

    def _finish(self, job: Job, status: JobStatus, error: str | None = None) -> None:
        if job.status.is_terminal:
            return
        job.error = error
        if status != JobStatus.SUCCEEDED and job.progress == 1.0:
            job.progress = None
        self.transition(job, status)

    async def cancel(self, job_id: str) -> bool:
        ctx = self._contexts.get(job_id)
        task = self._tasks.get(job_id)
        if ctx is None or task is None:
            return False
        ctx.cancel_event.set()
        ctx.log("취소 요청됨", "warning")
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=15)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        return True

    async def wait(self, job_id: str, timeout: float | None = None) -> Job:
        task = self._tasks.get(job_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        job = self.repo.get(job_id)
        assert job is not None
        return job

    def is_running(self, job_id: str) -> bool:
        return job_id in self._tasks

    def reconcile_on_startup(self) -> list[str]:
        """앱 재시작 시 비정상 종료된 작업을 실패로 표시 (손상된 결과를 완료 처리하지 않음)."""
        fixed = []
        for job in self.repo.list_active():
            if job.id in self._tasks:
                continue
            job.error = "앱 재시작으로 중단되었습니다. 결과가 불완전할 수 있어 재실행이 필요합니다."
            job.status = JobStatus.FAILED
            job.progress = None
            job.ended_at = utcnow()
            self.repo.save(job)
            self.repo.add_event(JobEvent(job_id=job.id, level="warning", message=job.error))
            fixed.append(job.id)
        return fixed

    async def shutdown(self) -> None:
        for job_id in list(self._tasks):
            await self.cancel(job_id)
