"""ViewModel 베이스 (MVVM 의 VM).

규칙:
- ViewModel 은 NiceGUI 를 import 하지 않는다 → UI 없이 단위 테스트 가능.
- View 는 ``ObservableProperty`` 를 구독해 화면을 갱신하고, 사용자 동작은 VM 의 커맨드(async 메서드)로 전달한다.
- 서비스 이벤트는 ``EventBus`` 구독으로 받는다. View 가 닫히면 ``dispose()`` 로 모든 구독을 해제한다.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, TypeVar

from app.core.container import ServiceContainer
from app.core.errors import StudioError
from app.core.observer import DomainEvent, EventBus, Observable, ObservableProperty, Subscription, Topics
from app.domain.entities import Job

log = logging.getLogger(__name__)
T = TypeVar("T")


class BaseViewModel(Observable):
    busy = ObservableProperty(False)
    error = ObservableProperty("")
    info = ObservableProperty("")

    def __init__(self, container: ServiceContainer | None = None) -> None:
        super().__init__()
        self.c = container or ServiceContainer()
        self.bus: EventBus = self.c.bus
        self._subs: list[Subscription] = []

    def listen(self, topic: str, handler: Callable[[DomainEvent], Any]) -> None:
        self._subs.append(self.bus.subscribe(topic, handler))

    async def run_command(self, fn: Callable[[], Awaitable[T]], *, success: str = "") -> T | None:
        """커맨드 실행 공통 처리: busy 표시, 사용자 메시지/진단 로그 분리."""
        self.busy = True
        self.error = ""
        try:
            result = await fn()
            if success:
                self.info = success
            return result
        except StudioError as exc:
            log.warning("커맨드 실패: %s", exc)
            self.error = exc.user_message
        except Exception as exc:  # noqa: BLE001
            log.exception("커맨드 예외")
            self.error = f"예상하지 못한 오류: {exc}"
        finally:
            self.busy = False
        return None

    def sync_command(self, fn: Callable[[], T], *, success: str = "") -> T | None:
        self.error = ""
        try:
            result = fn()
            if success:
                self.info = success
            return result
        except StudioError as exc:
            self.error = exc.user_message
        except Exception as exc:  # noqa: BLE001
            log.exception("커맨드 예외")
            self.error = f"예상하지 못한 오류: {exc}"
        return None

    def dispose(self) -> None:
        for sub in self._subs:
            sub.dispose()
        self._subs.clear()
        self.clear_listeners()


class JobTracker:
    """특정 job_id 의 상태/로그를 추적하는 옵저버 헬퍼. 여러 VM 에서 재사용."""

    def __init__(self, vm: BaseViewModel, on_update: Callable[[Job], Any], on_log: Callable[[str], Any] | None = None) -> None:
        self.vm = vm
        self.job_id: str | None = None
        self._on_update = on_update
        self._on_log = on_log
        vm.listen(Topics.JOB_UPDATED, self._handle_update)
        vm.listen(Topics.JOB_LOG, self._handle_log)

    def track(self, job: Job) -> None:
        self.job_id = job.id
        self._on_update(job)

    def _handle_update(self, event: DomainEvent) -> None:
        job: Job = event.payload["job"]
        if job.id == self.job_id:
            self._on_update(job)

    def _handle_log(self, event: DomainEvent) -> None:
        if self._on_log and event.payload.get("job_id") == self.job_id:
            ev = event.payload["event"]
            if ev.level != "debug":
                self._on_log(f"[{ev.level}] {ev.message}")
