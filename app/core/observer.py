"""옵저버 패턴.

두 가지 수준을 제공한다.

1. ``Observable`` / ``ObservableProperty``
   ViewModel 의 속성 변경을 View 에 알리는 세밀한(fine-grained) 옵저버.
   MVVM 의 데이터 바인딩 역할을 한다.

2. ``EventBus`` (싱글톤)
   서비스 계층에서 발생한 도메인 이벤트(잡 상태 변경, 곡 생성 완료, 엔진 상태 등)를
   여러 ViewModel 에 브로드캐스트하는 거친(coarse-grained) 발행/구독 채널.
   서비스는 UI 를 전혀 모른 채 이벤트만 발행한다.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Generic, TypeVar, Union

from app.core.singleton import SingletonMeta

log = logging.getLogger(__name__)

T = TypeVar("T")
Listener = Callable[..., Union[None, Awaitable[None]]]


class Subscription:
    """구독 해제 핸들. View 가 사라질 때 반드시 ``dispose()`` 한다."""

    def __init__(self, unsubscribe: Callable[[], None]) -> None:
        self._unsubscribe: Callable[[], None] | None = unsubscribe

    def dispose(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    @property
    def active(self) -> bool:
        return self._unsubscribe is not None


def _invoke(listener: Listener, *args: Any) -> None:
    """동기/비동기 리스너를 모두 지원. 리스너 예외는 다른 리스너에 전파하지 않는다."""
    try:
        result = listener(*args)
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(result)  # type: ignore[arg-type]
            else:
                loop.create_task(result)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 - 옵저버 하나의 오류가 전체를 막지 않도록
        log.exception("옵저버 리스너 실행 중 오류")


# ---------------------------------------------------------------------------
# 1) 속성 단위 옵저버 (MVVM 바인딩)
# ---------------------------------------------------------------------------
class Observable:
    """속성 변경 알림을 지원하는 베이스 클래스.

    ``ObservableProperty`` 디스크립터가 값 변경 시 ``notify(name, value)`` 를 호출한다.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[Listener]] = {}

    def subscribe(self, name: str, listener: Listener) -> Subscription:
        """``name`` 속성이 바뀌면 ``listener(new_value)`` 호출. ``"*"`` 은 모든 속성."""
        self._listeners.setdefault(name, []).append(listener)

        def _unsubscribe() -> None:
            listeners = self._listeners.get(name, [])
            if listener in listeners:
                listeners.remove(listener)

        return Subscription(_unsubscribe)

    def notify(self, name: str, value: Any) -> None:
        for listener in list(self._listeners.get(name, [])):
            _invoke(listener, value)
        for listener in list(self._listeners.get("*", [])):
            _invoke(listener, name, value)

    def clear_listeners(self) -> None:
        self._listeners.clear()


class ObservableProperty(Generic[T]):
    """값이 실제로 바뀔 때만 ``Observable.notify`` 를 호출하는 디스크립터."""

    def __init__(self, default: T | None = None, *, factory: Callable[[], T] | None = None) -> None:
        self._default = default
        self._factory = factory
        self._name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def _storage(self) -> str:
        return f"_obs_{self._name}"

    def __get__(self, instance: Observable | None, owner: type) -> Any:
        if instance is None:
            return self
        storage = self._storage()
        if not hasattr(instance, storage):
            value = self._factory() if self._factory else self._default
            setattr(instance, storage, value)
        return getattr(instance, storage)

    def __set__(self, instance: Observable, value: T) -> None:
        storage = self._storage()
        old = getattr(instance, storage, object())
        setattr(instance, storage, value)
        # 리스트/딕트 등 가변 객체는 동일 객체라도 재할당 시 알림 (명시적 갱신 의도)
        if old is value and not isinstance(value, (list, dict, set)):
            return
        if old == value and not isinstance(value, (list, dict, set)):
            return
        instance.notify(self._name, value)


# ---------------------------------------------------------------------------
# 2) 도메인 이벤트 버스 (싱글톤)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DomainEvent:
    """이벤트 버스로 흐르는 메시지. ``topic`` 으로 라우팅한다."""

    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class Topics:
    """이벤트 토픽 상수. 새 기능은 여기에 토픽만 추가하면 된다."""

    JOB_CREATED = "job.created"
    JOB_UPDATED = "job.updated"          # 상태/진행률 변경
    JOB_LOG = "job.log"                  # 로그 한 줄
    CATEGORY_CHANGED = "category.changed"
    LYRICS_SAVED = "lyrics.saved"
    SONG_CREATED = "song.created"
    ENGINE_STATUS = "engine.status"      # ACE / Qwen 연결 상태
    RESOURCE_LOCK = "resource.lock"      # GPU 락 획득/해제
    NOTIFY = "ui.notify"                 # 사용자 알림 (토스트)


class EventBus(metaclass=SingletonMeta):
    """앱 전역 발행/구독 버스 (싱글톤).

    - ``subscribe("job.*", fn)`` 처럼 접두 와일드카드를 지원한다.
    - 리스너가 async 함수여도 된다 (실행 중인 이벤트 루프에 태스크로 예약).
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[[DomainEvent], Any]]] = {}

    def subscribe(self, topic: str, listener: Callable[[DomainEvent], Any]) -> Subscription:
        self._subscribers.setdefault(topic, []).append(listener)

        def _unsubscribe() -> None:
            lst = self._subscribers.get(topic, [])
            if listener in lst:
                lst.remove(listener)

        return Subscription(_unsubscribe)

    def publish(self, topic: str, **payload: Any) -> DomainEvent:
        event = DomainEvent(topic=topic, payload=payload)
        for pattern, listeners in list(self._subscribers.items()):
            if self._matches(pattern, topic):
                for listener in list(listeners):
                    _invoke(listener, event)
        return event

    def subscriber_count(self, topic: str | None = None) -> int:
        if topic is None:
            return sum(len(v) for v in self._subscribers.values())
        return len(self._subscribers.get(topic, []))

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        if pattern == "*" or pattern == topic:
            return True
        if pattern.endswith(".*"):
            return topic.startswith(pattern[:-1])
        return False
