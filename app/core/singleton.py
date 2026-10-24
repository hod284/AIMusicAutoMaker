"""싱글톤 패턴.

앱 전역에서 하나만 존재해야 하는 객체(설정, DB, 이벤트 버스, 리소스 락, DI 컨테이너)에
메타클래스로 적용한다. 스레드 안전하며, 테스트에서 상태를 초기화할 수 있도록
`reset()` 훅을 제공한다.
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar


class SingletonMeta(type):
    """스레드 안전 싱글톤 메타클래스.

    사용법::

        class EventBus(metaclass=SingletonMeta):
            ...

        EventBus() is EventBus()  # True
    """

    _instances: ClassVar[dict[type, Any]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Double-checked locking: 생성 이후의 호출은 락 없이 빠르게 반환
        instance = SingletonMeta._instances.get(cls)
        if instance is None:
            with SingletonMeta._lock:
                instance = SingletonMeta._instances.get(cls)
                if instance is None:
                    instance = super().__call__(*args, **kwargs)
                    SingletonMeta._instances[cls] = instance
        return instance

    def has_instance(cls) -> bool:
        return cls in SingletonMeta._instances

    def reset(cls) -> None:
        """해당 클래스의 싱글톤 인스턴스를 폐기한다 (테스트/재설정 용도)."""
        with SingletonMeta._lock:
            SingletonMeta._instances.pop(cls, None)

    @staticmethod
    def reset_all() -> None:
        with SingletonMeta._lock:
            SingletonMeta._instances.clear()
