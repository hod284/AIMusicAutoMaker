"""리소스 락 (싱글톤, REQUIREMENTS 11.2 / 7.3).

- ``gpu``: Mac M5 32GB 에서 동시 GPU 중량 작업 수 제한 (기본 1)
- ``music_engine``: ACE 서버의 LoRA 상태는 서버 전역이므로
  "LoRA 로드 → 생성 요청 → 결과 수신"을 하나의 임계구역으로 보호
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from app.core.errors import ResourceBusyError
from app.core.observer import EventBus, Topics
from app.core.singleton import SingletonMeta


class ResourceLock(metaclass=SingletonMeta):
    def __init__(self, gpu_slots: int = 1) -> None:
        self.gpu_slots = max(1, gpu_slots)
        self._gpu = asyncio.Semaphore(self.gpu_slots)
        self._music_engine = asyncio.Lock()
        self._holders: dict[str, str] = {}  # job_id -> 설명

    @property
    def gpu_in_use(self) -> int:
        return len(self._holders)

    @property
    def holders(self) -> dict[str, str]:
        return dict(self._holders)

    @asynccontextmanager
    async def gpu(self, owner: str, description: str = "", *, wait: bool = True) -> AsyncIterator[None]:
        if not wait and self._gpu.locked():
            raise ResourceBusyError(detail=self.holders)
        await self._gpu.acquire()
        self._holders[owner] = description
        EventBus().publish(Topics.RESOURCE_LOCK, owner=owner, acquired=True, description=description)
        try:
            yield
        finally:
            self._holders.pop(owner, None)
            self._gpu.release()
            EventBus().publish(Topics.RESOURCE_LOCK, owner=owner, acquired=False, description=description)

    @asynccontextmanager
    async def music_engine(self) -> AsyncIterator[None]:
        async with self._music_engine:
            yield
