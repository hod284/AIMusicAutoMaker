from __future__ import annotations

from app.adapters.ports import EngineHealth
from app.core.observer import DomainEvent, ObservableProperty, Topics
from app.domain.entities import Job, Song
from app.viewmodels.base import BaseViewModel


class DashboardViewModel(BaseViewModel):
    ace_health = ObservableProperty[EngineHealth | None](None)
    qwen_health = ObservableProperty[EngineHealth | None](None)
    jobs = ObservableProperty[list[Job]](factory=list)
    recent_songs = ObservableProperty[list[Song]](factory=list)
    gpu_holders = ObservableProperty[dict[str, str]](factory=dict)
    category_count = ObservableProperty(0)

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.listen("job.*", self._on_job)
        self.listen(Topics.SONG_CREATED, lambda e: self.reload_lists())
        self.listen(Topics.ENGINE_STATUS, self._on_engine)
        self.listen(Topics.RESOURCE_LOCK, lambda e: setattr(self, "gpu_holders", self.c.lock.holders))
        self.reload_lists()

    def reload_lists(self) -> None:
        self.jobs = self.c.job_repo.list_recent(15)
        self.recent_songs = self.c.song_repo.list(limit=8)
        self.category_count = len(self.c.categories.list())
        self.gpu_holders = self.c.lock.holders

    def _on_job(self, event: DomainEvent) -> None:
        if event.topic in (Topics.JOB_CREATED, Topics.JOB_UPDATED):
            self.jobs = self.c.job_repo.list_recent(15)

    def _on_engine(self, event: DomainEvent) -> None:
        health: EngineHealth = event.payload["health"]
        if health.engine == "ace":
            self.ace_health = health
        else:
            self.qwen_health = health

    async def refresh_health(self) -> None:
        async def _do() -> None:
            self.ace_health = await self.c.music_engine.health()
            self.qwen_health = await self.c.lyrics_engine.health()
        await self.run_command(_do)

    async def cancel_job(self, job_id: str) -> None:
        await self.run_command(lambda: self.c.jobs.cancel(job_id), success="취소 요청을 보냈습니다.")
