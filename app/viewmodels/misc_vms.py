"""Library / Jobs / Settings / Training ViewModel."""

from __future__ import annotations

from app.adapters.ports import CapabilityReport
from app.core.observer import DomainEvent, ObservableProperty, Topics
from app.domain.entities import Category, Dataset, Job, JobEvent, Song
from app.domain.enums import DatasetKind, JobKind
from app.services.diagnostics_service import DiagnosticsReport
from app.services.training_service import JsonlReport
from app.viewmodels.base import BaseViewModel, JobTracker


class LibraryViewModel(BaseViewModel):
    songs = ObservableProperty[list[Song]](factory=list)
    categories = ObservableProperty[list[Category]](factory=list)
    category_filter = ObservableProperty[str | None](None)
    query = ObservableProperty("")

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.categories = self.c.categories.list()
        self.listen(Topics.SONG_CREATED, lambda e: self.load())
        self.load()

    def load(self) -> None:
        self.songs = self.c.music.list_songs(self.category_filter, self.query)

    def set_filter(self, category_id: str | None = None, query: str | None = None) -> None:
        if category_id is not None:
            self.category_filter = category_id or None
        if query is not None:
            self.query = query
        self.load()

    def category_name(self, category_id: str | None) -> str:
        if not category_id:
            return "카테고리 없음"
        return next((c.name for c in self.categories if c.id == category_id), category_id[:8])


class JobsViewModel(BaseViewModel):
    jobs = ObservableProperty[list[Job]](factory=list)
    selected_id = ObservableProperty[str | None](None)
    events = ObservableProperty[list[JobEvent]](factory=list)

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.listen("job.*", self._on_job)
        self.load()

    def load(self) -> None:
        self.jobs = self.c.job_repo.list_recent(100)

    def select(self, job_id: str) -> None:
        self.selected_id = job_id
        self.events = self.c.job_repo.events(job_id)

    def _on_job(self, event: DomainEvent) -> None:
        if event.topic == Topics.JOB_LOG:
            if event.payload.get("job_id") == self.selected_id:
                self.events = self.events + [event.payload["event"]]
        else:
            self.load()

    async def cancel(self, job_id: str) -> None:
        await self.run_command(lambda: self.c.jobs.cancel(job_id), success="취소 요청을 보냈습니다.")


class SettingsViewModel(BaseViewModel):
    report = ObservableProperty[DiagnosticsReport | None](None)
    capabilities = ObservableProperty[CapabilityReport | None](None)

    async def run_diagnostics(self) -> None:
        async def _do() -> None:
            await self.c.music_engine.capabilities(refresh=True)
            report = await self.c.diagnostics.run()
            self.report = report
            self.capabilities = report.capabilities
            self.bus.publish(Topics.ENGINE_STATUS, health=report.ace)
            self.bus.publish(Topics.ENGINE_STATUS, health=report.qwen)
        await self.run_command(_do, success="진단을 완료했습니다.")

    def settings_rows(self) -> list[tuple[str, str]]:
        s = self.c.settings
        return [
            ("NiceGUI", f"{s.ui_host}:{s.ui_port}"), ("ACE-Step API", s.ace_base_url),
            ("ACE 라우트 바인딩", str(s.ace_route_bindings)), ("ACE 경로", str(s.ace_root)),
            ("lyrics-ai 경로", str(s.lyrics_root)), ("Qwen 기본 모델", s.qwen_base_model),
            ("작업 폴더", str(s.workspace_dir)), ("DB", str(s.db_path)),
            ("음악 엔진", f"{s.music_engine}{' (MOCK)' if self.c.music_engine.is_mock else ''}"),
            ("가사 엔진", f"{s.lyrics_engine}{' (MOCK)' if self.c.lyrics_engine.is_mock else ''}"),
            ("GPU 동시 작업", str(s.gpu_heavy_concurrency)),
            ("ACE API 키", "설정됨" if s.ace_api_key else "없음"),
        ]


class TrainingViewModel(BaseViewModel):
    categories = ObservableProperty[list[Category]](factory=list)
    category_id = ObservableProperty[str | None](None)
    datasets = ObservableProperty[list[Dataset]](factory=list)
    jsonl_report = ObservableProperty[JsonlReport | None](None)
    train_options = ObservableProperty[list[str]](factory=list)
    job = ObservableProperty[Job | None](None)
    job_logs = ObservableProperty[list[str]](factory=list)
    losses = ObservableProperty[list[tuple[int, float]]](factory=list)
    capabilities = ObservableProperty[CapabilityReport | None](None)

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.tracker = JobTracker(self, self._on_job, self._on_log)
        self.categories = self.c.categories.list()
        if self.categories:
            self.select(self.categories[0].id)

    def select(self, category_id: str | None) -> None:
        self.category_id = category_id
        self.datasets = self.c.dataset_repo.list(category_id, DatasetKind.LYRICS) if category_id else []

    def preview_jsonl(self, text: str) -> None:
        _, report = self.c.lyrics_datasets.parse_jsonl(text)
        self.jsonl_report = report

    def build_dataset(self, text: str) -> None:
        if not self.category_id:
            self.error = "카테고리를 선택하세요."
            return
        result = self.sync_command(lambda: self.c.lyrics_datasets.build(self.category_id, text),
                                   success="데이터셋 버전을 만들었습니다.")
        if result:
            self.jsonl_report = result[1]
            self.select(self.category_id)

    async def load_train_options(self) -> None:
        async def _do() -> None:
            self.train_options = sorted(await self.c.lyrics_engine.training_options())
        await self.run_command(_do)

    async def load_capabilities(self) -> None:
        async def _do() -> None:
            self.capabilities = await self.c.music_engine.capabilities()
        await self.run_command(_do)

    def start_lyrics_training(self, dataset_id: str, iters: int, batch_size: int, learning_rate: float | None) -> None:
        if self.job and not self.job.status.is_terminal:
            self.error = "학습이 이미 실행 중입니다."
            return
        self.losses, self.job_logs = [], []
        job = self.sync_command(lambda: self.c.jobs.submit(JobKind.LYRICS_TRAIN, {
            "category_id": self.category_id, "dataset_id": dataset_id, "iters": iters, "batch_size": batch_size,
            "learning_rate": learning_rate,
        }, category_id=self.category_id, dataset_id=dataset_id))
        if job:
            self.tracker.track(job)

    async def cancel(self) -> None:
        if self.job:
            await self.run_command(lambda: self.c.jobs.cancel(self.job.id))

    def _on_job(self, job: Job) -> None:
        self.job = job
        if job.status.is_terminal and job.error:
            self.error = job.error

    def _on_log(self, line: str) -> None:
        import re
        if m := re.search(r"Iter\s+(\d+):\s*Train loss\s+([0-9.]+)", line):
            self.losses = self.losses + [(int(m.group(1)), float(m.group(2)))]
        self.job_logs = (self.job_logs + [line])[-300:]
