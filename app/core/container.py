"""서비스 컨테이너 (싱글톤 + Composition Root).

모든 의존성 조립은 여기서만 한다. ViewModel/테스트는 ``ServiceContainer()`` 로 같은 인스턴스를 얻는다.
테스트에서는 ``ServiceContainer.reset()`` 후 ``ServiceContainer(settings, music=Mock..., lyrics=Mock...)``
처럼 어댑터를 주입할 수 있다 (Dependency Injection).
"""

from __future__ import annotations

import logging

from app.adapters.factory import AdapterFactory
from app.adapters.ports import LyricsEnginePort, MusicEnginePort
from app.core.config import Settings, get_settings
from app.core.observer import EventBus, Topics
from app.core.singleton import SingletonMeta
from app.db.database import Database
from app.repositories.category_repo import CategoryRepository
from app.repositories.content_repo import DatasetRepository, LyricsRepository, SongRepository
from app.repositories.job_repo import JobRepository
from app.repositories.model_repo import ModelRepository
from app.services.category_service import CategoryService
from app.services.diagnostics_service import DiagnosticsService
from app.services.job_service import JobManager
from app.services.lyrics_service import LyricsGenerateHandler, LyricsService
from app.services.music_service import MusicGenerateHandler, MusicService
from app.services.pipeline_service import FullSongHandler
from app.services.prompt_service import PromptDirector, PromptGenerateHandler, QwenDirectorStrategy, TemplateStrategy
from app.services.resource_lock import ResourceLock
from app.services.training_service import LyricsDatasetBuilder, LyricsTrainHandler
from app.services.music_training_service import (
    MusicDatasetService,
    MusicLabelHandler,
    MusicPreprocessHandler,
    MusicScanHandler,
    MusicTrainHandler,
)
from app.services.workspace import WorkspaceManager
from app.utils.path_guard import PathGuard

log = logging.getLogger(__name__)


class ServiceContainer(metaclass=SingletonMeta):
    def __init__(self, settings: Settings | None = None, *, music: MusicEnginePort | None = None,
                 lyrics: LyricsEnginePort | None = None) -> None:
        self.settings = s = settings or get_settings()
        s.workspace_dir.mkdir(parents=True, exist_ok=True)
        s.data_dir.mkdir(parents=True, exist_ok=True)

        self.bus = EventBus()
        self.db = Database(s.db_path)
        self.lock = ResourceLock(s.gpu_heavy_concurrency)
        self.guard = PathGuard(
            allowed_roots=[s.workspace_dir, s.data_dir, s.ace_studio_share_dir],
            protected=s.protected_paths,
        )

        # --- 어댑터 (Adapter + Factory) ---
        self.music_engine = music or AdapterFactory.music(s)
        self.lyrics_engine = lyrics or AdapterFactory.lyrics(s)

        # --- 리포지토리 ---
        self.category_repo = CategoryRepository(self.db)
        self.model_repo = ModelRepository(self.db)
        self.job_repo = JobRepository(self.db)
        self.lyrics_repo = LyricsRepository(self.db)
        self.song_repo = SongRepository(self.db)
        self.dataset_repo = DatasetRepository(self.db)

        # --- 서비스 ---
        self.workspace = WorkspaceManager(s.categories_dir, self.guard, s.data_dir / "trash")
        self.categories = CategoryService(self.category_repo, self.model_repo, self.workspace, self.bus)
        self.lyrics = LyricsService(self.lyrics_engine, self.lyrics_repo, self.categories, self.bus)
        self.director = PromptDirector([QwenDirectorStrategy(self.lyrics_engine), TemplateStrategy()])
        self.music = MusicService(self.music_engine, self.song_repo, self.categories, self.lyrics, self.lock,
                                  poll_interval_s=s.ace_poll_interval_s, poll_timeout_s=s.ace_poll_timeout_s,
                                  bus=self.bus)
        self.diagnostics = DiagnosticsService(s, self.music_engine, self.lyrics_engine)
        self.lyrics_datasets = LyricsDatasetBuilder(self.categories, self.dataset_repo,
                                                    s.workspace_dir / "library" / "lyrics")
        self.music_datasets = MusicDatasetService(
            self.music_engine, self.categories, self.dataset_repo, self.guard, s.ace_studio_share_dir, self.lock,
            max_upload_mb=s.max_upload_mb,
        )

        # --- 잡 (Command 핸들러 등록) ---
        self.jobs = JobManager(self.job_repo, self.lock, self.bus)
        for handler in (
            LyricsGenerateHandler(self.lyrics),
            PromptGenerateHandler(self.director, self.categories, self.lyrics),
            MusicGenerateHandler(self.music),
            FullSongHandler(self.lyrics, self.director, self.music),
            LyricsTrainHandler(self.lyrics_engine, self.lyrics_datasets, s.qwen_base_model),
            MusicScanHandler(self.music_datasets),
            MusicLabelHandler(self.music_datasets),
            MusicPreprocessHandler(self.music_datasets),
            MusicTrainHandler(self.music_datasets, self.music),
        ):
            self.jobs.register(handler)

    async def startup(self) -> None:
        fixed = self.jobs.reconcile_on_startup()
        if fixed:
            log.warning("재시작으로 중단된 작업 %d개를 실패 처리했습니다.", len(fixed))
        for engine in (self.music_engine, self.lyrics_engine):
            try:
                health = await engine.health()
                self.bus.publish(Topics.ENGINE_STATUS, health=health)
            except Exception:  # noqa: BLE001 - 시작 시 엔진 오류로 앱이 죽지 않게
                log.exception("엔진 상태 확인 실패")

    async def shutdown(self) -> None:
        await self.jobs.shutdown()
        await self.music_engine.aclose()
