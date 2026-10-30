"""목소리 모델 화면 ViewModel.

- VoiceTrainingViewModel : [목소리 학습] 탭 — 녹음 올리기 → 학습(또는 참고 녹음만) → 목소리 모델 등록
- VoiceConvertViewModel  : [보컬 바꾸기] 탭 — 만든 곡 고르기 → 목소리 모델 고르기 → 분리·변환·합치기 → 새 곡
두 작업 모두 다른 화면에 다녀와도 진행 표시를 다시 붙이고, 실행 중에는 입력을 잠근다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.adapters.ports import EngineHealth
from app.adapters.voice.local_adapter import parse_seedvc_train_line
from app.core.observer import ObservableProperty, Topics
from app.domain.entities import Category, Job, ModelRecord, Song
from app.domain.enums import Engine, JobKind, JobStatus
from app.services.voice_service import VoiceTrainConfig
from app.viewmodels.base import BaseViewModel, JobTracker

ALL = ""


class _VoiceJobVM(BaseViewModel):
    job = ObservableProperty[Job | None](None)
    job_logs = ObservableProperty[list[str]](factory=list)
    health = ObservableProperty[EngineHealth | None](None)
    KINDS: tuple[JobKind, ...] = ()

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.tracker = JobTracker(self, self._on_job, self._on_log)

    @property
    def locked(self) -> bool:
        return self.job is not None and not self.job.status.is_terminal

    def _guard(self) -> bool:
        if self.locked:
            self.error = "작업이 끝난 뒤에 바꿀 수 있습니다."
            return True
        return False

    async def check_health(self) -> None:
        async def _do() -> None:
            self.health = await self.c.voice_engine.health()
        await self.run_command(_do)

    def _submit(self, kind: JobKind, params: dict[str, Any]) -> Job | None:
        if self.locked:
            self.error = "이전 작업이 아직 실행 중입니다."
            return None
        self.job_logs = []
        job = self.sync_command(lambda: self.c.jobs.submit(kind, params))
        if job:
            self.tracker.track(job)
        return job

    async def cancel(self) -> None:
        if self.job:
            await self.run_command(lambda: self.c.jobs.cancel(self.job.id))

    def _on_log(self, line: str) -> None:
        self.job_logs = (self.job_logs + [line])[-200:]

    def _on_job(self, job: Job) -> None:
        self.job = job
        if job.status == JobStatus.FAILED:
            self.error = job.error or "작업이 실패했습니다."


class VoiceTrainingViewModel(_VoiceJobVM):
    staged = ObservableProperty[list[str]](factory=list)
    model_name = ObservableProperty("")
    export_dir = ObservableProperty("")
    separate = ObservableProperty(False)
    max_steps = ObservableProperty(500)
    batch_size = ObservableProperty(2)
    losses = ObservableProperty[list[tuple[int, float]]](factory=list)


    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.svc = self.c.voice
        self.reload_staged()
        job = self.tracker.resume([JobKind.VOICE_TRAIN], recent_minutes=0)
        if job is not None and not job.status.is_terminal:
            cfg = job.params.get("config") or {}
            for key in ("model_name", "export_dir", "separate", "max_steps", "batch_size"):
                if key in cfg:
                    setattr(self, key, cfg[key])

    @property
    def max_upload_bytes(self) -> int:
        return self.svc.max_upload_bytes

    def reload_staged(self) -> None:
        self.staged = [p.name for p in self.svc.staged_files()]

    def upload(self, filename: str, data: bytes) -> None:
        if self._guard():
            return
        if self.sync_command(lambda: self.svc.stage_upload(filename, data)):
            self.reload_staged()

    def remove_staged(self, name: str) -> None:
        if self._guard():
            return
        self.sync_command(lambda: self.svc.remove_staged(name))
        self.reload_staged()

    def clear_staged(self) -> None:
        if self._guard():
            return
        self.sync_command(self.svc.clear_staged)
        self.reload_staged()

    @property
    def browse_root(self) -> Path:
        return Path.home()

    def browse(self, path: str | None = None) -> tuple[Path, list[tuple[str, bool]]]:
        result = self.sync_command(lambda: self.c.categories.browse_dirs(self.browse_root, path))
        return result if result else (self.browse_root, [])

    def start(self) -> None:
        if self._guard():
            return
        def _int(v: Any, default: int) -> int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return default
        form = {"model_name": (self.model_name or "").strip(), "export_dir": (self.export_dir or "").strip(),
                "mode": "train", "separate": bool(self.separate), "max_steps": _int(self.max_steps, 500),
                "batch_size": _int(self.batch_size, 2)}
        cfg = self.sync_command(lambda: VoiceTrainConfig(**form))
        if cfg is None:
            if not form["model_name"]:
                self.error = "목소리 모델 이름을 입력하세요."
            return
        self.losses = []
        self._submit(JobKind.VOICE_TRAIN, {"config": cfg.model_dump()})

    def _on_log(self, line: str) -> None:
        super()._on_log(line)
        tp = parse_seedvc_train_line(line)          # "[info] epoch 0, step 10, loss: 0.5"
        if tp and tp.step is not None and tp.train_loss is not None:
            self.losses = (self.losses + [(tp.step, tp.train_loss)])[-500:]

    def _on_job(self, job: Job) -> None:
        super()._on_job(job)
        self.reload_staged()
        if job.status == JobStatus.SUCCEEDED:
            model = self.c.categories.models.get(job.result.get("model_id", ""))
            self.info = (f"목소리 모델 '{model.display_name}' 를 만들었습니다. [보컬 변환]에서 쓸 수 있습니다."
                         if model else "목소리 모델을 만들었습니다.")
            self.model_name, self.export_dir = "", ""


class VoiceConvertViewModel(_VoiceJobVM):
    """[보컬 변환] 화면: ① 원곡(만든 곡/외부 파일) ② 목소리(내 모델/기본 모델+참고 녹음) ③ 키·제목·저장 폴더."""

    # ① 원곡
    source = ObservableProperty("song")                  # song | file
    songs = ObservableProperty[list[Song]](factory=list)
    song_query = ObservableProperty("")
    song_id = ObservableProperty[str | None](None)
    source_file = ObservableProperty[str | None](None)   # inputs/ 저장 이름
    # ② 목소리
    voice = ObservableProperty("model")                  # model | reference
    categories = ObservableProperty[list[Category]](factory=list)
    model_filter = ObservableProperty(ALL)
    models = ObservableProperty[list[ModelRecord]](factory=list)
    model_id = ObservableProperty[str | None](None)
    reference_file = ObservableProperty[str | None](None)
    rvc_model = ObservableProperty[str | None](None)
    rvc_index = ObservableProperty[str | None](None)
    rvc_index_rate = ObservableProperty(0.75)
    reference_separate = ObservableProperty(False)
    # ③ 설정·저장
    semi_tone_shift = ObservableProperty(0)
    diffusion_steps = ObservableProperty(30)
    title = ObservableProperty("")
    output_dir = ObservableProperty("")
    audio_format = ObservableProperty("mp3")
    results = ObservableProperty[list[Song]](factory=list)

    SOURCES = {"song": "만든 곡에서 고르기", "file": "외부 곡 파일 올리기"}
    VOICES = {"model": "내가 학습한 목소리 모델", "reference": "녹음 하나로 바꾸기 (학습 없이 기본 모델)", "rvc": "모델 넣기 (RVC)"}
    KEYS = {0: "원래 키 그대로", -12: "한 옥타브 아래 (-12)", 12: "한 옥타브 위 (+12)"}
    _FIELDS = ("rvc_model", "rvc_index", "rvc_index_rate", "source", "song_id", "source_file", "voice", "model_id", "reference_file", "reference_separate",
               "semi_tone_shift", "diffusion_steps", "title", "output_dir", "audio_format")

    def __init__(self, container=None, song_id: str | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.listen(Topics.CATEGORY_CHANGED, lambda e: self.load_models())
        self.listen(Topics.SONG_CREATED, lambda e: self.load_songs())
        self.load_models()
        self.load_songs()
        job = self.tracker.resume([JobKind.VOICE_CONVERT], recent_minutes=0)
        if job is not None and not job.status.is_terminal:     # 실행 중이면 누르기 직전 화면 그대로
            for key in self._FIELDS:
                if key in job.params:
                    setattr(self, key, job.params[key])
            if self.model_id and self.model_id not in {m.id for m in self.models}:
                self.model_filter = ALL
                self._load_models()
        elif song_id:
            self.select_song(song_id)

    # --- 조회 ---
    @property
    def song(self) -> Song | None:
        return self.c.song_repo.get(self.song_id) if self.song_id else None

    @property
    def max_upload_bytes(self) -> int:
        return self.c.voice.max_upload_bytes

    def label(self, stored_name: str | None) -> str:
        return self.c.voice.input_label(stored_name) if stored_name else ""

    def is_voice_song(self, song: Song) -> bool:
        return bool(song.generation.get("voice"))

    @property
    def browse_root(self) -> Path:
        return Path.home()

    def browse(self, path: str | None = None) -> tuple[Path, list[tuple[str, bool]]]:
        result = self.sync_command(lambda: self.c.categories.browse_dirs(self.browse_root, path))
        return result if result else (self.browse_root, [])

    # --- ① 원곡 ---
    def set_source(self, value: str) -> None:
        if self._guard():
            return
        self.source = value if value in self.SOURCES else "song"

    def load_songs(self) -> None:
        self.songs = self.c.music.list_songs(query=self.song_query, title_only=True, limit=200)

    def search_songs(self, query: str | None) -> None:
        if self.locked:
            return
        self.song_query = query or ""
        self.load_songs()

    def select_song(self, song_id: str | None) -> None:
        if self._guard():
            return
        self.source = "song"
        self.song_id = song_id or None

    def upload_source(self, filename: str, data: bytes) -> None:
        if self._guard():
            return
        path = self.sync_command(lambda: self.c.voice.store_input(filename, data))
        if path:
            self.source, self.source_file = "file", path.name

    # --- ② 목소리 ---
    def set_voice(self, value: str) -> None:
        if self._guard():
            return
        self.voice = value if value in self.VOICES else "model"

    def load_models(self) -> None:
        self.categories = self.c.categories.list()
        if self.model_filter and self.model_filter not in {c.id for c in self.categories}:
            self.model_filter = ALL
        self._load_models()

    def _load_models(self) -> None:
        self.models = self.c.voice.voice_models(self.model_filter or None)
        if self.model_id not in {m.id for m in self.models}:
            active = self.c.categories.active_model(self.model_filter, Engine.VOICE) if self.model_filter else None
            self.model_id = active.id if active else (self.models[0].id if len(self.models) == 1 else None)

    def set_model_filter(self, category_id: str | None) -> None:
        if self._guard():
            return
        self.model_filter = category_id or ALL
        self._load_models()

    def set_model(self, model_id: str | None) -> None:
        if self._guard():
            return
        self.model_id = model_id or None

    def upload_rvc(self, filename: str, data: bytes) -> None:
        if self._guard():
            return
        path = self.sync_command(lambda: self.c.voice.store_rvc(filename, data))
        if path:
            if path.suffix == ".pth":
                self.c.voice.remove_rvc(self.rvc_model)
                self.rvc_model = path.name
            else:
                self.c.voice.remove_rvc(self.rvc_index)
                self.rvc_index = path.name
            self.voice = "rvc"

    def upload_reference(self, filename: str, data: bytes) -> None:
        if self._guard():
            return
        path = self.sync_command(lambda: self.c.voice.store_input(filename, data))
        if path:
            self.voice, self.reference_file = "reference", path.name

    # --- 실행 ---
    def start(self) -> None:
        if self._guard():
            return
        if self.source == "song" and not self.song_id:
            self.error = "① 에서 바꿀 곡을 고르세요."
            return
        if self.source == "file" and not self.source_file:
            self.error = "① 에서 바꿀 곡 파일을 올리세요."
            return
        if self.voice == "model" and not self.model_id:
            self.error = "② 에서 목소리 모델을 고르세요."
            return
        if self.voice == "rvc" and not self.rvc_model:
            self.error = "② 에서 RVC 모델(.pth)을 넣으세요."
            return
        if self.voice == "reference" and not self.reference_file:
            self.error = "② 에서 바꿀 목소리의 참고 녹음을 올리세요."
            return
        try:
            steps = int(self.diffusion_steps or 30)
            key = int(self.semi_tone_shift or 0)
        except (TypeError, ValueError):
            self.error = "변환 스텝·키 값을 확인하세요."
            return
        if (self.output_dir or "").strip() and self.sync_command(
                lambda: self.c.voice.output_dir(self.output_dir)) is None:
            return
        params = {k: getattr(self, k) for k in self._FIELDS}
        params |= {"semi_tone_shift": key, "diffusion_steps": steps, "title": (self.title or "").strip(),
                   "output_dir": (self.output_dir or "").strip()}
        self._submit(JobKind.VOICE_CONVERT, params)

    def _on_job(self, job: Job) -> None:
        super()._on_job(job)
        if job.status == JobStatus.SUCCEEDED:
            self.results = [s for sid in job.result.get("song_ids", []) if (s := self.c.song_repo.get(sid))] \
                + self.results
            self.title = ""
            if job.params.get("voice") == "rvc":
                self.rvc_model = self.rvc_index = None
                self.rvc_index_rate = 0.75
            self.info = f"보컬 변환 완료 → {job.result.get('path', '')}  (만든 곡 목록에도 있습니다)"
