"""Library / Jobs / Settings / Training ViewModel."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.adapters.ports import CapabilityReport
from app.core.observer import DomainEvent, ObservableProperty, Topics
from app.domain.entities import Dataset, Job, JobEvent, Song
from app.domain.enums import JobKind, JobStatus
from app.repositories.song_category_repo import SongCategory
from app.services.diagnostics_service import DiagnosticsReport
from app.services.training_service import (
    JsonlReport,
    LyricsTrainConfig,
    decode_text,
    txt_files_to_jsonl,
    txt_to_record,
)
from app.viewmodels.base import BaseViewModel, JobTracker


class LibraryViewModel(BaseViewModel):
    """만든 곡 [곡 목록] 탭: 들어올 때는 항상 '전체', 카테고리를 고르면 그 카테고리 곡만. 곡 제목 검색."""

    songs = ObservableProperty[list[Song]](factory=list)
    categories = ObservableProperty[list[SongCategory]](factory=list)
    category_filter = ObservableProperty("")        # "" = 전체
    query = ObservableProperty("")
    membership = ObservableProperty[dict[str, list[str]]](factory=dict)

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.svc = self.c.song_categories
        self.listen(Topics.SONG_CREATED, lambda e: self.load())
        self.load()

    def load(self) -> None:
        self.categories = self.svc.list()
        if self.category_filter and self.category_filter not in {c.id for c in self.categories}:
            self.category_filter = ""
        self.membership = self.svc.membership()
        self.songs = self.svc.list_songs(self.category_filter or None, self.query)

    def set_filter(self, category_id: str | None) -> None:
        self.category_filter = category_id or ""
        self.load()

    def set_query(self, query: str | None) -> None:
        self.query = query or ""
        self.load()

    def category_names(self, song_id: str) -> list[str]:
        names = {c.id: c.name for c in self.categories}
        return [names[c] for c in self.membership.get(song_id, []) if c in names]

    def model_name(self, model_id: str | None) -> str:
        if not model_id:
            return "ACE 기본"
        model = self.c.categories.models.get(model_id)
        return model.label if model else "삭제된 모델"


class SongCategoriesViewModel(BaseViewModel):
    """만든 곡 [카테고리 편집] 탭: 카테고리 만들기/삭제, 곡 넣기/빼기, 곡 삭제."""

    categories = ObservableProperty[list[SongCategory]](factory=list)
    category_id = ObservableProperty[str | None](None)
    members = ObservableProperty[list[Song]](factory=list)
    all_songs = ObservableProperty[list[Song]](factory=list)
    query = ObservableProperty("")

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.svc = self.c.song_categories
        self.listen(Topics.SONG_CREATED, lambda e: self.load())
        self.load()

    @property
    def member_ids(self) -> set[str]:
        return {s.id for s in self.members}

    @property
    def category_name(self) -> str:
        return next((c.name for c in self.categories if c.id == self.category_id), "")

    def load(self) -> None:
        self.categories = self.svc.list()
        if self.category_id not in {c.id for c in self.categories}:
            self.category_id = self.categories[0].id if self.categories else None
        self.members = self.svc.list_songs(self.category_id) if self.category_id else []
        self.all_songs = self.svc.list_songs(None, self.query)

    def select(self, category_id: str | None) -> None:
        self.category_id = category_id
        self.load()

    def set_query(self, query: str | None) -> None:
        self.query = query or ""
        self.load()

    def create(self, name: str) -> None:
        cat = self.sync_command(lambda: self.svc.create(name), success=f"카테고리 '{name.strip()}' 를 만들었습니다.")
        if cat:
            self.category_id = cat.id
            self.load()

    def delete_category(self) -> None:
        if not self.category_id:
            return
        name = self.category_name
        self.sync_command(lambda: self.svc.delete(self.category_id))
        if not self.error:
            self.info = f"카테고리 '{name}' 를 삭제했습니다. (곡은 그대로 남습니다)"
            self.category_id = None
            self.load()

    def add(self, song_id: str) -> None:
        if not self.category_id:
            self.error = "왼쪽에서 카테고리를 먼저 만들거나 고르세요."
            return
        self.sync_command(lambda: self.svc.add_song(self.category_id, song_id), success="카테고리에 넣었습니다.")

    def remove(self, song_id: str) -> None:
        if self.category_id:
            self.sync_command(lambda: self.svc.remove_song(self.category_id, song_id),
                              success="카테고리에서 뺐습니다. (곡은 그대로)")

    def delete_song(self, song_id: str, delete_file: bool) -> None:
        title = next((s.title for s in self.all_songs if s.id == song_id), "")
        removed = self.sync_command(lambda: self.svc.delete_song(song_id, delete_file=delete_file))
        if not self.error:
            self.info = f"'{title}' 곡을 삭제했습니다." + (f" (파일도 삭제: {removed})" if removed else "")


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
            ("목소리 엔진", f"{s.voice_engine}{' (MOCK)' if self.c.voice_engine.is_mock else ''}"),
            ("voice-ai 경로", str(s.voice_root)), ("audio-separator", str(s.separator_bin)),
            ("Seed-VC", str(s.seedvc_root)),
            ("GPU 동시 작업", str(s.gpu_heavy_concurrency)),
            ("ACE API 키", "설정됨" if s.ace_api_key else "없음"),
        ]


class TrainingViewModel(BaseViewModel):
    """가사 LoRA 학습 화면 VM: JSONL → 데이터셋 버전 → mlx_lm.lora 학습 → 모델 등록. 카테고리 없음."""

    datasets = ObservableProperty[list[Dataset]](factory=list)
    dataset_id = ObservableProperty[str | None](None)
    jsonl_report = ObservableProperty[JsonlReport | None](None)
    train_options = ObservableProperty[list[str]](factory=list)
    job = ObservableProperty[Job | None](None)
    job_logs = ObservableProperty[list[str]](factory=list)
    losses = ObservableProperty[list[tuple[int, float]]](factory=list)
    val_losses = ObservableProperty[list[tuple[int, float]]](factory=list)
    txt_files = ObservableProperty[list[tuple[str, str]]](factory=list)   # 올린 가사 .txt (이름, 내용)
    model_title = ObservableProperty("")   # 결과 모델 이름
    export_dir = ObservableProperty("")    # 결과 저장 폴더 (비우면 studio/workspace/library/lyrics/adapters)

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.svc = self.c.lyrics_datasets
        self.form: dict[str, Any] = {"iters": 200, "batch_size": 1, "learning_rate": None, "num_layers": None,
                                     "max_seq_length": None, "save_every": None}
        self.tracker = JobTracker(self, self._on_job, self._on_log)
        self._reload_datasets(select_latest=True)
        job = self.tracker.resume([JobKind.LYRICS_TRAIN])   # 다른 화면에 다녀와도 학습 진행 표시 유지
        if job and job.dataset_id:
            self.dataset_id = job.dataset_id
        if job is not None and not job.status.is_terminal:  # 학습 중이면 시작할 때의 입력값을 그대로 보여 줌
            cfg = job.params.get("config") or {}
            self.form.update({k: cfg[k] for k in self.form if k in cfg})
            self.model_title = cfg.get("model_name", "")
            self.export_dir = cfg.get("export_dir", "")

    # --- 조회 ---
    @property
    def training(self) -> bool:
        """학습 작업이 실행 중인지. (BaseViewModel.busy 와 이름이 겹치지 않게 따로 둔다)"""
        return self.job is not None and not self.job.status.is_terminal

    @property
    def locked(self) -> bool:
        """학습 중에는 화면의 모든 입력을 잠근다."""
        return self.training

    def _guard(self) -> bool:
        if self.locked:
            self.error = "가사 LoRA 학습 중입니다. 끝난 뒤에 할 수 있습니다."
            return True
        return False

    @property
    def dataset(self) -> Dataset | None:
        return next((d for d in self.datasets if d.id == self.dataset_id), None)

    def dataset_label(self, ds: Dataset) -> str:
        return self.svc.label(ds)

    def dataset_counts(self, ds: Dataset) -> str:
        counts = []
        for name in ("train", "valid"):
            f = Path(ds.source_path) / f"{name}.jsonl"
            n = sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip()) if f.is_file() else 0
            counts.append(f"{name} {n}")
        return " · ".join(counts)

    @property
    def browse_root(self) -> Path:
        return Path.home()

    def browse(self, path: str | None = None) -> tuple[Path, list[tuple[str, bool]]]:
        result = self.sync_command(lambda: self.c.categories.browse_dirs(self.browse_root, path))
        return result if result else (self.browse_root, [])

    def _reload_datasets(self, select_latest: bool = False) -> None:
        self.datasets = self.svc.list()
        if select_latest or self.dataset_id not in {d.id for d in self.datasets}:
            self.dataset_id = self.datasets[-1].id if self.datasets else None

    def select_dataset(self, dataset_id: str | None) -> None:
        if self._guard():
            return
        self.dataset_id = dataset_id or None

    # --- ① 데이터셋: 가사 .txt ---
    MAX_TXT_BYTES = 200_000

    def add_txt(self, filename: str, data: bytes) -> None:
        if self._guard():
            return
        if not filename.lower().endswith(".txt"):
            self.error = f"{filename}: .txt 파일만 올릴 수 있습니다."
            return
        if len(data) > self.MAX_TXT_BYTES:
            self.error = f"{filename}: 파일이 너무 큽니다 (가사 1곡 = 파일 1개)."
            return
        text = self.sync_command(lambda: decode_text(data))
        if text is None:
            return
        if self.sync_command(lambda: txt_to_record(filename, text)) is None:   # 빈 가사 등 미리 거름
            return
        others = [(n, t) for n, t in self.txt_files if n != filename]
        self.txt_files = others + [(filename, text)]

    def remove_txt(self, filename: str) -> None:
        if self._guard():
            return
        self.txt_files = [(n, t) for n, t in self.txt_files if n != filename]

    def clear_txt(self) -> None:
        if self._guard():
            return
        self.txt_files = []

    def preview_txt(self, filename: str) -> str:
        text = next((t for n, t in self.txt_files if n == filename), "")
        msgs = {m["role"]: m["content"] for m in txt_to_record(filename, text)["messages"]}
        return f"[요청]\n{msgs.get('user', '')}\n\n[정답 가사]\n{msgs.get('assistant', '')}"

    def build_from_txt(self) -> None:
        if self._guard():
            return
        if len(self.txt_files) < 2:
            self.error = "가사 .txt 를 2개 이상 올리세요. (train/valid 로 나누려면 최소 2곡)"
            return
        jsonl = self.sync_command(lambda: txt_files_to_jsonl(self.txt_files))
        if jsonl is None:
            return
        if self.build_dataset(jsonl):
            self.txt_files = []

    # --- ① 데이터셋: JSONL (고급) ---
    def preview_jsonl(self, text: str) -> None:
        if not text.strip():
            self.error = "JSONL 내용이 비어 있습니다."
            return
        _, report = self.svc.parse_jsonl(text)
        self.jsonl_report = report

    def build_dataset(self, text: str) -> bool:
        if self._guard():
            return False
        if not text.strip():
            self.error = "JSONL 을 올리거나 붙여 넣으세요."
            return False
        result = self.sync_command(lambda: self.svc.build(text))
        if result:
            ds, report = result
            self.jsonl_report = report
            self._reload_datasets()
            self.dataset_id = ds.id
            self.info = f"데이터셋 {self.svc.label(ds)} 을 만들었습니다 ({self.dataset_counts(ds)}). 다음: 학습"
            return True
        return False

    # --- ② 학습 ---

    def start_training(self, form: dict[str, Any] | None = None) -> None:
        form = self.form if form is None else form

        def _num(v: Any, cast: type) -> Any:
            return cast(v) if v not in (None, "") else None
        if self.training:
            self.error = "학습이 이미 실행 중입니다."
            return
        if not self.dataset_id:
            self.error = "데이터셋을 먼저 만들거나 고르세요."
            return
        config = {
            "model_name": (self.model_title or "").strip(),
            "export_dir": (self.export_dir or "").strip(),
            "iters": _num(form.get("iters"), int) or 200,
            "batch_size": _num(form.get("batch_size"), int) or 1,
            "learning_rate": _num(form.get("learning_rate"), float),
            "num_layers": _num(form.get("num_layers"), int),
            "max_seq_length": _num(form.get("max_seq_length"), int),
            "save_every": _num(form.get("save_every"), int),
        }
        if not config["model_name"]:
            self.error = "맨 위 '결과 모델 이름'을 입력하세요. 모델 목록에 이 이름으로 표시됩니다."
            return
        if self.sync_command(lambda: LyricsTrainConfig(**config)) is None:
            return
        if self.sync_command(lambda: self.svc.resolve_export_dir(config["model_name"], config["export_dir"])) is None:
            return
        self.losses, self.val_losses, self.job_logs = [], [], []
        job = self.sync_command(lambda: self.c.jobs.submit(
            JobKind.LYRICS_TRAIN, {"dataset_id": self.dataset_id, "config": config}, dataset_id=self.dataset_id))
        if job:
            self.tracker.track(job)

    async def cancel(self) -> None:
        if self.job:
            await self.run_command(lambda: self.c.jobs.cancel(self.job.id))

    def _on_job(self, job: Job) -> None:
        self.job = job
        if job.status == JobStatus.FAILED:
            self.error = job.error or "학습이 실패했습니다."
        elif job.status == JobStatus.SUCCEEDED:
            self.info = "가사 LoRA 학습 완료. '모델 · 카테고리' 탭과 곡 만들기의 '가사 모델'에서 쓸 수 있습니다."

    def _on_log(self, line: str) -> None:
        if m := re.search(r"Iter\s+(\d+):\s*Train loss\s+([0-9.]+)", line):
            self.losses = self.losses + [(int(m.group(1)), float(m.group(2)))]
        elif m := re.search(r"Iter\s+(\d+):\s*Val loss\s+([0-9.]+)", line):
            self.val_losses = self.val_losses + [(int(m.group(1)), float(m.group(2)))]
        self.job_logs = (self.job_logs + [line])[-300:]
