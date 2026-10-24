"""Library / Jobs / Settings / Training ViewModel."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.adapters.ports import CapabilityReport
from app.core.observer import DomainEvent, ObservableProperty, Topics
from app.domain.entities import Category, Dataset, Job, JobEvent, Song
from app.domain.enums import JobKind, JobStatus
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
        self.tracker = JobTracker(self, self._on_job, self._on_log)
        self._reload_datasets(select_latest=True)

    # --- 조회 ---
    @property
    def training(self) -> bool:
        """학습 작업이 실행 중인지. (BaseViewModel.busy 와 이름이 겹치지 않게 따로 둔다)"""
        return self.job is not None and not self.job.status.is_terminal

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
        self.dataset_id = dataset_id or None

    # --- ① 데이터셋: 가사 .txt ---
    MAX_TXT_BYTES = 200_000

    def add_txt(self, filename: str, data: bytes) -> None:
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
        self.txt_files = [(n, t) for n, t in self.txt_files if n != filename]

    def clear_txt(self) -> None:
        self.txt_files = []

    def preview_txt(self, filename: str) -> str:
        text = next((t for n, t in self.txt_files if n == filename), "")
        rec = txt_to_record(filename, text)
        return f"[요청]\n{rec['messages'][0]['content']}\n\n[정답 가사]\n{rec['messages'][1]['content']}"

    def build_from_txt(self) -> None:
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

    def start_training(self, form: dict[str, Any]) -> None:
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
