"""음악 LoRA 학습 화면 VM: 업로드 → 스캔 → 라벨링·검수 → 전처리 → 학습 → Export·등록."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.adapters.ports import CapabilityReport
from app.core.observer import ObservableProperty, Topics
from app.domain.entities import Category, Dataset, Job, ModelRecord
from app.domain.enums import Engine, JobKind, JobStatus
from app.services.music_training_service import MusicTrainConfig
from app.viewmodels.base import BaseViewModel, JobTracker
from app.viewmodels.models_vm import MusicModelsViewModel  # noqa: F401  (화면에서 같이 import)

REQUIRED_CAPS = ("dataset_scan", "dataset_samples", "dataset_auto_label", "dataset_save", "preprocess",
                 "training_start", "training_status", "training_stop", "training_export")

# 데이터셋 상태 → 다음에 할 수 있는 단계
STEP_OF_STATUS = {"uploaded": 1, "scanned": 2, "labeled": 3, "preprocessed": 4, "trained": 5}
STATUS_KO = {"uploaded": "업로드됨", "scanned": "스캔됨", "labeled": "라벨링됨", "preprocessed": "전처리됨",
             "trained": "학습 완료"}

_LOSS_RE = re.compile(r"step (\S+) loss ([0-9.]+)")

STEP_NAME = {JobKind.MUSIC_SCAN: "② 스캔", JobKind.MUSIC_LABEL: "③ 자동 라벨링",
             JobKind.MUSIC_PREPROCESS: "④ 전처리", JobKind.MUSIC_TRAIN: "⑤ 학습"}


class MusicTrainingViewModel(BaseViewModel):
    staged = ObservableProperty[list[str]](factory=list)
    datasets = ObservableProperty[list[Dataset]](factory=list)
    dataset_id = ObservableProperty[str | None](None)
    samples = ObservableProperty[list[dict[str, Any]]](factory=list)
    job = ObservableProperty[Job | None](None)
    job_logs = ObservableProperty[list[str]](factory=list)
    losses = ObservableProperty[list[tuple[float, float]]](factory=list)
    capabilities = ObservableProperty[CapabilityReport | None](None)
    model_title = ObservableProperty("")   # 학습 결과 모델 이름 (맨 위 입력칸)
    export_dir = ObservableProperty("")    # 결과 저장 폴더 (비우면 기본 위치)

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.svc = self.c.music_datasets
        self.tracker = JobTracker(self, self._on_job, self._on_log)
        self._reload_staged()
        self._reload_datasets(select_latest=True)

    # --- 조회 ---
    @property
    def dataset(self) -> Dataset | None:
        return next((d for d in self.datasets if d.id == self.dataset_id), None)

    @property
    def step(self) -> int:
        return STEP_OF_STATUS.get(self.dataset.status, 0) if self.dataset else 0

    @property
    def step_name(self) -> str:
        return STEP_NAME.get(self.job.kind, self.job.kind.value) if self.job else ""

    @property
    def last_log(self) -> str:
        return self.job_logs[-1] if self.job_logs else ""

    @property
    def busy_job(self) -> bool:
        return self.job is not None and not self.job.status.is_terminal

    @property
    def missing_caps(self) -> list[str]:
        if self.capabilities is None:
            return []
        return [c for c in REQUIRED_CAPS if not self.capabilities.is_usable(c)]

    @property
    def server_has_dataset(self) -> bool:
        return self.dataset_id is not None and self.svc.server_dataset_id == self.dataset_id

    @property
    def browse_root(self) -> Path:
        return Path.home()

    def browse(self, path: str | None = None) -> tuple[Path, list[tuple[str, bool]]]:
        """결과 저장 폴더 고르기 (홈 폴더 안, 읽기 전용 탐색)."""
        result = self.sync_command(lambda: self.c.categories.browse_dirs(self.browse_root, path))
        return result if result else (self.browse_root, [])

    def dataset_label(self, ds: Dataset) -> str:
        return self.svc.label(ds)

    def _reload_staged(self) -> None:
        self.staged = [p.name for p in self.svc.staged_files()]

    def _reload_datasets(self, select_latest: bool = False) -> None:
        self.datasets = self.svc.list()
        if select_latest or self.dataset_id not in {d.id for d in self.datasets}:
            self.select_dataset(self.datasets[-1].id if self.datasets else None)

    def select_dataset(self, dataset_id: str | None) -> None:
        self.dataset_id = dataset_id
        self.samples = list(self.svc.samples.get(dataset_id, [])) if dataset_id else []

    async def load_capabilities(self) -> None:
        async def _do() -> None:
            self.capabilities = await self.c.music_engine.capabilities()
        await self.run_command(_do)

    # --- ① 업로드 ---
    def upload(self, filename: str, data: bytes) -> None:
        if self.sync_command(lambda: self.svc.stage_upload(filename, data)):
            self._reload_staged()

    def remove_staged(self, name: str) -> None:
        self.sync_command(lambda: self.svc.remove_staged(name))
        self._reload_staged()

    def create_dataset(self, rights_confirmed: bool, source_note: str) -> None:
        if not rights_confirmed:
            self.error = "학습·생성에 사용할 권리가 있는 음원인지 확인 체크가 필요합니다."
            return
        ds = self.sync_command(lambda: self.svc.create_dataset(True, source_note),
                               success="데이터셋 버전을 만들었습니다. 다음: 스캔")
        if ds:
            self._reload_staged()
            self._reload_datasets()
            self.select_dataset(ds.id)

    # --- ②~⑤ 단계 (잡으로 실행) ---
    def scan(self, custom_tag: str, all_instrumental: bool) -> None:
        self._submit(JobKind.MUSIC_SCAN, {"custom_tag": custom_tag.strip(), "all_instrumental": all_instrumental})

    def auto_label(self, options: dict[str, Any] | None = None) -> None:
        """options: transcribe_lyrics / format_lyrics / only_unlabeled / skip_metas (설치본 스키마 확인됨)."""
        opts = {k: bool(v) for k, v in (options or {}).items()}
        self._submit(JobKind.MUSIC_LABEL, {"options": opts})

    def preprocess(self) -> None:
        self._submit(JobKind.MUSIC_PREPROCESS, {})

    def train(self, form: dict[str, Any]) -> None:
        def _num(v: Any, cast: type) -> Any:
            return cast(v) if v not in (None, "") else None
        config = {
            "train_epochs": _num(form.get("train_epochs"), int) or 10,
            "learning_rate": _num(form.get("learning_rate"), float),
            "lora_rank": _num(form.get("lora_rank"), int),
            "lora_alpha": _num(form.get("lora_alpha"), int),
            "train_batch_size": _num(form.get("train_batch_size"), int),
            "save_every_n_epochs": _num(form.get("save_every_n_epochs"), int),
            "model_name": (self.model_title or "").strip(),
            "export_dir": (self.export_dir or "").strip(),
        }
        if not config["model_name"]:
            self.error = "맨 위 '결과 모델 이름'을 입력하세요. 모델 목록에 이 이름으로 표시됩니다."
            return
        if self.sync_command(lambda: MusicTrainConfig(**config)) is None:
            return
        if config["export_dir"] and self.sync_command(
                lambda: self.svc.check_export_dir(config["export_dir"], config["model_name"])) is None:
            return
        self.losses = []
        self._submit(JobKind.MUSIC_TRAIN, {"config": config})

    async def cancel(self) -> None:
        if self.job:
            await self.run_command(lambda: self.c.jobs.cancel(self.job.id))

    # --- 검수 ---
    async def save_sample(self, index: int, changes: dict[str, Any]) -> None:
        if not self.dataset_id:
            return
        await self.run_command(lambda: self.svc.update_sample(self.dataset_id, index, changes),
                               success=f"샘플 {index + 1} 수정 저장")
        self.samples = list(self.svc.samples.get(self.dataset_id, []))

    async def reload_samples(self) -> None:
        if not self.dataset_id:
            return
        async def _do() -> None:
            self.samples = await self.svc.fetch_samples(self.dataset_id)
        await self.run_command(_do)

    # --- 내부 ---
    def _submit(self, kind: JobKind, params: dict[str, Any]) -> None:
        if not self.dataset_id:
            self.error = "데이터셋을 먼저 선택하세요."
            return
        if self.busy_job:
            self.error = "이전 단계가 아직 실행 중입니다."
            return
        self.job_logs = []
        job = self.sync_command(lambda: self.c.jobs.submit(
            kind, {"dataset_id": self.dataset_id, **params}, dataset_id=self.dataset_id))
        if job:
            self.tracker.track(job)

    def _on_log(self, line: str) -> None:
        if m := _LOSS_RE.search(line):
            try:
                x = float(m.group(1)) if m.group(1) != "None" else float(len(self.losses))
                self.losses = self.losses + [(x, float(m.group(2)))]
            except ValueError:
                pass
        self.job_logs = (self.job_logs + [line])[-300:]

    def _on_job(self, job: Job) -> None:
        self.job = job
        if job.status == JobStatus.FAILED:
            self.error = job.error or "작업이 실패했습니다."
        if not job.status.is_terminal:
            return
        self._reload_datasets()
        self.select_dataset(self.dataset_id)
        if job.status == JobStatus.SUCCEEDED:
            done = {
                JobKind.MUSIC_SCAN: f"스캔 완료: 샘플 {job.result.get('sample_count')}개. 다음: 자동 라벨링",
                JobKind.MUSIC_LABEL: "자동 라벨링 완료. 아래 샘플을 검수한 뒤 전처리하세요.",
                JobKind.MUSIC_PREPROCESS: f"전처리 완료: 텐서 {job.result.get('tensor_count')}개. 다음: 학습",
                JobKind.MUSIC_TRAIN: "학습·Export 완료. '모델·카테고리' 탭에서 카테고리에 넣으세요.",
            }
            self.info = done.get(job.kind, "완료")
