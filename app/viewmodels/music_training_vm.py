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

    # 단계 버튼: (번호, 잡 종류, 이 단계를 하려면 데이터셋이 도달해 있어야 하는 단계 번호)
    STEPS = ((2, JobKind.MUSIC_SCAN, 1), (3, JobKind.MUSIC_LABEL, 2), (4, JobKind.MUSIC_PREPROCESS, 3),
             (5, JobKind.MUSIC_TRAIN, 4))
    STEP_HINT = {JobKind.MUSIC_SCAN: "", JobKind.MUSIC_LABEL: "② 스캔을 먼저 하세요.",
                 JobKind.MUSIC_PREPROCESS: "③ 자동 라벨링을 먼저 하세요.", JobKind.MUSIC_TRAIN: "④ 전처리를 먼저 하세요."}

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.svc = self.c.music_datasets
        # 화면 입력값 (단계 실행 때 작업 기록에 같이 남겨, 다시 들어오면 그대로 되살린다)
        self.scan_opts: dict[str, Any] = {"tag": "", "instrumental": False}
        self.label_opts: dict[str, Any] = {"transcribe_lyrics": False, "format_lyrics": False, "only_unlabeled": False}
        self.train_form: dict[str, Any] = {"train_epochs": 10, "learning_rate": None, "lora_rank": None,
                                           "lora_alpha": None, "train_batch_size": 1, "save_every_n_epochs": 5}
        self.tracker = JobTracker(self, self._on_job, self._on_log)
        self._reload_staged()
        self._reload_datasets(select_latest=True)
        # 다른 화면에 다녀와도 실행 중인 단계(라벨링·전처리·학습)의 진행 표시를 다시 붙인다
        job = self.tracker.resume(STEP_NAME)
        if job and job.dataset_id:
            self.select_dataset(job.dataset_id)
        if job is not None and not job.status.is_terminal:
            self._restore(job.params)

    def _restore(self, params: dict[str, Any]) -> None:
        if "custom_tag" in params:
            self.scan_opts.update(tag=params.get("custom_tag", ""), instrumental=bool(params.get("all_instrumental")))
        if isinstance(params.get("options"), dict):
            self.label_opts.update({k: v for k, v in params["options"].items() if k in self.label_opts})
        cfg = params.get("config")
        if isinstance(cfg, dict):
            self.train_form.update({k: cfg[k] for k in self.train_form if k in cfg})
            self.model_title = cfg.get("model_name", self.model_title)
            self.export_dir = cfg.get("export_dir", self.export_dir)

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
    def locked(self) -> bool:
        """단계가 실행 중이면 화면의 모든 입력과 다른 단계 버튼을 잠근다."""
        return self.busy_job

    def _guard(self) -> bool:
        if self.locked:
            self.error = f"{self.step_name} 이(가) 실행 중입니다. 끝난 뒤에 할 수 있습니다."
            return True
        return False

    def step_ready(self, kind: JobKind) -> bool:
        """이 단계 버튼을 누를 수 있는지 (실행 중이 아니고, 데이터셋이 앞 단계를 마쳤는지)."""
        need = next(n for _, k, n in self.STEPS if k == kind)
        return not self.locked and self.dataset is not None and self.step >= need

    def running_step(self, kind: JobKind) -> bool:
        return self.locked and self.job is not None and self.job.kind == kind

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

    def choose_dataset(self, dataset_id: str | None) -> None:
        """화면에서 데이터셋 바꾸기 (실행 중에는 막음)."""
        if self._guard():
            return
        self.select_dataset(dataset_id)

    def select_dataset(self, dataset_id: str | None) -> None:
        self.dataset_id = dataset_id
        self.samples = list(self.svc.samples.get(dataset_id, [])) if dataset_id else []

    async def load_capabilities(self) -> None:
        async def _do() -> None:
            self.capabilities = await self.c.music_engine.capabilities()
        await self.run_command(_do)

    # --- ① 업로드 ---
    def upload(self, filename: str, data: bytes) -> None:
        if self._guard():
            return
        if self.sync_command(lambda: self.svc.stage_upload(filename, data)):
            self._reload_staged()

    def remove_staged(self, name: str) -> None:
        if self._guard():
            return
        self.sync_command(lambda: self.svc.remove_staged(name))
        self._reload_staged()

    def create_dataset(self, rights_confirmed: bool, source_note: str) -> None:
        if self._guard():
            return
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
    def scan(self, custom_tag: str | None = None, all_instrumental: bool | None = None) -> None:
        custom_tag = self.scan_opts["tag"] if custom_tag is None else custom_tag
        all_instrumental = self.scan_opts["instrumental"] if all_instrumental is None else all_instrumental
        self._submit(JobKind.MUSIC_SCAN, {"custom_tag": custom_tag.strip(), "all_instrumental": all_instrumental})

    def auto_label(self, options: dict[str, Any] | None = None) -> None:
        """options: transcribe_lyrics / format_lyrics / only_unlabeled / skip_metas (설치본 스키마 확인됨)."""
        opts = {k: bool(v) for k, v in (self.label_opts if options is None else options).items()}
        self._submit(JobKind.MUSIC_LABEL, {"options": opts})

    def preprocess(self) -> None:
        self._submit(JobKind.MUSIC_PREPROCESS, {})

    def train(self, form: dict[str, Any] | None = None) -> None:
        form = self.train_form if form is None else form

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
        if self._guard() or not self.dataset_id:
            return
        await self.run_command(lambda: self.svc.update_sample(self.dataset_id, index, changes),
                               success=f"샘플 {index + 1} 수정 저장")
        self.samples = list(self.svc.samples.get(self.dataset_id, []))

    async def reload_samples(self) -> None:
        if self._guard() or not self.dataset_id:
            return
        async def _do() -> None:
            self.samples = await self.svc.fetch_samples(self.dataset_id)
        await self.run_command(_do)

    # --- 내부 ---
    def _submit(self, kind: JobKind, params: dict[str, Any]) -> None:
        if not self.dataset_id:
            self.error = "데이터셋을 먼저 선택하세요."
            return
        if self._guard():
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
