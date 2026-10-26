"""음악 LoRA 학습 파이프라인 (MUS-001 ~ MUS-005).

흐름:
  ① 업로드  : 오디오를 workspace 에 임시 저장(staging) → '데이터셋 버전 만들기'
              → workspace/categories/<cid>/music/source/vNNN/  (원본 보관)
              → ACE-Step-1.5/datasets/studio/<cid>/vNNN/audio/ (ACE 가 읽을 복사본)
  ② 스캔    : POST /v1/dataset/scan   {audio_dir, dataset_name, custom_tag, tag_position, all_instrumental}
  ③ 라벨링  : POST /v1/dataset/auto_label_async → 상태 폴링 → GET /v1/dataset/samples
     검수    : PUT  /v1/dataset/sample/{idx}  (샘플별 캡션/가사/연주곡 수정)
  ④ 전처리  : POST /v1/dataset/save {save_path} → POST /v1/dataset/preprocess_async {output_dir}
              → 상태 폴링 → 생성된 .pt 개수 검증 (0개면 실패)
  ⑤ 학습    : POST /v1/training/start {tensor_dir, lora_output_dir, ...} → GET /v1/training/status 폴링
  ⑥ Export  : POST /v1/training/export {export_path, lora_output_dir} → 어댑터 구조 검증
              → 모델 레지스트리 등록 → (선택) 카테고리 활성 음악 LoRA 로 연결

ACE 서버의 데이터셋/LoRA 상태는 서버 전체에 하나뿐이므로 모든 단계는 music_engine 락 안에서 실행한다.
요청 본문은 Capability Registry 가 설치본 OpenAPI 스키마로 필터링한다(추측 필드 전송 없음).
응답 형식은 문서화되지 않았으므로 상태 응답은 일반 키로 해석하고 원문(raw)을 로그에 남긴다.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.adapters.ports import MusicEnginePort
from app.core.errors import CapabilityUnsupportedError, EngineResponseError, ValidationError
from app.domain.entities import Dataset, utcnow
from app.domain.enums import DatasetKind, Engine, JobKind, JobStatus
from app.repositories.content_repo import DatasetRepository
from app.services.category_service import CategoryService, validate_adapter_dir
from app.services.job_service import JobContext, JobHandler
from app.services.music_service import MusicService
from app.services.resource_lock import ResourceLock
from app.utils.fileio import atomic_write_bytes, atomic_write_json, sha256_file
from app.utils.path_guard import PathGuard, safe_filename

log = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
# 실제 파일 헤더 확인 (확장자만 믿지 않음)
_MAGIC = {
    ".mp3": (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"),
    ".wav": (b"RIFF",),
    ".flac": (b"fLaC",),
    ".ogg": (b"OggS",),
}

_DONE = {"completed", "complete", "done", "success", "succeeded", "finished", "idle_completed"}
_FAILED = {"failed", "error", "errored", "cancelled", "canceled", "stopped_with_error"}


# ---------------------------------------------------------------------------
# 응답 해석 (문서화되지 않은 응답 → 일반 키로 해석, 원문 보존)
# ---------------------------------------------------------------------------
def _first(d: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


class StatusSummary(BaseModel):
    state: str = ""
    running: bool | None = None
    progress: float | None = None
    message: str = ""
    error: str = ""
    epoch: int | None = None
    total_epochs: int | None = None
    step: int | None = None
    loss: float | None = None
    current: int | None = None
    total: int | None = None
    done: bool = False
    failed: bool = False
    raw: Any = None

    def progress_text(self) -> str:
        if self.current is not None and self.total:
            return f"{self.current}/{self.total}"
        return f"{self.progress * 100:.0f}%" if self.progress is not None else ""


def summarize_status(data: Any) -> StatusSummary:
    d = data if isinstance(data, dict) else {}
    if isinstance(d.get("status"), dict):  # {"status": {...}} 형태
        d = {**d, **d["status"]}
    state = str(_first(d, "status", "state", "phase") or "").lower()
    running = _first(d, "is_training", "is_running", "running", "in_progress", "training")
    progress = _first(d, "progress", "percent", "percentage")
    try:
        progress = float(progress) if progress is not None else None
        if progress is not None and progress > 1:
            progress /= 100.0
    except (TypeError, ValueError):
        progress = None

    def _num(*keys: str, cast: type = int) -> Any:
        v = _first(d, *keys)
        try:
            return cast(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    error = str(_first(d, "error", "error_message", "exception") or "")
    summary = StatusSummary(
        state=state,
        running=bool(running) if isinstance(running, (bool, int)) else None,
        progress=progress,
        message=str(_first(d, "message", "msg", "status_message", "detail") or ""),
        error=error,
        epoch=_num("current_epoch", "epoch"),
        total_epochs=_num("total_epochs", "train_epochs", "epochs", "max_epochs"),
        step=_num("current_step", "global_step", "step"),
        loss=_num("loss", "current_loss", "train_loss", "last_loss", "avg_loss", cast=float),
        raw=data,
    )
    if summary.progress is None:
        cur = _num("processed", "completed", "current", "done_count", "finished", "index", cast=float)
        tot = _num("total", "total_samples", "total_count", "count", cast=float)
        if cur is not None and tot:
            summary.progress = max(0.0, min(cur / tot, 1.0))
            summary.current, summary.total = int(cur), int(tot)
    summary.failed = state in _FAILED or bool(error)
    summary.done = (state in _DONE) or (progress is not None and progress >= 1.0 and not summary.failed)
    if summary.progress is None and summary.epoch is not None and summary.total_epochs:
        summary.progress = min(summary.epoch / summary.total_epochs, 1.0)
    return summary


def extract_task_id(data: Any) -> str | None:
    if isinstance(data, dict):
        v = _first(data, "task_id", "taskId", "id", "job_id")
        return str(v) if v else None
    return None


def extract_samples(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)]
    if isinstance(data, dict):
        for key in ("samples", "items", "data", "dataset"):
            if isinstance(data.get(key), list):
                return [s for s in data[key] if isinstance(s, dict)]
    return []


def _short(raw: Any, limit: int = 400) -> str:
    text = raw if isinstance(raw, str) else __import__("json").dumps(raw, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def count_tensors(tensor_dir: Path) -> int:
    return sum(1 for _ in Path(tensor_dir).rglob("*.pt")) if Path(tensor_dir).exists() else 0


def find_adapter_dir(root: Path) -> Path | None:
    """export 결과에서 adapter_config.json + adapter_model.safetensors 가 있는 가장 얕은 폴더."""
    root = Path(root)
    candidates = sorted((p.parent for p in root.rglob("adapter_config.json")), key=lambda p: len(p.parts))
    for c in candidates:
        if validate_adapter_dir(Engine.ACE, c)[0]:
            return c
    return None


# ---------------------------------------------------------------------------
# 데이터셋 관리
# ---------------------------------------------------------------------------
class MusicTrainConfig(BaseModel):
    """학습 설정. None 이면 ACE 서버 기본값 사용. 1000 epoch 같은 긴 기본값 금지 → 10."""

    train_epochs: int = Field(default=10, ge=1, le=2000)
    learning_rate: float | None = Field(default=None, gt=0)
    lora_rank: int | None = Field(default=None, ge=1, le=512)
    lora_alpha: int | None = Field(default=None, ge=1, le=1024)
    train_batch_size: int | None = Field(default=1, ge=1, le=16)
    gradient_accumulation: int | None = Field(default=None, ge=1, le=64)
    save_every_n_epochs: int | None = Field(default=5, ge=1)
    training_seed: int | None = None
    base_model: str = "acestep-v15-turbo"
    model_name: str = Field(default="", max_length=80)   # 결과 모델 이름 (사용자 지정)
    export_dir: str = ""                                  # 결과 저장 폴더 (비우면 기본 위치)


class MusicDatasetService:
    def __init__(self, engine: MusicEnginePort, categories: CategoryService, datasets: DatasetRepository,
                 guard: PathGuard, ace_share_dir: Path, lock: ResourceLock, *, max_upload_mb: int = 200) -> None:
        self.engine = engine
        self.categories = categories
        self.datasets = datasets
        self.guard = guard
        self.ace_share_dir = Path(ace_share_dir)
        self.lock = lock
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self.server_dataset_id: str | None = None   # ACE 서버 메모리에 올라가 있는 데이터셋
        self.init_timeout_s = 1800.0                # 모델 로드 대기 (첫 로드는 오래 걸림)
        self._dit_ready = False                     # 이 앱 세션에서 /v1/init 으로 올렸는지
        self._llm_ready = False
        self.samples: dict[str, list[dict[str, Any]]] = {}

    # --- 경로 (음악 학습은 카테고리와 무관한 '라이브러리' 폴더를 쓴다) ---
    LIBRARY = "library"

    def library_source(self) -> Path:
        """workspace/library/music/source — 원본 오디오 보관 위치."""
        root = self.categories.workspace.categories_dir.parent
        path = self.guard.child(root, self.LIBRARY, "music", "source")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def staging_dir(self) -> Path:
        return self.library_source() / "_staging"

    def ace_dir(self, dataset: Dataset) -> Path:
        # 예전에 카테고리 안에서 만든 데이터셋은 원래 위치를 그대로 쓴다
        owner = dataset.category_id or self.LIBRARY
        return self.guard.child(self.ace_share_dir, owner, f"v{dataset.version:03d}")

    # --- ① 업로드 ---
    def stage_upload(self, filename: str, data: bytes) -> Path:
        name = safe_filename(filename)
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXTS:
            raise ValidationError(f"지원하지 않는 형식입니다: {ext or '확장자 없음'} (mp3/wav/flac/ogg/m4a)")
        if len(data) > self.max_upload_bytes:
            raise ValidationError(f"파일이 너무 큽니다 (최대 {self.max_upload_bytes // 1024 // 1024}MB)")
        magic = _MAGIC.get(ext)
        if magic and not data.startswith(magic):
            raise ValidationError(f"{name}: 파일 내용이 {ext} 형식이 아닙니다.")
        staging = self.staging_dir()
        dest = staging / name
        n = 2
        while dest.exists():
            dest = staging / f"{Path(name).stem}_{n}{ext}"
            n += 1
        return atomic_write_bytes(dest, data)

    def staged_files(self) -> list[Path]:
        d = self.staging_dir()
        return sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")) if d.exists() else []

    def remove_staged(self, name: str) -> None:
        path = self.guard.child(self.staging_dir(), safe_filename(name))
        if path.exists():
            path.unlink()

    def create_dataset(self, rights_confirmed: bool, source_note: str = "") -> Dataset:
        files = self.staged_files()
        if not files:
            raise ValidationError("업로드한 오디오가 없습니다.")
        seen: dict[str, str] = {}
        dupes: list[str] = []
        for f in files:
            h = sha256_file(f)
            if h in seen:
                dupes.append(f.name)
            seen.setdefault(h, f.name)
        if dupes:
            raise ValidationError(f"같은 내용의 파일이 있습니다: {', '.join(dupes)} — 하나만 남기세요.")
        # DB 를 지워도 디스크에 예전 vNNN 폴더가 남아 있을 수 있다 → 덮어쓰지 않고 빈 번호로 건너뛴다
        version = self.datasets.next_version(None, DatasetKind.MUSIC)
        while True:
            src_dir = self.guard.child(self.library_source(), f"v{version:03d}")
            dataset = Dataset(category_id=None, kind=DatasetKind.MUSIC, version=version, status="uploaded",
                              source_path=str(src_dir),
                              rights_status="confirmed" if rights_confirmed else "unverified")
            if not src_dir.exists() and not self.ace_dir(dataset).exists():
                break
            version += 1
        ace_audio = self.ace_dir(dataset) / "audio"
        src_dir.mkdir(parents=True)
        ace_audio.mkdir(parents=True)
        for f in files:
            shutil.copy2(f, ace_audio / f.name)   # ACE 가 읽을 복사본
            shutil.move(str(f), str(src_dir / f.name))  # 원본 보관
        atomic_write_json(src_dir / "manifest.json", {
            "dataset_id": dataset.id, "version": version,
            "files": {name: h for h, name in seen.items()}, "rights_confirmed": rights_confirmed,
            "source_note": source_note, "created_at": utcnow().isoformat(),
        })
        self.datasets.add(dataset)
        return dataset

    def list(self) -> list[Dataset]:
        """음악 데이터셋 전체 (예전에 카테고리 안에서 만든 것도 포함)."""
        return self.datasets.list_kind(DatasetKind.MUSIC)

    @staticmethod
    def label(ds: Dataset) -> str:
        return f"{'L' if ds.category_id is None else 'C'}-v{ds.version:03d}"

    def get(self, dataset_id: str) -> Dataset:
        ds = self.datasets.get(dataset_id)
        if ds is None or ds.kind != DatasetKind.MUSIC:
            raise ValidationError("음악 데이터셋을 찾을 수 없습니다.", detail=dataset_id)
        return ds

    def set_status(self, dataset: Dataset, status: str, **fields: str) -> Dataset:
        dataset.status = status
        for k, v in fields.items():
            setattr(dataset, k, v)
        return self.datasets.save(dataset)

    # --- 결과 저장 위치 ---
    def resolve_export_dir(self, cfg: "MusicTrainConfig", run_dir: Path, version: int) -> Path:
        """사용자가 고른 폴더 아래에 '모델 이름' 하위 폴더를 새로 만든다. 기존 폴더/보호 경로는 거부."""
        name = safe_filename(cfg.model_name.strip() or f"lora_v{version:03d}")
        if not cfg.export_dir.strip():
            return run_dir / name          # 기본 위치도 '모델 이름' 폴더로
        base = Path(cfg.export_dir).expanduser().resolve()
        if not base.is_dir():
            raise ValidationError("결과 저장 폴더가 없습니다.", detail=str(base))
        target = base / name
        for protected in self.guard.protected:
            if target == protected or protected in target.parents:
                raise ValidationError("기존 ACE 자산 폴더(lora_output, checkpoints 등)에는 저장할 수 없습니다.",
                                      detail=str(target))
        if target.exists():
            raise ValidationError(f"'{target}' 폴더가 이미 있습니다. 덮어쓰지 않습니다. 모델 이름을 바꾸세요.")
        return target

    def check_export_dir(self, export_dir: str, model_name: str) -> Path | None:
        """학습 시작 전에 화면에서 미리 검사 (같은 규칙)."""
        if not export_dir.strip():
            return None
        cfg = MusicTrainConfig(model_name=model_name, export_dir=export_dir)
        return self.resolve_export_dir(cfg, Path("/nonexistent"), 0)

    # --- 검수 ---
    async def fetch_samples(self, dataset_id: str) -> list[dict[str, Any]]:
        self._require_loaded(dataset_id)
        data = await self.engine.invoke("dataset_samples")
        samples = extract_samples(data)
        self.samples[dataset_id] = samples
        return samples

    async def update_sample(self, dataset_id: str, index: int, changes: dict[str, Any]) -> None:
        self._require_loaded(dataset_id)
        async with self.lock.music_engine():
            await self.engine.invoke("dataset_sample_update", {"sample_idx": index, **changes},
                                     path_params={"sample_idx": index})
        cached = self.samples.get(dataset_id)
        if cached and 0 <= index < len(cached):
            cached[index].update(changes)

    def _require_loaded(self, dataset_id: str) -> None:
        if self.server_dataset_id != dataset_id:
            raise ValidationError("ACE 서버에 다른 데이터셋이 올라가 있습니다. 이 데이터셋을 먼저 '스캔'하세요.")

    # --- ACE 모델 준비 ---
    async def ensure_models(self, ctx: JobContext, *, need_llm: bool = False, force: bool = False) -> None:
        """라벨링·전처리·학습 전에 ACE 모델을 메모리에 올린다.

        ACE 서버는 모델을 '첫 생성 요청' 때만 자동 로드한다. 전처리는 모델이 없으면 오류 없이
        0개 처리 후 'completed' 를 돌려주므로, 오류를 기다리지 않고 미리 /v1/init 을 호출한다.
        이 앱이 켜져 있는 동안 한 번 올렸으면 다시 호출하지 않는다.
        """
        if not force and self._dit_ready and (self._llm_ready or not need_llm):
            return
        # ACE 의 /v1/init 은 이미 올라가 있어도 무조건 다시 로드한다 → 메모리 두 배로 'MPS out of memory'.
        # 그래서 먼저 ACE 에 실제로 올라가 있는지 물어보고, 음악 모델(DiT)이 있으면 다시 올리지 않는다.
        # (분석용 LM 은 라벨링 요청 때 ACE 가 필요하면 스스로 올리고, 안 올라가 있다고 하면 그때 force 로 재시도)
        if not force:
            state = await self.ace_model_state()
            if state is not None and state[0]:
                self._dit_ready = True
                self._llm_ready = self._llm_ready or state[1]
                ctx.log("ACE 모델이 이미 메모리에 있습니다 — 다시 로드하지 않습니다."
                        + ("" if state[1] or not need_llm else " (분석용 LM 은 필요할 때 올립니다)"))
                return
        ctx.log("ACE 모델 로드 중 (/v1/init" + (", 분석용 LM 포함" if need_llm else "")
                + ") — 처음에는 수 분 걸릴 수 있습니다")
        init = await self.engine.invoke("init_model", {"init_llm": need_llm}, timeout=self.init_timeout_s)
        self._dit_ready = True
        self._llm_ready = self._llm_ready or need_llm
        ctx.log(f"ACE 모델 로드 완료 — ACE 응답: {_short(init)}", details={"raw": init})

    async def ace_model_state(self) -> tuple[bool, bool] | None:
        """(음악 모델 로드됨, 분석용 LM 로드됨). 확인할 수 없으면 None."""
        try:
            data = await self.engine.invoke("models")
        except Exception:  # noqa: BLE001 - 확인 실패는 '모름' 으로 처리
            return None
        if isinstance(data, dict) and isinstance(data.get("data"), dict):
            data = data["data"]
        if not isinstance(data, dict) or "models" not in data:
            return None
        dit = any(isinstance(m, dict) and m.get("is_loaded") for m in data.get("models") or [])
        llm = bool(data.get("llm_initialized") or data.get("loaded_lm_model"))
        return dit, llm

    async def invoke_ready(self, ctx: JobContext, capability: str, payload: dict[str, Any] | None = None, *,
                           need_llm: bool = False) -> Any:
        """모델을 먼저 준비하고 호출. 그래도 'Model not initialized' 면(ACE 재시작 등) 다시 올리고 한 번 재시도."""
        await self.ensure_models(ctx, need_llm=need_llm)
        try:
            return await self.engine.invoke(capability, payload)
        except EngineResponseError as exc:
            if "not initialized" not in str(exc.detail).lower():
                raise
        self._dit_ready = self._llm_ready = False
        await self.ensure_models(ctx, need_llm=need_llm, force=True)
        return await self.engine.invoke(capability, payload)

    # --- 비동기 작업 폴링 공통 ---
    async def poll_task(self, ctx: JobContext, status_cap: str, latest_cap: str, task_id: str | None,
                        label: str, timeout_s: float = 6 * 3600) -> StatusSummary:
        deadline = time.monotonic() + timeout_s
        interval = 2.0
        last_msg = ""
        while True:
            ctx.check_cancelled()
            if task_id:
                data = await self.engine.invoke(status_cap, path_params={"task_id": task_id})
            else:
                data = await self.engine.invoke(latest_cap)
            st = summarize_status(data)
            if st.progress is not None:
                ctx.progress(st.progress)
            else:
                ctx.heartbeat()
            msg = " ".join(x for x in (st.message or st.state, st.progress_text()) if x)
            if msg and msg != last_msg:
                ctx.log(f"{label}: {msg}")
                last_msg = msg
            if st.failed:
                raise EngineResponseError(f"{label} 실패", detail=st.error or st.raw)
            if st.done:
                ctx.log(f"{label} 완료 — ACE 응답: {_short(st.raw)}", details={"raw": st.raw})
                return st
            if time.monotonic() > deadline:
                raise EngineResponseError(f"{label} 시간 초과")
            await asyncio.sleep(interval)
            interval = min(interval * 1.3, 10.0)


# ---------------------------------------------------------------------------
# 단계별 잡 핸들러 (Command)
# ---------------------------------------------------------------------------
class _MusicStepHandler(JobHandler):
    def __init__(self, svc: MusicDatasetService) -> None:
        self.svc = svc

    def validate(self, params: dict[str, Any]) -> None:
        if not params.get("dataset_id"):
            raise ValidationError("dataset_id 가 필요합니다.")
        self.svc.get(params["dataset_id"])

    async def _require(self, *caps: str) -> None:
        report = await self.svc.engine.capabilities()
        missing = [c for c in caps if not report.is_usable(c)]
        if missing:
            raise CapabilityUnsupportedError(f"ACE 서버에서 확인되지 않은 기능: {', '.join(missing)}")


class MusicScanHandler(_MusicStepHandler):
    kind = JobKind.MUSIC_SCAN

    def describe(self, params: dict[str, Any]) -> str:
        return "ACE 데이터 스캔"

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        await self._require("dataset_scan", "dataset_samples")
        ds = self.svc.get(ctx.params["dataset_id"])
        audio_dir = self.svc.ace_dir(ds) / "audio"
        payload = {
            "audio_dir": str(audio_dir),
            "dataset_name": f"{(ds.category_id or 'lib')[:8]}_v{ds.version:03d}",
            "custom_tag": ctx.params.get("custom_tag") or None,
            # ACE 기본값은 'replace' (캡션을 태그로 통째로 바꿈) → 캡션을 살리고 태그를 앞에 붙이는 'prepend' 로
            "tag_position": ctx.params.get("tag_position") or ("prepend" if ctx.params.get("custom_tag") else None),
            "all_instrumental": bool(ctx.params.get("all_instrumental", False)),
        }
        async with self.svc.lock.music_engine():
            ctx.log(f"스캔: {audio_dir}")
            await self.svc.engine.invoke("dataset_scan", payload)
            self.svc.server_dataset_id = ds.id
            samples = await self.svc.fetch_samples(ds.id)
        if not samples:
            raise EngineResponseError("스캔 결과 샘플이 0개입니다. 오디오 형식/경로를 확인하세요.")
        self.svc.set_status(ds, "scanned")
        ctx.log(f"샘플 {len(samples)}개 스캔 완료")
        return {"sample_count": len(samples)}


class MusicLabelHandler(_MusicStepHandler):
    kind = JobKind.MUSIC_LABEL

    def describe(self, params: dict[str, Any]) -> str:
        return "ACE 자동 라벨링"

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        await self._require("dataset_auto_label", "dataset_samples")
        ds = self.svc.get(ctx.params["dataset_id"])
        self.svc._require_loaded(ds.id)  # noqa: SLF001
        async with self.svc.lock.music_engine():
            ctx.log("자동 라벨링 시작 (캡션/장르/BPM/키 — AI 추정값이므로 검수 필요)")
            resp = await self.svc.invoke_ready(ctx, "dataset_auto_label", ctx.params.get("options") or {},
                                               need_llm=True)
            await self.svc.poll_task(ctx, "auto_label_status", "auto_label_status_latest", extract_task_id(resp),
                                     "자동 라벨링")
            samples = await self.svc.fetch_samples(ds.id)
        self.svc.set_status(ds, "labeled")
        return {"sample_count": len(samples)}


class MusicPreprocessHandler(_MusicStepHandler):
    kind = JobKind.MUSIC_PREPROCESS

    def describe(self, params: dict[str, Any]) -> str:
        return "ACE 텐서 전처리"

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        await self._require("dataset_save", "preprocess")
        ds = self.svc.get(ctx.params["dataset_id"])
        self.svc._require_loaded(ds.id)  # noqa: SLF001
        base = self.svc.ace_dir(ds)
        json_path = base / "dataset.json"
        tensor_dir = base / "tensors"
        if count_tensors(tensor_dir):
            raise ValidationError("이 버전의 텐서가 이미 있습니다. 덮어쓰지 않습니다. 새 데이터셋 버전을 만드세요.")
        expected = len(self.svc.samples.get(ds.id, []))
        async with self.svc.lock.music_engine():
            ctx.log(f"데이터셋 JSON 저장: {json_path}")
            await self.svc.engine.invoke("dataset_save", {
                "save_path": str(json_path), "dataset_name": f"{(ds.category_id or 'lib')[:8]}_v{ds.version:03d}",
            })
            tensor_dir.mkdir(parents=True, exist_ok=True)
            ctx.log(f"전처리 시작 → {tensor_dir}")
            resp = await self.svc.invoke_ready(ctx, "preprocess", {"output_dir": str(tensor_dir), "skip_existing": False})
            final = await self.svc.poll_task(ctx, "preprocess_status", "preprocess_status_latest",
                                             extract_task_id(resp), "전처리")
        made = count_tensors(tensor_dir)
        if made == 0 and expected == 0:
            ctx.log("ACE 서버에 올라간 샘플 목록이 비어 있습니다. ② 스캔을 다시 하세요.", "warning")
        if made == 0:
            files = sorted(p.name for p in tensor_dir.rglob("*") if p.is_file())[:10]
            raise EngineResponseError(
                "전처리 결과 텐서가 0개입니다 (ACE 는 완료라고 응답). 오디오 디코딩 실패일 가능성이 큽니다 — "
                "ACE 서버 터미널의 오류와 FFmpeg 8(DYLD_LIBRARY_PATH) 전달 여부를 확인하세요.",
                detail=f"출력 폴더 {tensor_dir} 파일: {files or '없음'} | ACE 응답: {_short(final.raw)}",
            )
        if expected and made < expected:
            ctx.log(f"주의: 샘플 {expected}개 중 텐서 {made}개만 생성되었습니다.", "warning")
        self.svc.set_status(ds, "preprocessed", json_path=str(json_path), tensor_path=str(tensor_dir))
        return {"tensor_count": made, "expected": expected, "tensor_dir": str(tensor_dir)}


class MusicTrainHandler(_MusicStepHandler):
    kind = JobKind.MUSIC_TRAIN

    MAX_STATUS_ERRORS = 60   # 5초 간격 × 60 = 약 5분 동안 계속 실패하면 발산으로 보고 멈춘다

    def __init__(self, svc: MusicDatasetService, music: MusicService, poll_s: float = 5.0) -> None:
        super().__init__(svc)
        self.music = music
        self.poll_s = poll_s

    def describe(self, params: dict[str, Any]) -> str:
        return "ACE 음악 LoRA 학습"

    def validate(self, params: dict[str, Any]) -> None:
        super().validate(params)
        ds = self.svc.get(params["dataset_id"])
        if ds.rights_status != "confirmed":
            raise ValidationError("학습 권리가 확인되지 않은 데이터셋입니다. 업로드 시 권리 확인이 필요합니다.")
        if not ds.tensor_path or not count_tensors(Path(ds.tensor_path)):
            raise ValidationError("전처리된 텐서가 없습니다. 먼저 전처리를 완료하세요.")
        MusicTrainConfig(**params.get("config", {}))

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        await self._require("training_start", "training_status", "training_stop", "training_export")
        ds = self.svc.get(ctx.params["dataset_id"])
        cfg = MusicTrainConfig(**ctx.params.get("config", {}))
        version = self.svc.categories.models.next_version(None, Engine.ACE)
        run_no = version
        while (self.svc.ace_dir(ds) / "runs" / f"lora_v{run_no:03d}").exists():   # 남아 있는 예전 폴더는 건너뜀
            run_no += 1
        run_dir = self.svc.ace_dir(ds) / "runs" / f"lora_v{run_no:03d}"
        output_dir = run_dir / "output"
        export_dir = self.svc.resolve_export_dir(cfg, run_dir, version)
        output_dir.mkdir(parents=True)
        ctx.log(f"결과 저장 위치: {export_dir}")

        payload = {k: v for k, v in cfg.model_dump().items()
                   if k not in ("base_model", "model_name", "export_dir") and v is not None}
        payload.update({"tensor_dir": ds.tensor_path, "output_dir": str(output_dir)})

        losses: list[tuple[int, float]] = []
        nonfinite = 0
        async with self.svc.lock.music_engine():
            self.music.invalidate_lora_state()
            ctx.log(f"학습 시작: epochs={cfg.train_epochs}, 출력={output_dir}")
            await self.svc.invoke_ready(ctx, "training_start", payload)
            seen_running = False
            status_errors = 0
            try:
                while True:
                    ctx.check_cancelled()
                    try:
                        raw_status = await self.svc.engine.invoke("training_status")
                    except Exception as exc:  # noqa: BLE001
                        # ACE 서버는 loss 가 NaN 이면 상태 JSON 을 못 만들어 500 을 돌려준다
                        # (ValueError: Out of range float values are not JSON compliant: nan).
                        # 한두 번은 학습 초반/순간적인 값일 수 있으므로 바로 멈추지 않고 계속 확인한다.
                        status_errors += 1
                        if status_errors == 1 or status_errors % 6 == 0:
                            ctx.log(f"ACE 학습 상태 응답 오류 {status_errors}회 (loss 가 NaN 일 때 생김). "
                                    f"학습은 멈추지 않고 계속 확인합니다.", "warning")
                        if status_errors >= self.MAX_STATUS_ERRORS:
                            raise EngineResponseError(
                                "ACE 학습 상태를 계속 읽을 수 없습니다. loss 가 NaN 으로 발산했을 가능성이 큽니다 — "
                                "learning rate 를 낮추거나(예: 1e-4 → 5e-5) 데이터를 확인한 뒤 다시 학습하세요.",
                                detail=str(exc)[:500]) from exc
                        await asyncio.sleep(self.poll_s)
                        continue
                    if status_errors:
                        ctx.log(f"ACE 학습 상태 응답 복구 (오류 {status_errors}회 후)")
                        status_errors = 0
                    st = summarize_status(raw_status)
                    if st.running:
                        seen_running = True
                    if st.progress is not None:
                        ctx.progress(st.progress)
                    else:
                        ctx.heartbeat()
                    if st.loss is not None:
                        x = st.step if st.step is not None else (st.epoch or len(losses))
                        if not losses or losses[-1] != (x, st.loss):
                            losses.append((x, st.loss))
                            ctx.log(f"epoch {st.epoch}/{st.total_epochs or cfg.train_epochs} "
                                    f"step {st.step} loss {st.loss:.4f}")
                    if "non-finite" in (st.message + st.error).lower():
                        nonfinite += 1
                        ctx.log(f"Non-finite gradients 경고 ({nonfinite}회) — 반복되면 학습률을 낮추세요.", "warning")
                    if st.failed:
                        raise EngineResponseError("ACE 학습 실패", detail=st.error or st.raw)
                    if st.done or (seen_running and st.running is False):
                        ctx.log("학습 종료", details={"raw": st.raw})
                        break
                    await asyncio.sleep(self.poll_s)
            except (asyncio.CancelledError, Exception):
                ctx.log("학습 중지 요청 → ACE 서버에 stop 전송", "warning")
                try:
                    await self.svc.engine.invoke("training_stop")
                except Exception:  # noqa: BLE001
                    log.exception("training_stop 실패")
                raise

            ctx.set_status(JobStatus.EXPORTING)
            ctx.log(f"Export → {export_dir}")
            await self.svc.engine.invoke("training_export", {
                "export_path": str(export_dir), "lora_output_dir": str(output_dir),
            })

        ctx.set_status(JobStatus.VALIDATING_ADAPTER)
        adapter = find_adapter_dir(export_dir) or find_adapter_dir(output_dir)
        if adapter is None:
            raise ValidationError("학습은 끝났지만 adapter_config.json / adapter_model.safetensors 를 찾지 못했습니다.",
                                  detail=str(run_dir))
        model = self.svc.categories.register_model(
            None, Engine.ACE, cfg.base_model, adapter, dataset_id=ds.id,
            # export_root: 이 앱이 만든 결과 폴더 → 이름을 바꾸면 이 폴더 이름도 같이 바뀐다
            train_config={**payload, "nonfinite_warnings": nonfinite, "export_root": str(export_dir)},
            notes=f"앱에서 학습 (데이터셋 {MusicDatasetService.label(ds)})",
            display_name=cfg.model_name.strip() or f"데이터셋 {MusicDatasetService.label(ds)} 학습",
        )
        ctx.log(f"모델 등록: {model.label} — '모델·카테고리' 탭에서 카테고리에 넣으세요.")
        self.svc.set_status(ds, "trained")
        ctx.log("로드 성공 ≠ 실제 적용. 곡 만들기의 A/B 비교로 효과를 확인하세요.", "info")
        return {"model_id": model.id, "adapter_path": str(adapter), "losses": losses[-500:],
                "nonfinite_warnings": nonfinite}
