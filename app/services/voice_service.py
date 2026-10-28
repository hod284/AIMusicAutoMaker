"""목소리 모델 서비스.

- 목소리 모델 만들기 (JobKind.VOICE_TRAIN)
  업로드한 내 노래 녹음 → (선택) 반주 제거 → 1~30초 조각 → Seed-VC 학습 → 목소리 모델 등록
  학습 없이 '참고 녹음만'으로 등록하는 빠른 방식도 있다 (Seed-VC 기본 노래 모델 + 참고 녹음).
- 보컬 바꾸기 (JobKind.VOICE_CONVERT)
  만든 곡 → 보컬/반주 분리 → 보컬을 목소리 모델로 변환 → 반주와 합치기 → '새 곡'으로 저장 (원곡은 그대로)

목소리 모델 폴더 (<결과 저장 폴더>/<모델 이름>/):
  reference.wav   참고 녹음 (변환 때 --target)
  ft_model.pth    학습 결과 (학습한 모델만) + <프리셋>.yml
  source/         업로드한 원본, clips/ 학습에 쓴 조각
기존 폴더·파일은 덮어쓰지 않는다. 모델 카테고리는 음악/가사 모델과 같은 '모델·카테고리' 기능을 쓴다.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.adapters.voice.local_adapter import last_error
from app.adapters.ports import SEEDVC_SVC_PRESET, TrainProgress, VoiceConvertSpec, VoiceEnginePort, VoiceTrainSpec
from app.core.errors import NotFoundError, ValidationError
from app.core.observer import EventBus, Topics
from app.domain.entities import ModelRecord, Song, new_id, utcnow
from app.domain.enums import Engine, JobKind, JobStatus
from app.repositories.content_repo import SongRepository
from app.services.category_service import (
    VOICE_CHECKPOINT,
    VOICE_REFERENCE,
    CategoryService,
    validate_adapter_dir,
    voice_model_files,
)
from app.services.job_service import JobContext, JobHandler
from app.services.music_training_service import _MAGIC, AUDIO_EXTS
from app.utils.fileio import atomic_write_bytes
from app.utils.path_guard import PathGuard, safe_filename

CLIP_SECONDS = 15.0


class VoiceTrainConfig(BaseModel):
    model_name: str = Field(min_length=1, max_length=80)
    export_dir: str = ""                 # 비우면 studio/workspace/library/voice/models
    mode: str = Field(default="train", pattern="^(train|reference)$")   # train = 학습, reference = 참고 녹음만
    separate: bool = False               # 업로드에 반주가 섞여 있으면 보컬만 뽑아서 쓴다
    max_steps: int = Field(default=500, ge=50, le=20_000)
    batch_size: int = Field(default=2, ge=1, le=16)


class VoiceConvertConfig(BaseModel):
    """보컬 변환 설정.

    원곡: source="song" (만든 곡) | "file" (밖에서 가져온 곡, 보컬 변환 탭에서 올림)
    목소리: voice="model" (내가 만든 목소리 모델) | "reference" (Seed-VC 기본 노래 모델 + 참고 목소리 녹음)
    """

    source: str = Field(default="song", pattern="^(song|file)$")
    song_id: str | None = None
    source_file: str | None = None        # inputs/ 안의 파일 이름
    voice: str = Field(default="model", pattern="^(model|reference|rvc)$")
    model_id: str | None = None
    reference_file: str | None = None     # inputs/ 안의 참고 목소리 파일 이름
    rvc_model: str | None = None
    rvc_index: str | None = None
    rvc_index_rate: float = Field(default=0.3, ge=0, le=1)
    reference_separate: bool = False      # 참고 녹음에 반주가 있으면 보컬만 뽑아서 쓴다
    semi_tone_shift: int = Field(default=0, ge=-24, le=24)
    diffusion_steps: int = Field(default=30, ge=10, le=100)
    title: str = Field(default="", max_length=80)
    output_dir: str = ""                  # 비우면 studio/workspace/library/songs
    audio_format: str = Field(default="mp3", pattern="^(mp3|wav|flac)$")


class VoiceService:
    def __init__(self, engine: VoiceEnginePort, categories: CategoryService, songs: SongRepository, guard: PathGuard,
                 library_dir: Path, bus: EventBus | None = None, max_upload_mb: int = 200,
                 resolve_output_dir: Any = None) -> None:
        self.engine = engine
        self.categories = categories
        self.songs = songs
        self.guard = guard
        self.library_dir = Path(library_dir)          # workspace/library/voice
        self.bus = bus or EventBus()
        self.max_upload_bytes = max_upload_mb * 1024 * 1024
        self._resolve_output_dir = resolve_output_dir     # MusicService.resolve_output_dir (곡 저장 폴더 규칙 공유)

    # --- 폴더 ---
    def _dir(self, *parts: str) -> Path:
        path = self.guard.child(self.library_dir, *parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def staging_dir(self) -> Path:
        return self._dir("_staging")

    def default_models_dir(self) -> Path:
        return self._dir("models")

    def work_dir(self, job_id: str) -> Path:
        return self.guard.child(self._dir("work"), job_id)

    def inputs_dir(self) -> Path:
        """보컬 변환 탭에서 올린 외부 곡·참고 목소리 보관 (원본은 사용자의 파일이라 복사본만 둔다)."""
        return self._dir("inputs")

    def output_dir(self, output_dir: str = "") -> Path:
        if self._resolve_output_dir is None:
            return self._dir("output")
        return self._resolve_output_dir(output_dir)

    # --- 업로드 ---
    def stage_upload(self, filename: str, data: bytes) -> Path:
        name, ext = self._check_audio(filename, data)
        staging = self.staging_dir()
        dest = staging / name
        n = 2
        while dest.exists():
            dest = staging / f"{Path(name).stem}_{n}{ext}"
            n += 1
        return atomic_write_bytes(dest, data)

    def _check_audio(self, filename: str, data: bytes) -> tuple[str, str]:
        name = safe_filename(filename)
        ext = Path(name).suffix.lower()
        if ext not in AUDIO_EXTS:
            raise ValidationError(f"지원하지 않는 형식입니다: {ext or '확장자 없음'} (mp3/wav/flac/ogg/m4a)")
        if len(data) > self.max_upload_bytes:
            raise ValidationError(f"파일이 너무 큽니다 (최대 {self.max_upload_bytes // 1024 // 1024}MB)")
        magic = _MAGIC.get(ext)
        if magic and not data.startswith(magic):
            raise ValidationError(f"{name}: 파일 내용이 {ext} 형식이 아닙니다.")
        return name, ext

    def store_input(self, filename: str, data: bytes) -> Path:
        """외부 곡 / 참고 목소리 올리기 → inputs/<짧은 id>_<이름> (같은 이름도 덮어쓰지 않음)."""
        name, _ = self._check_audio(filename, data)
        return atomic_write_bytes(self.inputs_dir() / f"{new_id()[:8]}_{name}", data)

    def rvc_dir(self) -> Path:
        return self._dir("rvc")

    def rvc_files(self, ext: str) -> list[str]:
        return sorted(p.name for p in self.rvc_dir().glob(f"*{ext}") if p.is_file())

    def store_rvc(self, filename: str, data: bytes) -> Path:
        name = safe_filename(filename)
        if Path(name).suffix not in {".pth", ".index"}:
            raise ValidationError("RVC 모델은 .pth, 인덱스는 .index 파일을 올리세요.")
        if not data or len(data) > self.max_upload_bytes:
            raise ValidationError(f"빈 파일이거나 업로드 한도({self.max_upload_bytes // 1024 // 1024}MB)를 넘었습니다.")
        return atomic_write_bytes(self.rvc_dir() / f"{new_id()[:8]}_{name}", data)

    def rvc_path(self, name: str, ext: str) -> Path:
        path = self.guard.child(self.rvc_dir(), safe_filename(name))
        if path.suffix != ext or not path.is_file() or path.stat().st_size == 0:
            raise ValidationError(f"RVC {ext} 파일을 찾을 수 없습니다. 다시 올리세요.")
        return path

    def input_path(self, name: str) -> Path:
        path = self.guard.child(self.inputs_dir(), safe_filename(name))
        if not path.is_file():
            raise ValidationError("올린 파일을 찾을 수 없습니다. 다시 올리세요.", detail=name)
        return path

    @staticmethod
    def input_label(name: str) -> str:
        """inputs 저장 이름에서 앞의 짧은 id 를 뗀 원래 이름."""
        return name.split("_", 1)[1] if len(name) > 9 and name[8] == "_" else name

    def staged_files(self) -> list[Path]:
        d = self.staging_dir()
        return sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith("."))

    def remove_staged(self, name: str) -> None:
        path = self.guard.child(self.staging_dir(), safe_filename(name))
        if path.is_file():
            path.unlink()

    def clear_staged(self) -> None:
        for p in self.staged_files():
            p.unlink()

    # --- 결과 폴더 ---
    def resolve_export_dir(self, model_name: str, export_dir: str = "") -> Path:
        """<결과 저장 폴더>/<모델 이름>. 이미 있는 폴더·보호 경로는 거부 (덮어쓰지 않음)."""
        name = safe_filename(model_name.strip())
        if not name:
            raise ValidationError("목소리 모델 이름을 입력하세요.")
        if export_dir.strip():
            base = Path(export_dir).expanduser().resolve()
            if not base.is_dir():
                raise ValidationError("결과 저장 폴더가 없습니다.", detail=str(base))
        else:
            base = self.default_models_dir()
        target = base / name
        for protected in self.guard.protected:
            if target == protected or protected in target.parents:
                raise ValidationError("보호된 기존 자산 폴더에는 저장할 수 없습니다.", detail=str(target))
        if target.exists():
            raise ValidationError(f"'{name}' 폴더가 이미 있습니다. 다른 모델 이름을 쓰세요.", detail=str(target))
        return target

    # --- 조회 ---
    def voice_models(self, category_id: str | None = None) -> list[ModelRecord]:
        if category_id:
            return self.categories.category_models(category_id, Engine.VOICE)
        return self.categories.list_models(engine=Engine.VOICE)

    def get_model(self, model_id: str) -> ModelRecord:
        model = self.categories.models.get(model_id)
        if model is None or model.engine != Engine.VOICE:
            raise NotFoundError("목소리 모델을 찾을 수 없습니다.", detail=model_id)
        return model

    def get_song(self, song_id: str) -> Song:
        song = self.songs.get(song_id)
        if song is None:
            raise NotFoundError("곡을 찾을 수 없습니다.", detail=song_id)
        if not song.audio_path or not Path(song.audio_path).is_file():
            raise ValidationError("곡의 음원 파일이 없습니다.", detail=song.audio_path)
        return song


# ---------------------------------------------------------------------------
# 잡 핸들러
# ---------------------------------------------------------------------------
class VoiceTrainHandler(JobHandler):
    kind = JobKind.VOICE_TRAIN

    def __init__(self, svc: VoiceService) -> None:
        self.svc = svc

    def describe(self, params: dict[str, Any]) -> str:
        return "목소리 모델 만들기 (Seed-VC)"

    def validate(self, params: dict[str, Any]) -> None:
        cfg = VoiceTrainConfig(**params.get("config", {}))
        self.svc.resolve_export_dir(cfg.model_name, cfg.export_dir)
        if not self.svc.staged_files():
            raise ValidationError("목소리 녹음 파일을 먼저 올리세요.")

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        cfg = VoiceTrainConfig(**ctx.params.get("config", {}))
        root = self.svc.resolve_export_dir(cfg.model_name, cfg.export_dir)   # 아직 없는 폴더만 (이 작업이 새로 만듦)
        files = self.svc.staged_files()
        if not files:
            raise ValidationError("목소리 녹음 파일이 없습니다.")
        work = self.svc.work_dir(ctx.job.id)
        try:
            return await self._run(ctx, cfg, root, files, work)
        except BaseException:
            # 실패·취소: 올린 녹음은 다시 '올린 파일'로 돌려놓고, 이 작업이 새로 만든 모델 폴더는 지운다
            src = root / "source"
            if src.is_dir():
                for f in src.iterdir():
                    dest = self.svc.staging_dir() / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)
            raise

    async def _run(self, ctx: JobContext, cfg: VoiceTrainConfig, root: Path, files: list[Path],
                   work: Path) -> dict[str, Any]:
        source_dir = root / "source"
        clips_dir = root / "clips"
        source_dir.mkdir(parents=True)
        for f in files:                                   # 올린 원본은 모델 폴더에 보관 (다시 학습할 수 있게)
            shutil.move(str(f), str(source_dir / f.name))
        ctx.log(f"목소리 모델 폴더: {root}")
        log = lambda line: ctx.log(line, "debug")          # noqa: E731

        # ① (선택) 반주 제거 → ② 1~30초 조각
        clips: list[Path] = []
        sources = sorted(source_dir.iterdir())
        for i, f in enumerate(sources):
            ctx.check_cancelled()
            src = f
            if cfg.separate:
                ctx.log(f"반주 제거: {f.name}")
                src, _ = await self.svc.engine.separate(f, work / f"sep_{i:03d}", log)
            clips += await self.svc.engine.split_clips(src, clips_dir / f"{i:03d}", CLIP_SECONDS)
            ctx.progress(0.1 * (i + 1) / len(sources))
        if not clips:
            raise ValidationError("1초 이상인 녹음 조각이 없습니다. 더 긴 녹음을 올리세요.")
        total_s = len(clips) * CLIP_SECONDS
        ctx.log(f"학습용 조각 {len(clips)}개 (약 {total_s / 60:.0f}분 이하)")
        reference = max(clips, key=lambda p: p.stat().st_size)    # 가장 긴 조각을 참고 녹음으로
        shutil.copy2(reference, root / VOICE_REFERENCE)

        train_info: dict[str, Any] = {"mode": cfg.mode, "clips": len(clips), "separate": cfg.separate}
        if cfg.mode == "train":
            spec = VoiceTrainSpec(dataset_dir=clips_dir, log_dir=root.parent, run_name=root.name,
                                  preset=SEEDVC_SVC_PRESET, max_steps=cfg.max_steps, save_every=cfg.max_steps,
                                  batch_size=cfg.batch_size)
            ctx.log(f"Seed-VC 학습 시작: {cfg.max_steps} 스텝, 배치 {cfg.batch_size}")
            losses: list[dict[str, Any]] = []
            tail_lines: list[str] = []

            def on_progress(tp: TrainProgress) -> None:
                ctx.check_cancelled()
                if tp.line:
                    tail_lines.append(tp.line)
                    del tail_lines[:-80]
                if tp.step is not None:
                    losses.append({"step": tp.step, "loss": tp.train_loss})
                    ctx.progress(0.1 + 0.85 * min(tp.step / cfg.max_steps, 1.0))
                if tp.line:
                    ctx.log(tp.line, "info" if tp.step is not None or "saved" in tp.line.lower() else "debug")

            handle = await self.svc.engine.start_training(spec, on_progress)
            try:
                code = await handle.wait()
            except BaseException:
                await handle.cancel()
                raise
            if code != 0:
                reason = last_error(tail_lines)
                raise ValidationError(f"Seed-VC 학습이 오류로 끝났습니다 (종료 코드 {code})"
                                      + (f": {reason}" if reason else ". 위 로그를 확인하세요."), detail=str(root))
            ctx.set_status(JobStatus.EXPORTING)
            for leftover in root.glob("DiT_epoch_*.pth"):     # 이 작업이 만든 중간 체크포인트 (최종본만 남김)
                leftover.unlink()
                ctx.log(f"중간 체크포인트 정리: {leftover.name}", "debug")
            train_info |= {"max_steps": cfg.max_steps, "batch_size": cfg.batch_size, "preset": SEEDVC_SVC_PRESET,
                           "losses": losses[-200:]}
            ctx.set_status(JobStatus.VALIDATING_ADAPTER)
            if not (root / VOICE_CHECKPOINT).is_file():
                raise ValidationError("학습은 끝났지만 ft_model.pth 가 만들어지지 않았습니다.", detail=str(root))
        else:
            ctx.set_status(JobStatus.EXPORTING)
            ctx.set_status(JobStatus.VALIDATING_ADAPTER)

        ok, msg = validate_adapter_dir(Engine.VOICE, root)
        if not ok:
            raise ValidationError(f"목소리 모델 검증 실패: {msg}", detail=str(root))
        base = "seed-vc:whisper_base_f0_44k" + ("" if cfg.mode == "train" else " (학습 없음)")
        model = await asyncio.to_thread(
            self.svc.categories.register_model, None, Engine.VOICE, base, root,
            train_config={**train_info, "export_root": str(root)},
            notes=msg, display_name=cfg.model_name.strip(),
        )
        shutil.rmtree(work, ignore_errors=True)
        ctx.log(f"목소리 모델 등록: {model.label} — [보컬 변환]에서 쓸 수 있습니다.")
        return {"model_id": model.id, "path": str(root), "clips": len(clips)}


class VoiceConvertHandler(JobHandler):
    """보컬 변환: 원곡(만든 곡 또는 외부 파일) → 분리 → 변환 → 반주와 합치기 → 제목으로 지정 폴더에 저장."""

    kind = JobKind.VOICE_CONVERT

    def __init__(self, svc: VoiceService) -> None:
        self.svc = svc

    def describe(self, params: dict[str, Any]) -> str:
        return "보컬 변환 (분리 → 변환 → 합치기)"

    def _source(self, cfg: VoiceConvertConfig) -> tuple[Path, Song | None]:
        if cfg.source == "song":
            if not cfg.song_id:
                raise ValidationError("바꿀 곡을 고르세요.")
            song = self.svc.get_song(cfg.song_id)
            return Path(song.audio_path), song
        if not cfg.source_file:
            raise ValidationError("바꿀 곡 파일을 올리세요.")
        return self.svc.input_path(cfg.source_file), None

    def _voice(self, cfg: VoiceConvertConfig) -> tuple[ModelRecord | None, Path | None]:
        if cfg.voice == "rvc":
            if not cfg.rvc_model:
                raise ValidationError("RVC 모델(.pth)을 넣으세요.")
            self.svc.rvc_path(cfg.rvc_model, ".pth")
            if cfg.rvc_index:
                self.svc.rvc_path(cfg.rvc_index, ".index")
            validate = getattr(self.svc.engine, "validate_rvc", None)
            if validate:
                validate()
            return None, None
        if cfg.voice == "model":
            if not cfg.model_id:
                raise ValidationError("목소리 모델을 고르세요.")
            model = self.svc.get_model(cfg.model_id)
            ok, msg = validate_adapter_dir(Engine.VOICE, Path(model.adapter_path))
            if not ok:
                raise ValidationError(f"목소리 모델 파일 문제: {msg}", detail=model.adapter_path)
            return model, None
        if not cfg.reference_file:
            raise ValidationError("기본 모델을 쓰려면 바꿀 목소리의 참고 녹음을 올리세요.")
        return None, self.svc.input_path(cfg.reference_file)

    def validate(self, params: dict[str, Any]) -> None:
        cfg = VoiceConvertConfig(**params)
        self._source(cfg)
        self._voice(cfg)
        self.svc.output_dir(cfg.output_dir)

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        created: list[Path] = []
        try:
            return await self._run(ctx, created)
        except BaseException:
            for p in created:            # 이 작업이 만들다 만 결과만 지운다 (원곡은 건드리지 않음)
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()
            raise
        finally:
            shutil.rmtree(self.svc.work_dir(ctx.job.id), ignore_errors=True)

    async def _run(self, ctx: JobContext, created: list[Path]) -> dict[str, Any]:
        cfg = VoiceConvertConfig(**ctx.params)
        src, song = self._source(cfg)
        model, uploaded_ref = self._voice(cfg)
        out_dir = self.svc.output_dir(cfg.output_dir)
        work = self.svc.work_dir(ctx.job.id)
        log = lambda line: ctx.log(line, "debug")          # noqa: E731
        src_label = (song.title if song else self.svc.input_label(src.name)) or "무제"

        if cfg.voice == "rvc":
            reference, config = None, None
            checkpoint = self.svc.rvc_path(cfg.rvc_model, ".pth")
            voice_name = self.svc.input_label(cfg.rvc_model)
        elif model is not None:
            reference, checkpoint, config = voice_model_files(Path(model.adapter_path))
            voice_name = model.display_name or model.label
        else:
            reference, checkpoint, config = uploaded_ref, None, None   # type: ignore[assignment]
            voice_name = f"기본 모델 + {self.svc.input_label(uploaded_ref.name)}"   # type: ignore[union-attr]
            if cfg.reference_separate:
                ctx.log("참고 녹음에서 반주 제거")
                reference, _ = await self.svc.engine.separate(uploaded_ref, work / "reference", log)  # type: ignore[arg-type]

        ctx.log(f"① 보컬/반주 분리: {src_label}")
        vocals, inst = await self.svc.engine.separate(src, work / "separated", log)
        ctx.progress(0.3)
        ctx.check_cancelled()

        detail = "RVC / Applio" if cfg.voice == "rvc" else f"스텝 {cfg.diffusion_steps}"
        ctx.log(f"② 목소리 변환: {voice_name} (키 {cfg.semi_tone_shift:+d}, {detail})")
        converted = await self.svc.engine.convert(VoiceConvertSpec(
            source=vocals, reference=reference, output_dir=work / "converted", checkpoint=checkpoint,
            config=config if checkpoint else None, diffusion_steps=cfg.diffusion_steps,
            semi_tone_shift=cfg.semi_tone_shift, backend="rvc" if cfg.voice == "rvc" else "seed-vc",
            index=self.svc.rvc_path(cfg.rvc_index, ".index") if cfg.voice == "rvc" and cfg.rvc_index else None,
            index_rate=cfg.rvc_index_rate), log)
        ctx.progress(0.8)
        ctx.check_cancelled()

        ctx.log("③ 반주와 합치기 (FFmpeg)")
        default_title = f"{Path(src_label).stem if song is None else src_label} · " \
                        f"{voice_name if cfg.voice == 'rvc' else model.display_name if model and model.display_name else '목소리 변환'}"
        title = cfg.title.strip() or default_title
        dest = self._unique(out_dir, safe_filename(title).replace(".", "_") or "voice", f".{cfg.audio_format}")
        stems = dest.parent / f"{dest.stem}_stems"         # 바뀐 보컬·반주도 따로 남김 (다시 섞을 때)
        created.append(dest)
        await self.svc.engine.mix(converted, inst, dest)
        created.append(stems)
        stems.mkdir()
        shutil.move(str(converted), str(stems / "vocals_converted.wav"))
        shutil.move(str(inst), str(stems / "instrumental.wav"))

        new = Song(category_id=song.category_id if song else None,
                   lyrics_draft_id=song.lyrics_draft_id if song else None,
                   music_model_id=song.music_model_id if song else None,
                   qwen_model_id=song.qwen_model_id if song else None,
                   title=title, prompt=song.prompt if song else "", audio_path=str(dest), status="succeeded")
        new.generation = {
            "voice": {"model_id": model.id if model else None, "model_name": voice_name,
                      "trained": checkpoint is not None, "base_model": cfg.voice == "reference",
                      "backend": "rvc" if cfg.voice == "rvc" else "seed-vc",
                      "rvc_model": cfg.rvc_model if cfg.voice == "rvc" else None,
                      "rvc_index": cfg.rvc_index if cfg.voice == "rvc" else None,
                      "rvc_index_rate": cfg.rvc_index_rate if cfg.voice == "rvc" else None,
                      "reference": str(reference) if reference else None, "semi_tone_shift": cfg.semi_tone_shift,
                      "diffusion_steps": cfg.diffusion_steps, "stems_dir": str(stems)},
            "source": {"type": cfg.source, "song_id": song.id if song else None, "path": str(src)},
            "source_song_id": song.id if song else None,
            "lyrics": song.generation.get("lyrics", "") if song else "",
            "seed": song.generation.get("seed") if song else None, "engine": self.svc.engine.name,
            "is_mock": bool(self.svc.engine.is_mock or (song and song.generation.get("is_mock"))),
            "finished_at": utcnow().isoformat(),
        }
        self.svc.songs.add(new)
        self.svc.bus.publish(Topics.SONG_CREATED, song=new)
        ctx.log(f"저장: {dest}  (원곡은 그대로)")
        return {"song_ids": [new.id], "path": str(dest), "stems_dir": str(stems)}

    @staticmethod
    def _unique(folder: Path, stem: str, ext: str) -> Path:
        """제목으로 파일 이름을 만들되, 이미 있으면 _2, _3 … (덮어쓰지 않음)."""
        candidate = folder / f"{stem}{ext}"
        n = 2
        while candidate.exists() or (folder / f"{candidate.stem}_stems").exists():
            candidate = folder / f"{stem}_{n}{ext}"
            n += 1
        return candidate


__all__ = ["VoiceService", "VoiceTrainHandler", "VoiceConvertHandler", "VoiceTrainConfig", "VoiceConvertConfig",
           "new_id"]
