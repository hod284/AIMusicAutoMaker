"""음악 생성 서비스 (MUS-006 / MUS-007).

흐름 (임계구역 안에서 수행 — ACE LoRA 상태는 서버 전역):
  활성 LoRA 조회 → 현재 상태 확인 → 필요 시 Unload → Load → 상태 확인
  → /release_task → /query_result 폴링(백오프) → /v1/audio 다운로드 → Song 저장

'로드 성공'과 '실제 생성 적용'은 별개다. 로드 결과는 ``lora.load_status`` 에,
실제 적용 검증은 A/B 비교(같은 seed/가사/프롬프트)로 사용자가 판단하도록 기록만 남긴다.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.adapters.ports import LoraStatus, MusicEnginePort, MusicGenerationRequest, TaskResult, TaskState
from app.core.errors import CapabilityUnsupportedError, EngineResponseError, ValidationError
from app.core.observer import EventBus, Topics
from app.domain.entities import ModelRecord, Song, utcnow
from app.domain.enums import Engine, JobKind
from app.repositories.content_repo import SongRepository
from app.services.category_service import CategoryService, validate_adapter_dir
from app.services.job_service import JobContext, JobHandler
from app.services.lyrics_service import LyricsService
from app.services.resource_lock import ResourceLock
from app.utils.path_guard import safe_filename

log = logging.getLogger(__name__)
LogFn = Callable[..., None]


class MusicGenerationSpec(BaseModel):
    category_id: str | None = None     # 선택 (곡 만들기에서 모델을 직접 고르면 없음)
    title: str = ""
    draft_id: str | None = None
    lyrics: str | None = None          # draft 대신 직접 가사
    prompt: str = ""                   # 음악 프롬프트 (선택). 비우면 가사 + 음악 모델만으로 생성
    vocal_language: str = "ko"
    bpm: int | None = None
    use_lora: bool = True              # False = LoRA OFF (기본 ACE 모델)
    model_id: str | None = None        # 음악 모델 (ACE LoRA). None 이면 카테고리 활성 LoRA 또는 기본
    output_dir: str = ""               # 결과 저장 폴더 (비우면 studio/workspace/library/songs)
    audio_duration: float | None = None
    seed: int | None = None
    audio_format: str = "mp3"
    batch_size: int = Field(default=1, ge=1, le=8)
    inference_steps: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    prompt_meta: dict[str, Any] = Field(default_factory=dict)  # 디렉터 출처/보정 기록


class MusicService:
    def __init__(self, engine: MusicEnginePort, songs: SongRepository, categories: CategoryService,
                 lyrics: LyricsService, lock: ResourceLock, *, poll_interval_s: float = 2.0,
                 poll_timeout_s: float = 1800.0, bus: EventBus | None = None) -> None:
        self.engine = engine
        self.songs = songs
        self.categories = categories
        self.lyrics = lyrics
        self.lock = lock
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        self.bus = bus or EventBus()
        self._believed_adapter: str | None | object = _UNKNOWN  # 앱이 알고 있는 서버 LoRA 상태

    # ------------------------------------------------------------------
    def resolve_lora(self, spec: MusicGenerationSpec) -> ModelRecord | None:
        if not spec.use_lora:
            return None
        if spec.model_id:
            model = self.categories.models.get(spec.model_id)
            if model is None or model.engine != Engine.ACE:
                raise ValidationError("ACE 음악 모델이 아닙니다.", detail=spec.model_id)
            return model
        return self.categories.active_model(spec.category_id, Engine.ACE) if spec.category_id else None

    async def apply_lora(self, model: ModelRecord | None, log_fn: LogFn) -> dict[str, Any]:
        """서버 LoRA 상태를 목표 상태로 맞춘다. 반환값은 생성 기록에 저장된다."""
        caps = await self.engine.capabilities()
        target = model.adapter_path if model else None
        info: dict[str, Any] = {"model_id": model.id if model else None, "adapter_path": target, "requested": bool(model)}

        if target is None:
            if self._believed_adapter is None:
                info["load_status"] = "already_off"
                return info
            if caps.is_usable("lora_unload"):
                status = await self.engine.unload_lora()
                self._believed_adapter = None
                info["load_status"] = "unloaded"
                info["status"] = status.model_dump(mode="json")
                log_fn("LoRA 언로드 → 기본 모델로 생성")
            else:
                info["load_status"] = "off_unverified"
                log_fn("LoRA 언로드 라우트 미지원 — 서버에 이전 LoRA 가 남아 있을 수 있습니다.", "warning")
            return info

        ok, message = validate_adapter_dir(Engine.ACE, Path(target))
        if not ok:
            raise ValidationError(message, detail=target)
        if not caps.is_usable("lora_load"):
            raise CapabilityUnsupportedError(
                "설치된 ACE API 에서 LoRA 로드 라우트를 확인하지 못했습니다. 설정 > ACE 기능 표를 확인하거나 "
                "LoRA OFF 로 생성하세요.",
                detail=caps.capabilities.get("lora_load").reason if caps.capabilities.get("lora_load") else None,
            )
        if self._believed_adapter == target:
            current = await self.engine.lora_status()
            if current.adapter_path in (None, target) and current.loaded is not False:
                info["load_status"] = "already_loaded"
                info["status"] = current.model_dump(mode="json")
                return info
        if self._believed_adapter not in (None, _UNKNOWN) and caps.is_usable("lora_unload"):
            await self.engine.unload_lora()
            log_fn("기존 LoRA 언로드")
        status: LoraStatus = await self.engine.load_lora(Path(target))
        if status.loaded is False:
            raise EngineResponseError("LoRA 로드 실패", detail=status.raw)
        self._believed_adapter = target
        info["load_status"] = "loaded" if status.loaded else "load_requested_unverified"
        info["status"] = status.model_dump(mode="json")
        log_fn(f"LoRA 로드: {target} (확인 방식: {status.verified_by or '미확인'})")
        return info

    async def poll(self, task_id: str, log_fn: LogFn, on_tick: Callable[[], None] | None = None) -> TaskResult:
        deadline = time.monotonic() + self.poll_timeout_s
        interval = self.poll_interval_s
        while True:
            results = await self.engine.query_results([task_id])
            result = next((r for r in results if r.task_id == task_id), None)
            if result is None:
                raise EngineResponseError("query_result 에 요청한 task_id 가 없습니다.", detail=task_id)
            if result.state == TaskState.SUCCEEDED:
                return result
            if result.state == TaskState.FAILED:
                raise EngineResponseError("ACE 생성 실패", detail=result.error or result.raw)
            if time.monotonic() > deadline:
                raise EngineResponseError("ACE 생성 시간 초과", detail=task_id)
            if on_tick:
                on_tick()
            await asyncio.sleep(interval)
            interval = min(interval * 1.5, 10.0)

    async def generate(self, spec: MusicGenerationSpec, log_fn: LogFn | None = None,
                       on_tick: Callable[[], None] | None = None) -> list[Song]:
        log_fn = log_fn or (lambda *a, **k: None)
        category = self.categories.get(spec.category_id) if spec.category_id else None
        draft = self.lyrics.get(spec.draft_id) if spec.draft_id else None
        lyrics = spec.lyrics if spec.lyrics is not None else (draft.lyrics if draft else "")
        if not lyrics.strip():
            raise ValidationError("가사가 비어 있습니다. (연주곡은 '[Instrumental]' 을 입력하세요)")
        model = self.resolve_lora(spec)
        if spec.use_lora and model is None:
            log_fn("선택한 음악 LoRA 가 없어 ACE 기본 모델로 생성합니다.", "warning")
        out_dir = self.resolve_output_dir(spec.output_dir)

        request = MusicGenerationRequest(
            prompt=spec.prompt, lyrics=lyrics, vocal_language=spec.vocal_language, audio_format=spec.audio_format,
            audio_duration=spec.audio_duration, bpm=spec.bpm, seed=spec.seed, batch_size=spec.batch_size,
            inference_steps=spec.inference_steps, extra=spec.extra,
        )
        started = utcnow()
        async with self.lock.music_engine():
            lora_info = await self.apply_lora(model, log_fn)
            task_id = await self.engine.submit_generation(request)
            log_fn(f"ACE 작업 등록: task_id={task_id}")
            result = await self.poll(task_id, log_fn, on_tick)

        songs: list[Song] = []
        for idx, audio in enumerate(result.audios):
            song = Song(category_id=category.id if category else None, lyrics_draft_id=draft.id if draft else None,
                        music_model_id=model.id if model else None, qwen_model_id=draft.model_id if draft else None,
                        title=(spec.title or (draft.title if draft else "") or "무제") + (f" #{idx + 1}" if len(result.audios) > 1 else ""),
                        prompt=spec.prompt, status="downloading")
            dest = out_dir / f"{safe_filename(song.title).replace('.', '_')}_{song.id[:8]}"
            path = await self.engine.download_audio(audio, dest)
            song.audio_path = str(path)
            song.status = "succeeded"
            song.generation = {
                "task_id": task_id, "request": request.model_dump(), "lyrics": lyrics,
                "lora": lora_info, "seed": audio.seed, "metas": audio.metas, "dit_model": audio.dit_model,
                "lm_model": audio.lm_model, "engine": self.engine.name, "is_mock": self.engine.is_mock,
                "prompt_meta": spec.prompt_meta, "started_at": started.isoformat(),
                "finished_at": utcnow().isoformat(),
            }
            self.songs.add(song)
            songs.append(song)
            self.bus.publish(Topics.SONG_CREATED, song=song)
            log_fn(f"음원 저장: {path.name}")
        if not songs:
            raise EngineResponseError("생성 결과에 음원이 없습니다.")
        return songs

    def resolve_output_dir(self, output_dir: str) -> Path:
        """곡 저장 폴더. 지정하지 않으면 studio/workspace/library/songs."""
        if not output_dir.strip():
            root = self.categories.workspace.categories_dir.parent
            path = self.categories.workspace.guard.child(root, "library", "songs")
            path.mkdir(parents=True, exist_ok=True)
            return path
        path = Path(output_dir).expanduser().resolve()
        if not path.is_dir():
            raise ValidationError("결과 저장 폴더가 없습니다.", detail=str(path))
        for protected in self.categories.workspace.guard.protected:
            if path == protected or protected in path.parents:
                raise ValidationError("기존 ACE 자산 폴더에는 저장할 수 없습니다.", detail=str(path))
        return path

    def invalidate_lora_state(self) -> None:
        """학습 등으로 서버 LoRA 상태가 바뀌었을 수 있을 때 호출 → 다음 생성 때 다시 맞춘다."""
        self._believed_adapter = _UNKNOWN

    def list_songs(self, category_id: str | None = None, query: str = "", *, title_only: bool = False,
                   limit: int = 1000) -> list[Song]:
        return self.songs.list(category_id, query, limit, title_only=title_only)


_UNKNOWN = object()


class MusicGenerateHandler(JobHandler):
    kind = JobKind.MUSIC_GENERATE

    def __init__(self, service: MusicService) -> None:
        self.service = service

    def validate(self, params: dict[str, Any]) -> None:
        """params = {"spec": MusicGenerationSpec dict, "ab_compare": bool}"""
        if "spec" not in params:
            raise ValidationError("spec 이 필요합니다.")
        spec = MusicGenerationSpec(**params["spec"])
        if params.get("ab_compare") and spec.seed is None:
            # A/B 비교는 같은 seed 가 필수 → 검증 단계에서 고정하고 기록
            params["spec"]["seed"] = random.randint(0, 2**31 - 1)

    def describe(self, params: dict[str, Any]) -> str:
        return "ACE-Step 음악 생성"

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        spec = MusicGenerationSpec(**ctx.params["spec"])

        def tick() -> None:
            ctx.check_cancelled()
            ctx.heartbeat()

        if ctx.params.get("ab_compare"):
            ctx.log(f"A/B 비교 생성 (seed={spec.seed}) — A: LoRA OFF")
            a = await self.service.generate(spec.model_copy(update={"use_lora": False, "title": f"{spec.title} [A: 기본]"}),
                                            ctx.log, tick)
            ctx.progress(0.5)
            ctx.log("B: LoRA ON")
            b = await self.service.generate(spec.model_copy(update={"use_lora": True, "title": f"{spec.title} [B: LoRA]"}),
                                            ctx.log, tick)
            return {"song_ids": [s.id for s in a + b], "ab": {"a": [s.id for s in a], "b": [s.id for s in b]},
                    "seed": spec.seed}
        songs = await self.service.generate(spec, ctx.log, tick)
        return {"song_ids": [s.id for s in songs]}
