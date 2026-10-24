"""엔진 포트 (어댑터 패턴의 Target 인터페이스).

서비스 계층은 오직 이 인터페이스에만 의존한다.
- ``MusicEnginePort``: ACE-Step REST 어댑터 / Mock 어댑터 / (향후) ACE CLI fallback 어댑터
- ``LyricsEnginePort``: MLX-LM 서브프로세스 어댑터 / Mock 어댑터 / (향후) 상주 MLX 서버 어댑터

새 엔진을 붙일 때는 이 포트를 구현한 어댑터를 만들고 ``AdapterFactory`` 에 등록만 하면 된다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 공통 DTO
# ---------------------------------------------------------------------------
class EngineHealth(BaseModel):
    engine: str
    ok: bool
    message: str = ""
    version: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    is_mock: bool = False


class CapabilityState(str, Enum):
    AVAILABLE = "available"      # 바인딩 존재 + OpenAPI 에서 확인됨
    MISSING = "missing"          # 바인딩은 있으나 설치본 OpenAPI 에 없음
    UNBOUND = "unbound"          # 아직 라우트 매핑 안 됨 (추측 금지 → 비활성)
    UNKNOWN = "unknown"          # 서버 미연결 등으로 확인 불가
    MOCK = "mock"


class Capability(BaseModel):
    name: str
    label_ko: str
    state: CapabilityState
    method: str | None = None
    path: str | None = None
    reason: str = ""
    suggestions: list[str] = Field(default_factory=list)
    request_fields: list[str] = Field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.state in (CapabilityState.AVAILABLE, CapabilityState.MOCK)


class CapabilityReport(BaseModel):
    engine: str
    capabilities: dict[str, Capability] = Field(default_factory=dict)
    discovered_paths: list[str] = Field(default_factory=list)
    source: str = ""  # "openapi" | "mock" | "unavailable"

    def is_usable(self, name: str) -> bool:
        cap = self.capabilities.get(name)
        return bool(cap and cap.usable)


# ---------------------------------------------------------------------------
# 음악 엔진
# ---------------------------------------------------------------------------
class MusicGenerationRequest(BaseModel):
    prompt: str
    lyrics: str
    vocal_language: str = "ko"
    audio_format: str = "mp3"
    audio_duration: float | None = Field(default=None, ge=10, le=600)
    bpm: int | None = Field(default=None, ge=30, le=300)
    key_scale: str | None = None
    time_signature: str | None = None
    seed: int | None = None
    batch_size: int = Field(default=1, ge=1, le=8)
    inference_steps: int | None = None
    thinking: bool = False
    model: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)  # 고급 옵션(문서화된 필드만 통과)


class TaskState(int, Enum):
    RUNNING = 0
    SUCCEEDED = 1
    FAILED = 2


class GeneratedAudio(BaseModel):
    file: str
    seed: str = ""
    metas: dict[str, Any] = Field(default_factory=dict)
    dit_model: str = ""
    lm_model: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskResult(BaseModel):
    task_id: str
    state: TaskState
    audios: list[GeneratedAudio] = Field(default_factory=list)
    error: str = ""
    raw: Any = None


class LoraStatus(BaseModel):
    loaded: bool | None = None       # None = 확인 불가
    enabled: bool | None = None
    adapter_path: str | None = None
    verified_by: str = ""            # "api" | "assumed" | "mock"
    raw: Any = None


class MusicEnginePort(ABC):
    name: str = "music"
    is_mock: bool = False

    @abstractmethod
    async def health(self) -> EngineHealth: ...

    @abstractmethod
    async def capabilities(self, refresh: bool = False) -> CapabilityReport: ...

    @abstractmethod
    async def submit_generation(self, req: MusicGenerationRequest) -> str:
        """생성 작업을 등록하고 task_id 를 반환한다."""

    @abstractmethod
    async def query_results(self, task_ids: list[str]) -> list[TaskResult]: ...

    @abstractmethod
    async def download_audio(self, audio: GeneratedAudio, dest: Path) -> Path: ...

    @abstractmethod
    async def load_lora(self, adapter_path: Path) -> LoraStatus: ...

    @abstractmethod
    async def unload_lora(self) -> LoraStatus: ...

    @abstractmethod
    async def lora_status(self) -> LoraStatus: ...

    async def invoke(self, capability: str, payload: dict[str, Any] | None = None, *,
                     path_params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        """학습/데이터셋 계열 기능을 Capability 이름으로 호출하는 확장 지점."""
        from app.core.errors import CapabilityUnsupportedError

        raise CapabilityUnsupportedError(detail=capability)

    async def aclose(self) -> None:  # pragma: no cover - 기본 no-op
        return None


# ---------------------------------------------------------------------------
# 가사(LLM) 엔진
# ---------------------------------------------------------------------------
class LLMRequest(BaseModel):
    system: str = ""
    user: str
    adapter_path: str | None = None    # None = Qwen 기본 모델
    model: str | None = None
    max_tokens: int = Field(default=1024, ge=16, le=8192)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    seed: int | None = None
    enable_thinking: bool = False


class LLMResult(BaseModel):
    text: str                  # thinking 이 제거된 본문
    thinking: str = ""
    model: str = ""
    adapter_path: str | None = None
    elapsed_s: float = 0.0
    is_mock: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


class LoraTrainSpec(BaseModel):
    """가사 LoRA 학습 사양. CLI 옵션 매핑은 어댑터가 --help 로 검증한다."""

    model: str
    data_dir: Path
    adapter_path: Path
    iters: int = Field(default=100, ge=1)
    batch_size: int = Field(default=1, ge=1)
    learning_rate: float | None = None
    max_seq_length: int | None = None
    num_layers: int | None = None
    save_every: int | None = None
    steps_per_eval: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class TrainProgress(BaseModel):
    step: int | None = None
    total: int | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    line: str = ""


ProgressCallback = Callable[[TrainProgress], Awaitable[None] | None]


class TrainHandle(ABC):
    @abstractmethod
    async def wait(self) -> int:
        """종료 코드 반환."""

    @abstractmethod
    async def cancel(self, grace_s: float = 10.0) -> None: ...


class LyricsEnginePort(ABC):
    name: str = "lyrics"
    is_mock: bool = False

    @abstractmethod
    async def health(self) -> EngineHealth: ...

    @abstractmethod
    async def generate(self, req: LLMRequest) -> LLMResult: ...

    async def training_options(self) -> set[str]:
        """설치된 학습 CLI 가 지원하는 옵션 집합 (``--help`` 파싱)."""
        return set()

    async def start_training(self, spec: LoraTrainSpec, on_progress: ProgressCallback) -> TrainHandle:
        from app.core.errors import CapabilityUnsupportedError

        raise CapabilityUnsupportedError(detail="lyrics_train")
