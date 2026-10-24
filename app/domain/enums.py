from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - 표시용
        return self.value


class Engine(StrEnum):
    QWEN = "qwen"
    ACE = "ace"


class DatasetKind(StrEnum):
    LYRICS = "lyrics"
    MUSIC = "music"


class CategoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class LyricsSource(StrEnum):
    """LYR-001: AI 생성물/사용자 원문/허락받은 제3자 자료 구분."""

    AI_GENERATED = "ai_generated"
    USER_ORIGINAL = "user_original"
    LICENSED_THIRD_PARTY = "licensed_third_party"


class JobKind(StrEnum):
    LYRICS_GENERATE = "lyrics_generate"
    PROMPT_GENERATE = "prompt_generate"
    MUSIC_GENERATE = "music_generate"
    LYRICS_TRAIN = "lyrics_train"
    MUSIC_SCAN = "music_scan"
    MUSIC_LABEL = "music_label"
    MUSIC_PREPROCESS = "music_preprocess"
    MUSIC_TRAIN = "music_train"
    FULL_SONG = "full_song"

    @property
    def is_gpu_heavy(self) -> bool:
        return self in _GPU_HEAVY


_GPU_HEAVY = {
    JobKind.LYRICS_GENERATE,
    JobKind.PROMPT_GENERATE,
    JobKind.MUSIC_GENERATE,
    JobKind.LYRICS_TRAIN,
    JobKind.MUSIC_LABEL,
    JobKind.MUSIC_PREPROCESS,
    JobKind.MUSIC_TRAIN,
    JobKind.FULL_SONG,
}


class JobStatus(StrEnum):
    CREATED = "created"
    VALIDATING = "validating"
    QUEUED = "queued"
    RUNNING = "running"
    EXPORTING = "exporting"
    VALIDATING_ADAPTER = "validating_adapter"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def label_ko(self) -> str:
        return _STATUS_KO[self]


_STATUS_KO = {
    JobStatus.CREATED: "생성됨",
    JobStatus.VALIDATING: "검증 중",
    JobStatus.QUEUED: "대기 중",
    JobStatus.RUNNING: "실행 중",
    JobStatus.EXPORTING: "내보내는 중",
    JobStatus.VALIDATING_ADAPTER: "어댑터 검증 중",
    JobStatus.SUCCEEDED: "성공",
    JobStatus.FAILED: "실패",
    JobStatus.CANCELLED: "취소됨",
}


class ValidationStatus(StrEnum):
    UNVERIFIED = "unverified"        # 파일만 존재
    STRUCTURE_OK = "structure_ok"    # adapter_config/safetensors 구조 확인
    LOADED = "loaded"                # 엔진 로드 성공
    APPLIED_VERIFIED = "applied_verified"  # A/B 생성으로 실제 적용 확인
    INVALID = "invalid"
