"""도메인 엔티티 (MVVM 의 Model).

Pydantic 모델로 검증하며, 리포지토리가 SQLite 행 ↔ 엔티티 변환을 담당한다.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.domain.enums import (
    CategoryStatus,
    DatasetKind,
    Engine,
    JobKind,
    JobStatus,
    LyricsSource,
    ValidationStatus,
)


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StyleConstraints(BaseModel):
    """CAT-003 카테고리 프롬프트 가드레일. 학습 데이터가 아니라 '설정'이다."""

    allowed_genres: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    vocal: str | None = None                      # 예: "male", "female"
    preferred_bpm_min: int | None = Field(default=None, ge=30, le=300)
    preferred_bpm_max: int | None = Field(default=None, ge=30, le=300)
    instruments: list[str] = Field(default_factory=list)
    mood_tags: list[str] = Field(default_factory=list)
    era: str | None = None
    base_prompt_template: str = ""                # Qwen 실패 시 fallback 템플릿
    lyrics_instructions: str = ""                 # 작사 지시

    @model_validator(mode="after")
    def _bpm_range(self) -> "StyleConstraints":
        if (
            self.preferred_bpm_min is not None
            and self.preferred_bpm_max is not None
            and self.preferred_bpm_min > self.preferred_bpm_max
        ):
            raise ValueError("BPM 최소값이 최대값보다 큽니다.")
        return self


class Category(BaseModel):
    id: str = Field(default_factory=new_id)
    slug: str
    name: str
    description: str = ""
    language: str = "ko"
    tags: list[str] = Field(default_factory=list)
    status: CategoryStatus = CategoryStatus.ACTIVE
    style: StyleConstraints = Field(default_factory=StyleConstraints)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    @field_validator("name")
    @classmethod
    def _name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("카테고리 이름은 비울 수 없습니다.")
        return v.strip()


class License(BaseModel):
    id: str = Field(default_factory=new_id)
    source_url: str = ""
    creator: str = ""
    license_name: str = ""
    scope: str = ""            # 학습/생성 허용 범위
    evidence_path: str = ""
    verified_at: datetime | None = None

    @property
    def is_verified(self) -> bool:
        return self.verified_at is not None and bool(self.license_name)


class Dataset(BaseModel):
    id: str = Field(default_factory=new_id)
    category_id: str | None          # None = 카테고리 없는 라이브러리 데이터셋 (음악 학습)
    kind: DatasetKind
    version: int
    status: str = "draft"
    source_path: str = ""
    json_path: str = ""
    tensor_path: str = ""
    rights_status: str = "unverified"
    created_at: datetime = Field(default_factory=utcnow)


class DatasetItem(BaseModel):
    id: str = Field(default_factory=new_id)
    dataset_id: str
    source_path: str = ""
    sha256: str = ""
    caption: str = ""
    lyrics: str = ""
    is_instrumental: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    license_id: str | None = None
    lyrics_source: LyricsSource | None = None


class ModelRecord(BaseModel):
    """CAT-002 모델 레지스트리 항목 (LoRA 한 버전)."""

    id: str = Field(default_factory=new_id)
    category_id: str | None
    engine: Engine
    base_model: str
    adapter_path: str
    version: int
    dataset_id: str | None = None
    train_config: dict[str, Any] = Field(default_factory=dict)
    file_hash: str = ""
    validation_status: ValidationStatus = ValidationStatus.UNVERIFIED
    notes: str = ""
    display_name: str = ""          # 사용자가 붙인 이름 (목록에 표시)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def label(self) -> str:
        """화면 표시용: '내가 붙인 이름 (v3)' — 이름이 없으면 어댑터 폴더 위치로 대신 표시."""
        if self.display_name:
            return f"{self.display_name} (v{self.version})"
        parts = Path(self.adapter_path).parts
        short = "/".join(parts[-3:]) if len(parts) >= 3 else self.adapter_path
        return f"v{self.version} · …/{short}"


class Job(BaseModel):
    id: str = Field(default_factory=new_id)
    kind: JobKind
    status: JobStatus = JobStatus.CREATED
    category_id: str | None = None
    dataset_id: str | None = None
    model_id: str | None = None
    progress: float | None = None   # None = 진행률 알 수 없음 (인위적 100% 금지)
    params: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    retries: int = 0
    heartbeat_at: datetime | None = None
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    ended_at: datetime | None = None


class JobEvent(BaseModel):
    id: int | None = None
    job_id: str
    timestamp: datetime = Field(default_factory=utcnow)
    level: str = "info"
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class LyricsDraft(BaseModel):
    id: str = Field(default_factory=new_id)
    category_id: str | None = None
    lineage_id: str = Field(default_factory=new_id)  # 같은 가사의 버전들을 묶는 ID
    title: str = ""
    topic: str = ""
    lyrics: str
    version: int = 1
    source: LyricsSource = LyricsSource.AI_GENERATED
    model_id: str | None = None      # None = Qwen 기본 모델
    prompt: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime | None = None   # 가사집에서 마지막으로 저장(덮어쓰기)한 시각


class Song(BaseModel):
    id: str = Field(default_factory=new_id)
    category_id: str | None = None
    lyrics_draft_id: str | None = None
    music_model_id: str | None = None   # None = ACE 기본 모델 (LoRA OFF)
    qwen_model_id: str | None = None
    title: str = ""
    audio_path: str = ""
    prompt: str = ""
    generation: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = Field(default_factory=utcnow)
