"""가사 생성/편집/버전 관리 서비스 (LYR-004)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.adapters.ports import LLMRequest, LyricsEnginePort
from app.core.errors import NotFoundError, ValidationError
from app.core.observer import EventBus, Topics
from app.domain.entities import Category, LyricsDraft, StyleConstraints
from app.domain.enums import Engine, JobKind, LyricsSource
from app.repositories.content_repo import LyricsRepository
from app.services.category_service import CategoryService
from app.services.job_service import JobContext, JobHandler
from app.services.lyrics_analysis import LyricsReport, analyze, clean_llm_lyrics

LYRICS_SYSTEM = (
    "당신은 한국어 대중가요 전문 작사가입니다. 요청한 구조 마커([Verse 1], [Pre-Chorus], [Chorus], "
    "[Bridge], [Outro])를 사용해 가사만 출력합니다. 설명, 해설, 코드블록은 쓰지 않습니다. "
    "첫 줄에 '제목: ...' 한 줄을 쓰고 이어서 가사를 씁니다. 실존 곡의 가사를 인용하거나 복제하지 않습니다."
)


class LyricsRequest(BaseModel):
    category_id: str | None = None   # 선택: 카테고리 작사 지시/금지어 적용용
    model_id: str | None = None      # 가사 모델 (Qwen LoRA). None = Qwen 기본 모델
    topic: str = Field(min_length=1)
    speaker: str = ""
    mood: str = ""
    viewpoint: str = ""
    structure: list[str] = Field(default_factory=lambda: ["Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Chorus",
                                                          "Bridge", "Chorus"])
    syllable_min: int | None = Field(default=7, ge=1)
    syllable_max: int | None = Field(default=12, ge=1)
    forbidden: list[str] = Field(default_factory=list)
    variants: int = Field(default=1, ge=1, le=4)
    use_lora: bool = False         # (예전 방식) model_id 없을 때 카테고리 가사 LoRA 사용
    temperature: float = 0.8
    max_tokens: int = 1200
    seed: int | None = None


def build_lyrics_prompt(category: Category | None, req: LyricsRequest) -> str:
    style = category.style if category else StyleConstraints()
    parts: list[str] = []
    if category:
        parts.append(f"카테고리: {category.name}" + (f" ({category.description})" if category.description else ""))
    parts.append(f"주제: {req.topic}")
    if req.speaker:
        parts.append(f"화자: {req.speaker}")
    if req.mood or style.mood_tags:
        parts.append(f"분위기: {req.mood or ', '.join(style.mood_tags)}")
    if req.viewpoint:
        parts.append(f"시점: {req.viewpoint}")
    if style.era:
        parts.append(f"시대 감성: {style.era}")
    parts.append(f"구조: {', '.join(req.structure)}")
    if req.syllable_min or req.syllable_max:
        parts.append(f"각 줄 {req.syllable_min or 1}~{req.syllable_max or 20}음절")
    forbidden = list(dict.fromkeys(req.forbidden + style.forbidden_terms))
    if forbidden:
        parts.append(f"금지 표현: {', '.join(forbidden)}")
    if style.lyrics_instructions:
        parts.append(f"작사 지시: {style.lyrics_instructions}")
    lang_code = category.language if category else "ko"
    language = "한국어" if lang_code == "ko" else lang_code
    parts.append(f"{language}로 작성. 후렴(Chorus)은 반복될 때 같은 가사를 유지.")
    return "\n".join(parts)


class LyricsService:
    def __init__(self, engine: LyricsEnginePort, repo: LyricsRepository, categories: CategoryService,
                 bus: EventBus | None = None) -> None:
        self.engine = engine
        self.repo = repo
        self.categories = categories
        self.bus = bus or EventBus()

    async def generate(self, req: LyricsRequest) -> list[LyricsDraft]:
        category = self.categories.get(req.category_id) if req.category_id else None
        if req.model_id:
            model = self.categories.models.get(req.model_id)
            if model is None or model.engine != Engine.QWEN:
                raise ValidationError("가사 모델(Qwen LoRA)을 찾을 수 없습니다.", detail=req.model_id)
        elif category and req.use_lora:
            model = self.categories.active_model(category.id, Engine.QWEN)
        else:
            model = None
        user_prompt = build_lyrics_prompt(category, req)
        drafts: list[LyricsDraft] = []
        for i in range(req.variants):
            seed = None if req.seed is None else req.seed + i
            result = await self.engine.generate(LLMRequest(
                system=LYRICS_SYSTEM, user=user_prompt, adapter_path=model.adapter_path if model else None,
                max_tokens=req.max_tokens, temperature=req.temperature, seed=seed,
            ))
            lyrics, title = clean_llm_lyrics(result.text)
            if not lyrics.strip():
                raise ValidationError("Qwen 이 빈 가사를 반환했습니다.", detail=result.text[:300])
            draft = LyricsDraft(
                category_id=category.id if category else None, title=title or req.topic[:40], topic=req.topic,
                lyrics=lyrics, version=1,
                source=LyricsSource.AI_GENERATED, model_id=model.id if model else None,
                prompt={
                    "system": LYRICS_SYSTEM, "user": user_prompt, "seed": seed, "temperature": req.temperature,
                    "max_tokens": req.max_tokens, "engine_model": result.model,
                    "adapter_path": result.adapter_path, "is_mock": result.is_mock,
                    "elapsed_s": round(result.elapsed_s, 2), "request": req.model_dump(),
                },
            )
            self.repo.add(draft)
            drafts.append(draft)
            self.bus.publish(Topics.LYRICS_SAVED, draft=draft)
        return drafts

    def save_edit(self, draft_id: str, lyrics: str, title: str | None = None) -> LyricsDraft:
        """수정본을 새 버전으로 저장. 원본은 보존된다."""
        base = self.repo.get(draft_id)
        if base is None:
            raise NotFoundError("가사를 찾을 수 없습니다.", detail=draft_id)
        if not lyrics.strip():
            raise ValidationError("가사가 비어 있습니다.")
        if lyrics.strip() == base.lyrics.strip() and (title is None or title == base.title):
            return base
        new = LyricsDraft(
            category_id=base.category_id, lineage_id=base.lineage_id, title=title if title is not None else base.title,
            topic=base.topic, lyrics=lyrics.strip(), version=self.repo.latest_version(base.lineage_id) + 1,
            source=base.source,  # 출처 구분은 유지하고 수동 수정 여부는 prompt 메타로 기록
            model_id=base.model_id, prompt={**base.prompt, "edited_from": base.id},
        )
        new.prompt["manually_edited"] = True
        self.repo.add(new)
        self.bus.publish(Topics.LYRICS_SAVED, draft=new)
        return new

    def create_manual(self, category_id: str | None, title: str, lyrics: str, topic: str = "") -> LyricsDraft:
        if category_id:
            self.categories.get(category_id)
        draft = LyricsDraft(category_id=category_id, title=title, topic=topic, lyrics=lyrics.strip(),
                            source=LyricsSource.USER_ORIGINAL)
        self.repo.add(draft)
        self.bus.publish(Topics.LYRICS_SAVED, draft=draft)
        return draft

    def check(self, lyrics: str, category_id: str | None = None, syllable_min: int | None = 7,
              syllable_max: int | None = 12, forbidden: list[str] | None = None) -> LyricsReport:
        terms = list(forbidden or [])
        if category_id:
            terms += self.categories.get(category_id).style.forbidden_terms
        return analyze(lyrics, syllable_min, syllable_max, terms)

    def history(self, draft_id: str) -> list[LyricsDraft]:
        draft = self.repo.get(draft_id)
        return self.repo.history(draft.lineage_id) if draft else []

    def list_recent(self, limit: int = 50) -> list[LyricsDraft]:
        return self.repo.list_recent(limit)

    def list_for_category(self, category_id: str) -> list[LyricsDraft]:
        return self.repo.list_for_category(category_id)

    def get(self, draft_id: str) -> LyricsDraft:
        draft = self.repo.get(draft_id)
        if draft is None:
            raise NotFoundError("가사를 찾을 수 없습니다.", detail=draft_id)
        return draft


class LyricsGenerateHandler(JobHandler):
    kind = JobKind.LYRICS_GENERATE

    def __init__(self, service: LyricsService) -> None:
        self.service = service

    def validate(self, params: dict[str, Any]) -> None:
        LyricsRequest(**params)

    def describe(self, params: dict[str, Any]) -> str:
        return "Qwen 작사"

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        req = LyricsRequest(**ctx.params)
        ctx.log(f"Qwen 작사 시작 (버전 {req.variants}개)")
        ctx.progress(None)
        drafts = await self.service.generate(req)
        ctx.log(f"가사 {len(drafts)}개 저장")
        return {"draft_ids": [d.id for d in drafts]}
