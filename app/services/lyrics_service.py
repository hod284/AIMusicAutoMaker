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

# 가사 언어별 작사 지시 (화면에서 고른 '가사 언어' 에 따라 자동 선택)
LYRICS_LANGUAGES: dict[str, str] = {"ko": "한국어", "ko_en": "한국어 + 영어 (K-pop 스타일)", "en": "영어", "ja": "일본어"}
TITLE_LABEL = {"ko": "제목", "ko_en": "제목", "en": "Title", "ja": "タイトル"}
# ACE-Step 에 보내는 보컬 언어 (섞어 쓰기는 주 언어 기준)
VOCAL_LANGUAGE = {"ko": "ko", "ko_en": "ko", "en": "en", "ja": "ja"}

_SYSTEM = {
    "ko": ("당신은 한국어 대중가요 전문 작사가입니다. 요청한 구조 마커([Verse 1], [Pre-Chorus], [Chorus], "
           "[Bridge], [Outro])를 사용해 가사만 출력합니다. 설명, 해설, 코드블록은 쓰지 않습니다. "
           "첫 줄에 '제목: ...' 한 줄을 쓰고 이어서 가사를 씁니다. 실존 곡의 가사를 인용하거나 복제하지 않습니다."),
    "ko_en": ("당신은 K-pop 전문 작사가입니다. 가사는 한국어 위주로 쓰되, 후렴이나 강조하는 구절에는 "
              "자연스러운 영어 표현을 섞습니다 (예: 'baby', 'tonight', 'I miss you'). 한자·일본어 등 다른 언어는 쓰지 않습니다. "
              "요청한 구조 마커([Verse 1], [Pre-Chorus], [Chorus], [Bridge], [Outro])를 사용해 가사만 출력합니다. "
              "설명, 해설, 코드블록은 쓰지 않습니다. 첫 줄에 '제목: ...' 한 줄을 쓰고 이어서 가사를 씁니다. "
              "실존 곡의 가사를 인용하거나 복제하지 않습니다."),
    "en": ("You are a professional English pop songwriter. Use the requested section markers ([Verse 1], "
           "[Pre-Chorus], [Chorus], [Bridge], [Outro]) and output only the lyrics, written entirely in English. "
           "No explanations, commentary or code blocks. Write the first line as 'Title: ...' and then the lyrics. "
           "Never quote or copy lyrics from existing songs."),
    "ja": ("あなたは日本語ポップスのプロの作詞家です。指定された構成マーカー([Verse 1], [Pre-Chorus], [Chorus], "
           "[Bridge], [Outro])を使い、日本語の歌詞だけを出力します。説明・解説・コードブロックは書きません。"
           "1行目に 'タイトル: ...' を書き、続けて歌詞を書きます。実在する曲の歌詞を引用・複製しません。"),
}
_WRITE_IN = {
    "ko": "한국어로 작성. 후렴(Chorus)은 반복될 때 같은 가사를 유지.",
    "ko_en": "한국어 위주로 작성하고 영어 구절을 자연스럽게 섞기. 후렴(Chorus)은 반복될 때 같은 가사를 유지.",
    "en": "Write in English. Keep the Chorus lyrics identical when it repeats.",
    "ja": "日本語で作詞。Chorus は繰り返す時に同じ歌詞を保つ。",
}


def lyrics_system(language: str = "ko") -> str:
    return _SYSTEM.get(language, _SYSTEM["ko"])


def write_in_line(language: str = "ko") -> str:
    return _WRITE_IN.get(language, _WRITE_IN["ko"])


LYRICS_SYSTEM = lyrics_system("ko")   # 예전 코드 호환


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
    language: str = "ko"             # 가사 언어: ko / en / ja
    variants: int = Field(default=1, ge=1, le=4)
    use_lora: bool = False         # (예전 방식) model_id 없을 때 카테고리 가사 LoRA 사용
    temperature: float = 0.7          # 0.8 이상이면 다국어 모델이 다른 언어 토큰을 섞기 쉽다
    max_tokens: int = 1200
    seed: int | None = None


_LABELS = {
    "ko": {"category": "카테고리", "topic": "주제", "speaker": "화자", "mood": "분위기", "viewpoint": "시점",
           "era": "시대 감성", "structure": "구조", "syllables": "각 줄 {a}~{b}음절", "forbidden": "금지 표현",
           "instructions": "작사 지시"},
    "en": {"category": "Category", "topic": "Topic", "speaker": "Narrator", "mood": "Mood", "viewpoint": "Point of view",
           "era": "Era", "structure": "Structure", "syllables": "Each line {a}-{b} syllables", "forbidden": "Avoid",
           "instructions": "Instructions"},
    "ja": {"category": "カテゴリー", "topic": "テーマ", "speaker": "語り手", "mood": "雰囲気", "viewpoint": "視点",
           "era": "時代感", "structure": "構成", "syllables": "各行{a}〜{b}音節", "forbidden": "禁止表現",
           "instructions": "作詞指示"},
}


def prompt_labels(language: str = "ko") -> dict[str, str]:
    """요청(user) 항목 이름. 가사 언어와 같은 언어로 써야 모델이 그 언어로 답하기 쉽다 (학습 데이터도 같은 형식)."""
    return _LABELS.get("ko" if language == "ko_en" else language, _LABELS["ko"])


def build_lyrics_prompt(category: Category | None, req: LyricsRequest) -> str:
    style = category.style if category else StyleConstraints()
    L = prompt_labels(req.language)
    parts: list[str] = []
    if category:
        parts.append(f"{L['category']}: {category.name}" + (f" ({category.description})" if category.description else ""))
    parts.append(f"{L['topic']}: {req.topic}")
    if req.speaker:
        parts.append(f"{L['speaker']}: {req.speaker}")
    if req.mood or style.mood_tags:
        parts.append(f"{L['mood']}: {req.mood or ', '.join(style.mood_tags)}")
    if req.viewpoint:
        parts.append(f"{L['viewpoint']}: {req.viewpoint}")
    if style.era:
        parts.append(f"{L['era']}: {style.era}")
    parts.append(f"{L['structure']}: {', '.join(req.structure)}")
    if req.syllable_min or req.syllable_max:
        parts.append(L["syllables"].format(a=req.syllable_min or 1, b=req.syllable_max or 20))
    forbidden = list(dict.fromkeys(req.forbidden + style.forbidden_terms))
    if forbidden:
        parts.append(f"{L['forbidden']}: {', '.join(forbidden)}")
    if style.lyrics_instructions:
        parts.append(f"{L['instructions']}: {style.lyrics_instructions}")
    parts.append(write_in_line(req.language))
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
                system=lyrics_system(req.language), user=user_prompt, adapter_path=model.adapter_path if model else None,
                max_tokens=req.max_tokens, temperature=req.temperature, seed=seed,
                lock_language=req.language,
                sections=list(req.structure) or None, title_label=TITLE_LABEL.get(req.language, "제목"),
            ))
            lyrics, title = clean_llm_lyrics(result.text)
            if not lyrics.strip():
                raise ValidationError("Qwen 이 빈 가사를 반환했습니다.", detail=result.text[:300])
            draft = LyricsDraft(
                category_id=category.id if category else None, title=title or req.topic[:40], topic=req.topic,
                lyrics=lyrics, version=1,
                source=LyricsSource.AI_GENERATED, model_id=model.id if model else None,
                prompt={
                    "system": lyrics_system(req.language), "user": user_prompt, "seed": seed, "temperature": req.temperature,
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
        """덮어쓰기 저장 (가사 하나 = 기록 하나, 버전을 새로 만들지 않는다)."""
        base = self.repo.get(draft_id)
        if base is None:
            raise NotFoundError("가사를 찾을 수 없습니다.", detail=draft_id)
        if not lyrics.strip():
            raise ValidationError("가사가 비어 있습니다.")
        new_title = base.title if title is None else (title.strip() or base.title)
        if lyrics.strip() != base.lyrics.strip() or new_title != base.title:
            self.repo.update_content(base.id, new_title, lyrics.strip())
            base = self.repo.get(base.id)  # type: ignore[assignment]
            self.bus.publish(Topics.LYRICS_SAVED, draft=base)
        return base  # type: ignore[return-value]

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

    def list_latest(self, query: str = "", limit: int = 50) -> tuple[list[LyricsDraft], int]:
        """(가사 목록, 검색 결과 전체 개수)"""
        return self.repo.list_latest(query, limit), self.repo.count_latest(query)

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
