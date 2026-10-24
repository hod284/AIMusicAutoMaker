"""음악 프롬프트 AI 디렉터 (LYR-005, CAT-003).

패턴:
- **Strategy**: ``QwenDirectorStrategy`` (Qwen 기본 모델이 JSON 생성) / ``TemplateStrategy`` (카테고리 템플릿)
  → ``PromptDirector`` 가 순서대로 시도하고, 실패하면 다음 전략으로 넘어간다.
- **Chain of Responsibility**: ``Guardrail`` 체인이 결과를 검증·보정한다.
  새 규칙은 Guardrail 서브클래스를 만들어 체인에 추가하면 된다.

Qwen 은 가사 LoRA 가 아닌 **기본 모델**을 사용한다 (역할 분리).
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from app.adapters.ports import LLMRequest, LyricsEnginePort
from app.core.errors import ValidationError
from app.domain.entities import Category, StyleConstraints
from app.domain.enums import JobKind
from app.services.job_service import JobContext, JobHandler

log = logging.getLogger(__name__)


class MusicPrompt(BaseModel):
    """ACE 로 전달할 음악 조건. 스키마 검증 대상."""

    prompt: str = Field(min_length=3, max_length=1000)
    genre: str = ""
    vocal_language: str = "ko"
    bpm: int | None = Field(default=None, ge=30, le=300)
    instruments: list[str] = Field(default_factory=list)
    mood: list[str] = Field(default_factory=list)
    structure_hint: str = ""
    source: str = ""                 # "qwen" | "template" | "manual"
    notes: list[str] = Field(default_factory=list)  # 가드레일 보정 내역


class GuardrailViolation(Exception):
    pass


# ---------------------------------------------------------------------------
# Chain of Responsibility: 가드레일
# ---------------------------------------------------------------------------
class Guardrail(ABC):
    def __init__(self) -> None:
        self._next: Guardrail | None = None

    def then(self, nxt: "Guardrail") -> "Guardrail":
        self._next = nxt
        return nxt

    def handle(self, prompt: MusicPrompt, style: StyleConstraints) -> MusicPrompt:
        prompt = self.check(prompt, style)
        return self._next.handle(prompt, style) if self._next else prompt

    @abstractmethod
    def check(self, prompt: MusicPrompt, style: StyleConstraints) -> MusicPrompt: ...


class NonEmptyGuard(Guardrail):
    def check(self, prompt: MusicPrompt, style: StyleConstraints) -> MusicPrompt:
        if not prompt.prompt.strip():
            raise GuardrailViolation("빈 프롬프트")
        return prompt


class GenreGuard(Guardrail):
    """허용 장르가 설정된 경우 장르/프롬프트가 그 범위 안인지 확인. 이탈 시 거부."""

    def check(self, prompt: MusicPrompt, style: StyleConstraints) -> MusicPrompt:
        if not style.allowed_genres:
            return prompt
        allowed = [g.lower().replace("_", " ") for g in style.allowed_genres]
        text = f"{prompt.genre} {prompt.prompt}".lower().replace("_", " ")
        if not any(g in text for g in allowed):
            raise GuardrailViolation(f"허용 장르({', '.join(style.allowed_genres)}) 이탈: {prompt.genre!r}")
        return prompt


class ForbiddenTermGuard(Guardrail):
    def check(self, prompt: MusicPrompt, style: StyleConstraints) -> MusicPrompt:
        found = [t for t in style.forbidden_terms if t and t.lower() in prompt.prompt.lower()]
        if found:
            raise GuardrailViolation(f"금지어 포함: {found}")
        return prompt


class BpmGuard(Guardrail):
    """BPM 은 거부하지 않고 선호 범위로 보정한다."""

    def check(self, prompt: MusicPrompt, style: StyleConstraints) -> MusicPrompt:
        lo, hi = style.preferred_bpm_min, style.preferred_bpm_max
        if lo is None and hi is None:
            return prompt
        bpm = prompt.bpm
        if bpm is None:
            new = round(((lo or hi) + (hi or lo)) / 2)  # type: ignore[operator]
            prompt.notes.append(f"BPM 미지정 → {new}")
            prompt.bpm = new
        elif lo is not None and bpm < lo:
            prompt.notes.append(f"BPM {bpm} → {lo} (범위 보정)")
            prompt.bpm = lo
        elif hi is not None and bpm > hi:
            prompt.notes.append(f"BPM {bpm} → {hi} (범위 보정)")
            prompt.bpm = hi
        return prompt


class VocalGuard(Guardrail):
    """보컬 성별 등 고정 조건이 프롬프트에 없으면 추가한다."""

    def check(self, prompt: MusicPrompt, style: StyleConstraints) -> MusicPrompt:
        if style.vocal and style.vocal.lower() not in prompt.prompt.lower():
            prompt.prompt = f"{prompt.prompt.rstrip(', ')}, {style.vocal} vocals"
            prompt.notes.append(f"보컬 조건 추가: {style.vocal}")
        return prompt


def default_guardrails() -> Guardrail:
    head = NonEmptyGuard()
    head.then(ForbiddenTermGuard()).then(GenreGuard()).then(BpmGuard()).then(VocalGuard())
    return head


# ---------------------------------------------------------------------------
# Strategy: 프롬프트 생성 전략
# ---------------------------------------------------------------------------
class PromptStrategy(ABC):
    name: str

    @abstractmethod
    async def produce(self, category: Category, lyrics: str, wishes: str) -> MusicPrompt: ...


DIRECTOR_SYSTEM = (
    "You are a music director for the ACE-Step music generation model. Read the lyrics and category "
    "constraints, then output ONLY one JSON object (no markdown) with keys: "
    '"prompt" (English comma-separated music description: genre, instruments, vocal style, mood, production; '
    "max 60 words; never name real artists or songs), "
    '"genre" (string), "bpm" (integer), "vocal_language" (ISO code), "instruments" (array), '
    '"mood" (array), "structure_hint" (string).'
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"```(?:json)?", "", text)
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("JSON 객체를 찾을 수 없음")
    return json.loads(match.group(0))


class QwenDirectorStrategy(PromptStrategy):
    name = "qwen"

    def __init__(self, engine: LyricsEnginePort, temperature: float = 0.4) -> None:
        self.engine = engine
        self.temperature = temperature

    async def produce(self, category: Category, lyrics: str, wishes: str) -> MusicPrompt:
        s = category.style
        constraints = {
            "category": category.name, "description": category.description, "language": category.language,
            "allowed_genres": s.allowed_genres, "vocal": s.vocal, "bpm_range": [s.preferred_bpm_min, s.preferred_bpm_max],
            "instruments": s.instruments, "mood": s.mood_tags, "era": s.era, "forbidden_terms": s.forbidden_terms,
        }
        user = (
            f"Category constraints (JSON): {json.dumps(constraints, ensure_ascii=False)}\n"
            f"User wishes: {wishes or '(none)'}\n"
            f"Lyrics:\n{lyrics[:3000]}\n\nReturn the JSON object now."
        )
        # adapter_path=None → 가사 LoRA 가 아닌 Qwen 기본 모델
        result = await self.engine.generate(LLMRequest(system=DIRECTOR_SYSTEM, user=user, adapter_path=None,
                                                       temperature=self.temperature, max_tokens=512))
        data = extract_json(result.text)
        data.setdefault("vocal_language", category.language)
        if isinstance(data.get("bpm"), str) and data["bpm"].isdigit():
            data["bpm"] = int(data["bpm"])
        return MusicPrompt(**{k: v for k, v in data.items() if k in MusicPrompt.model_fields}, source="qwen")


class TemplateStrategy(PromptStrategy):
    """카테고리 기본 템플릿. LLM 없이 항상 성공해야 하는 마지막 fallback."""

    name = "template"

    async def produce(self, category: Category, lyrics: str, wishes: str) -> MusicPrompt:
        s = category.style
        if s.base_prompt_template.strip():
            text = s.base_prompt_template
        else:
            pieces = [", ".join(s.allowed_genres).replace("_", " ") or category.name]
            if s.era:
                pieces.append(f"{s.era} style")
            pieces += s.instruments
            pieces += s.mood_tags
            if wishes:
                pieces.append(wishes)
            text = ", ".join(p for p in pieces if p)
        genre = s.allowed_genres[0].replace("_", " ") if s.allowed_genres else ""
        if genre and genre.lower() not in text.lower():
            text = f"{genre}, {text}"
        return MusicPrompt(prompt=text, genre=genre, vocal_language=category.language, instruments=s.instruments,
                           mood=s.mood_tags, source="template")


class PromptDirector:
    def __init__(self, strategies: list[PromptStrategy], guardrails: Guardrail | None = None,
                 retries_per_strategy: int = 2) -> None:
        if not strategies:
            raise ValueError("전략이 하나 이상 필요합니다.")
        self.strategies = strategies
        self.guardrails = guardrails or default_guardrails()
        self.retries = retries_per_strategy

    async def direct(self, category: Category, lyrics: str, wishes: str = "") -> MusicPrompt:
        errors: list[str] = []
        for strategy in self.strategies:
            attempts = 1 if isinstance(strategy, TemplateStrategy) else self.retries
            for attempt in range(1, attempts + 1):
                try:
                    produced = await strategy.produce(category, lyrics, wishes)
                    checked = self.guardrails.handle(produced, category.style)
                    if errors:
                        checked.notes.append(f"이전 시도 실패 {len(errors)}회 후 {strategy.name} 사용")
                    return checked
                except (GuardrailViolation, ValueError, PydanticValidationError, json.JSONDecodeError) as exc:
                    msg = f"{strategy.name}#{attempt}: {exc}"
                    log.info("프롬프트 디렉터 재시도 — %s", msg)
                    errors.append(msg)
        raise ValidationError("음악 프롬프트를 만들지 못했습니다. 카테고리 템플릿/가드레일을 확인하세요.",
                              detail=errors)

    def validate_manual(self, category: Category, prompt_text: str, bpm: int | None = None) -> MusicPrompt:
        """사용자가 직접 수정한 프롬프트도 같은 가드레일을 통과해야 한다."""
        try:
            prompt = MusicPrompt(prompt=prompt_text, bpm=bpm, vocal_language=category.language, source="manual")
            return self.guardrails.handle(prompt, category.style)
        except (GuardrailViolation, PydanticValidationError) as exc:
            raise ValidationError(f"프롬프트 검증 실패: {exc}") from exc


def free_category() -> Category:
    """카테고리 없이 곡을 만들 때 쓰는 빈 규칙 (DB 에 저장하지 않음)."""
    return Category(slug="free", name="자유", style=StyleConstraints())


class PromptGenerateHandler(JobHandler):
    kind = JobKind.PROMPT_GENERATE

    def __init__(self, director: PromptDirector, categories: Any, lyrics: Any) -> None:
        self.director = director
        self.categories = categories
        self.lyrics = lyrics

    def validate(self, params: dict[str, Any]) -> None:
        if not (params.get("draft_id") or params.get("lyrics")):
            raise ValidationError("가사(draft_id 또는 lyrics)가 필요합니다.")

    def describe(self, params: dict[str, Any]) -> str:
        return "Qwen 음악 디렉터"

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        category = (self.categories.get(ctx.params["category_id"]) if ctx.params.get("category_id")
                    else free_category())
        lyrics = ctx.params.get("lyrics") or self.lyrics.get(ctx.params["draft_id"]).lyrics
        ctx.log("음악 프롬프트 생성 (Qwen 기본 모델)")
        prompt = await self.director.direct(category, lyrics, ctx.params.get("wishes", ""))
        ctx.log(f"프롬프트 출처: {prompt.source}", details=prompt.notes)
        return {"music_prompt": prompt.model_dump()}
