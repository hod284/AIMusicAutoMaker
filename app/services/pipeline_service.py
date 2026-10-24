"""원클릭 완성곡 파이프라인 (Phase 6): 작사 → 음악 프롬프트 → 음악 생성.

각 단계 결과를 즉시 저장하므로 음악 생성이 실패해도 가사/프롬프트/작업 이력은 남는다.
GPU 락은 잡 전체에 한 번만 걸리므로 Qwen → ACE 가 순차 실행된다(32GB 메모리 경쟁 방지).
"""

from __future__ import annotations

from typing import Any

from app.domain.enums import JobKind
from app.services.job_service import JobContext, JobHandler
from app.services.lyrics_service import LyricsRequest, LyricsService
from app.services.music_service import MusicGenerationSpec, MusicService
from app.services.prompt_service import PromptDirector


class FullSongHandler(JobHandler):
    kind = JobKind.FULL_SONG

    def __init__(self, lyrics: LyricsService, director: PromptDirector, music: MusicService) -> None:
        self.lyrics = lyrics
        self.director = director
        self.music = music

    def validate(self, params: dict[str, Any]) -> None:
        LyricsRequest(category_id=params.get("category_id", ""), topic=params.get("topic", ""))

    def describe(self, params: dict[str, Any]) -> str:
        return "완성곡 자동 생성"

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        result: dict[str, Any] = {}
        ctx.log("1/3 Qwen 작사")
        drafts = await self.lyrics.generate(LyricsRequest(category_id=p["category_id"], topic=p["topic"],
                                                          mood=p.get("mood", ""), variants=1))
        draft = drafts[0]
        result["draft_id"] = draft.id
        ctx.progress(0.25)
        ctx.check_cancelled()

        ctx.log("2/3 음악 프롬프트 (Qwen 기본 모델)")
        category = self.lyrics.categories.get(p["category_id"])
        prompt = await self.director.direct(category, draft.lyrics, p.get("wishes", ""))
        result["music_prompt"] = prompt.model_dump()
        ctx.progress(0.4)
        ctx.check_cancelled()

        ctx.log("3/3 ACE-Step 음악 생성")
        spec = MusicGenerationSpec(
            category_id=category.id, title=draft.title, draft_id=draft.id, prompt=prompt.prompt,
            vocal_language=prompt.vocal_language, bpm=prompt.bpm, use_lora=p.get("use_lora", True),
            audio_duration=p.get("audio_duration"), seed=p.get("seed"),
            prompt_meta={"source": prompt.source, "notes": prompt.notes},
        )
        songs = await self.music.generate(spec, ctx.log, ctx.heartbeat)
        result["song_ids"] = [s.id for s in songs]
        return result
