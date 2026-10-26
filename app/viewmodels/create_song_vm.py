"""곡 만들기 VM: ① 가사집에서 가사 선택 → ② 음악 프롬프트(선택) → ③ 음악 모델 선택·생성 → 재생.

가사는 여기서 쓰거나 고치지 않는다 (가사집에서 만들고 고친다). 여기서는 고르기만 한다.
음악 모델 목록은 [음악 LoRA 학습 → 모델·카테고리] 에서 나눈 카테고리로 걸러 볼 수 있다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.observer import ObservableProperty, Topics
from app.domain.entities import Category, Job, LyricsDraft, ModelRecord, Song
from app.domain.enums import Engine, JobKind, JobStatus
from app.repositories.lyric_category_repo import LyricCategory
from app.services.lyrics_service import LYRICS_LANGUAGES, VOCAL_LANGUAGE
from app.services.training_service import detect_language
from app.services.prompt_service import MusicPrompt, free_category
from app.viewmodels.base import BaseViewModel, JobTracker

ALL = ""   # '전체' 카테고리 필터 값


class CreateSongViewModel(BaseViewModel):
    LANGUAGES = LYRICS_LANGUAGES   # {"ko": "한국어", "en": "영어", "ja": "일본어"}

    # --- 음악 모델 선택 ---
    categories = ObservableProperty[list[Category]](factory=list)          # 음악 모델 분류
    music_filter = ObservableProperty(ALL)                                 # 음악 모델 카테고리 필터
    music_models = ObservableProperty[list[ModelRecord]](factory=list)     # 필터된 음악 LoRA 목록
    music_model_id = ObservableProperty[str | None](None)                  # None = ACE 기본
    output_dir = ObservableProperty("")                                    # 결과 저장 폴더
    language = ObservableProperty("ko")                                    # 노래(보컬) 언어 — 가사를 고르면 자동
    song_title = ObservableProperty("")                                    # 곡 제목 (비우면 가사 제목)

    # --- ① 가사 선택 (가사집) ---
    lyric_categories = ObservableProperty[list[LyricCategory]](factory=list)
    lyric_filter = ObservableProperty(ALL)    # 들어올 때는 항상 '전체'
    drafts = ObservableProperty[list[LyricsDraft]](factory=list)
    draft_query = ObservableProperty("")
    draft_total = ObservableProperty(0)
    PAGE = 50
    draft_id = ObservableProperty[str | None](None)    # 고른 가사
    title = ObservableProperty("")
    lyrics_text = ObservableProperty("")

    # --- 음악 ---
    music_prompt = ObservableProperty[MusicPrompt | None](None)
    prompt_text = ObservableProperty("")
    bpm = ObservableProperty[int | None](None)

    job = ObservableProperty[Job | None](None)
    job_logs = ObservableProperty[list[str]](factory=list)
    songs = ObservableProperty[list[Song]](factory=list)

    def __init__(self, container=None, lyric_id: str | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.tracker = JobTracker(self, self._on_job_update, self._on_job_log)
        self.listen(Topics.CATEGORY_CHANGED, lambda e: self.load_models())
        self.listen(Topics.LYRICS_SAVED, lambda e: self.reload_drafts())   # 가사집에서 바뀐 내용 반영
        self.load_models()
        self.reload_drafts()
        # 다른 화면에 다녀와도 실행 중인 작사/음악 생성 작업을 다시 표시 (끝난 작업은 다시 불러오지 않음)
        self.tracker.resume([JobKind.PROMPT_GENERATE, JobKind.MUSIC_GENERATE], recent_minutes=0)
        if lyric_id:                       # 가사집의 [이 가사로 곡 만들기]
            self.select_lyric(lyric_id)

    # --- 조회 ---
    @property
    def job_running(self) -> bool:
        return self.job is not None and not self.job.status.is_terminal

    @property
    def music_model(self) -> ModelRecord | None:
        if not self.music_model_id:
            return None
        return self.c.categories.models.get(self.music_model_id)

    @property
    def selection_summary(self) -> str:
        lyric = self.title or "선택 안 함"
        mm = self.music_model.label if self.music_model else "ACE 기본 (LoRA 없음)"
        return f"가사: {lyric}  →  음악 모델: {mm}  ·  노래 언어: {self.LANGUAGES.get(self.language, self.language)}"

    # --- 모델 선택 ---
    def load_models(self) -> None:
        self.categories = self.c.categories.list()
        if self.music_filter and self.music_filter not in {c.id for c in self.categories}:
            self.music_filter = ALL
        self._load_music_models()

    def _load_music_models(self) -> None:
        if self.music_filter:
            self.music_models = self.c.categories.category_models(self.music_filter, Engine.ACE)
        else:
            self.music_models = self.c.categories.list_models(engine=Engine.ACE)
        if self.music_model_id not in {m.id for m in self.music_models}:
            # 카테고리를 고르면 그 카테고리의 '사용 중' 모델을 기본으로
            active = (self.c.categories.active_model(self.music_filter, Engine.ACE)
                      if self.music_filter else None)
            self.music_model_id = active.id if active else None

    def set_music_filter(self, category_id: str | None) -> None:
        self.music_filter = category_id or ALL
        self._load_music_models()

    def set_music_model(self, model_id: str | None) -> None:
        self.music_model_id = model_id or None

    @property
    def browse_root(self) -> Path:
        return Path.home()

    def browse(self, path: str | None = None) -> tuple[Path, list[tuple[str, bool]]]:
        result = self.sync_command(lambda: self.c.categories.browse_dirs(self.browse_root, path))
        return result if result else (self.browse_root, [])

    # --- ① 가사 선택 (가사집 목록: 카테고리 필터 + 검색 + 50개씩) ---
    def reload_drafts(self, limit: int | None = None) -> None:
        self.lyric_categories = self.c.lyrics_book.categories()
        if self.lyric_filter and self.lyric_filter not in {c.id for c in self.lyric_categories}:
            self.lyric_filter = ALL
        limit = limit or max(self.PAGE, len(self.drafts))
        self.drafts, self.draft_total = self.c.lyrics_book.list(self.lyric_filter or None, self.draft_query, limit)
        if self.draft_id:                      # 가사집에서 고친 내용이 있으면 반영
            cur = self.c.lyrics.get(self.draft_id)
            if cur is None:
                self.clear_lyric()
            elif cur.lyrics != self.lyrics_text or cur.title != self.title:
                self.title, self.lyrics_text = cur.title, cur.lyrics

    def set_lyric_filter(self, category_id: str | None) -> None:
        self.lyric_filter = category_id or ALL
        self.reload_drafts(self.PAGE)

    def search_drafts(self, query: str | None) -> None:
        self.draft_query = query or ""
        self.reload_drafts(self.PAGE)

    def more_drafts(self) -> None:
        self.reload_drafts(len(self.drafts) + self.PAGE)

    def select_lyric(self, lyric_id: str) -> None:
        draft = self.sync_command(lambda: self.c.lyrics_book.get(lyric_id))
        if not draft:
            return
        self.draft_id = draft.id
        self.title = draft.title
        self.lyrics_text = draft.lyrics
        self.language = detect_language(draft.lyrics)   # 가사 언어 → 노래 언어 (바꿀 수 있음)

    load_draft = select_lyric   # 예전 이름 호환

    def clear_lyric(self) -> None:
        self.draft_id, self.title, self.lyrics_text = None, "", ""

    # --- ② 음악 프롬프트 ---
    async def generate_prompt(self, wishes: str = "") -> None:
        if not self._ensure_saved():
            return
        self._submit(JobKind.PROMPT_GENERATE, {"draft_id": self.draft_id, "wishes": wishes})

    def apply_manual_prompt(self) -> bool:
        checked = self.sync_command(
            lambda: self.c.director.validate_manual(free_category(), self.prompt_text, self.bpm))
        if checked:
            self.music_prompt = checked
            self.prompt_text = checked.prompt
            self.bpm = checked.bpm
            return True
        return False

    # --- ③ 음악 생성 ---
    async def generate_music(self, options: dict[str, Any]) -> None:
        if not self._ensure_saved():
            return
        # 음악 프롬프트는 선택 사항 — 비어 있으면 가사와 음악 모델만으로 생성
        prompt = self.prompt_text.strip()
        if prompt and (self.music_prompt is None or self.music_prompt.prompt != prompt) \
                and not self.apply_manual_prompt():
            return
        if self.output_dir.strip() and self.sync_command(
                lambda: self.c.music.resolve_output_dir(self.output_dir)) is None:
            return
        seed = options.get("seed")
        spec = {
            "title": self.song_title.strip() or self.title, "draft_id": self.draft_id, "prompt": prompt,
            "vocal_language": VOCAL_LANGUAGE.get(self.language, "ko"), "bpm": self.bpm,
            "use_lora": self.music_model_id is not None, "model_id": self.music_model_id,
            "output_dir": self.output_dir.strip(),
            "audio_duration": float(options["duration"]) if options.get("duration") else None,
            "seed": int(seed) if seed not in (None, "") else None,
            "audio_format": options.get("audio_format", "mp3"),
            "prompt_meta": {"source": (self.music_prompt.source if self.music_prompt else "manual") if prompt
                            else "none",
                            "notes": self.music_prompt.notes if (prompt and self.music_prompt) else []},
        }
        ab = bool(options.get("ab_compare")) and self.music_model_id is not None
        self._submit(JobKind.MUSIC_GENERATE, {"spec": spec, "ab_compare": ab})

    async def cancel_job(self) -> None:
        if self.job:
            await self.run_command(lambda: self.c.jobs.cancel(self.job.id))

    # --- 내부 ---
    def _ensure_saved(self) -> bool:
        """가사가 골라져 있는지 (가사는 가사집에 이미 저장되어 있다)."""
        if not self.draft_id or self.c.lyrics.get(self.draft_id) is None:
            self.error = "① 에서 가사집의 가사를 먼저 고르세요."
            return False
        return True

    def _submit(self, kind: JobKind, params: dict[str, Any]) -> None:
        if self.job_running:
            self.error = "이전 작업이 아직 실행 중입니다."
            return
        self.job_logs = []
        job = self.sync_command(lambda: self.c.jobs.submit(kind, params))
        if job:
            self.tracker.track(job)

    def _on_job_log(self, line: str) -> None:
        self.job_logs = (self.job_logs + [line])[-200:]

    def _on_job_update(self, job: Job) -> None:
        self.job = job
        if job.status == JobStatus.FAILED:
            self.error = job.error or "작업이 실패했습니다."
        if job.status != JobStatus.SUCCEEDED:
            return
        if job.kind == JobKind.PROMPT_GENERATE:
            mp = MusicPrompt(**job.result["music_prompt"])
            self.music_prompt = mp
            self.prompt_text = mp.prompt
            self.bpm = mp.bpm
            self.info = f"음악 프롬프트 생성 완료 (출처: {mp.source})"
        elif job.kind == JobKind.MUSIC_GENERATE:
            self.songs = [s for sid in job.result.get("song_ids", []) if (s := self.c.song_repo.get(sid))]
            folder = str(Path(self.songs[0].audio_path).parent) if self.songs else ""
            self.info = f"음원 {len(self.songs)}개 생성 완료 → {folder}"
