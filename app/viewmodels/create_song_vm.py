"""곡 만들기 VM: ① 가사집에서 가사 선택 → ② 음악 프롬프트(선택) → ③ 음악 모델 선택·생성 → 재생.

가사는 여기서 쓰거나 고치지 않는다 (가사집에서 만들고 고친다). 여기서는 고르기만 한다.
음악 모델 목록은 [음악 LoRA 학습 → 모델·카테고리] 에서 나눈 카테고리로 걸러 볼 수 있다.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
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
    wishes = ObservableProperty("")                 # AI 프롬프트 희망 요소
    bpm = ObservableProperty[int | None](None)
    seed = ObservableProperty[int | None](None)     # 빈칸 = 랜덤
    duration = ObservableProperty(60)
    audio_format = ObservableProperty("mp3")
    ab_compare = ObservableProperty(False)

    job = ObservableProperty[Job | None](None)
    job_logs = ObservableProperty[list[str]](factory=list)
    songs = ObservableProperty[list[Song]](factory=list)

    def __init__(self, container=None, lyric_id: str | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        # 화면(브라우저)이 지금 실제로 연결되어 있는지 — View 가 넣어 준다. 연결이 끊긴 사이에 끝난 결과는
        # '아직 못 본 결과'로 남겨 두었다가 화면이 다시 열리면 보여 준다.
        self.is_live: Callable[[], bool] = lambda: True
        self.tracker = JobTracker(self, self._on_job_update, self._on_job_log)
        self.listen(Topics.CATEGORY_CHANGED, lambda e: self.load_models())
        self.listen(Topics.LYRICS_SAVED, lambda e: self.reload_drafts())   # 가사집에서 바뀐 내용 반영
        self.load_models()
        self.reload_drafts()
        # 다른 화면에 다녀와도 실행 중인 작업을 다시 붙이고, 버튼을 누르기 직전 화면 상태를 그대로 되살린다.
        # (끝난 작업은 다시 불러오지 않는다 → 처음 들어온 화면)
        job = self.tracker.resume([JobKind.PROMPT_GENERATE, JobKind.MUSIC_GENERATE], recent_minutes=0)
        if job is not None and not job.status.is_terminal:
            self._restore(job.params.get("ui") or {})
            return
        self._show_unseen_results()        # 화면을 보고 있지 않을 때 끝난 곡이 있으면 결과를 보여 줌 (입력은 처음 상태)
        if lyric_id:                       # 가사집의 [이 가사로 곡 만들기]
            self.select_lyric(lyric_id)

    # --- 조회 ---
    @property
    def job_running(self) -> bool:
        return self.job is not None and not self.job.status.is_terminal

    @property
    def locked(self) -> bool:
        """생성 중에는 화면의 모든 입력을 잠근다 (같은 작업을 두 번 보내 충돌하는 것 방지)."""
        return self.job_running

    def _guard(self) -> bool:
        if self.locked:
            self.error = "음악(또는 프롬프트)을 만드는 중입니다. 끝난 뒤에 바꿀 수 있습니다."
            return True
        return False

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
        if self._guard():
            return
        self.music_filter = category_id or ALL
        self._load_music_models()

    def set_music_model(self, model_id: str | None) -> None:
        if self._guard():
            return
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
        if self._guard():
            return
        self.lyric_filter = category_id or ALL
        self.reload_drafts(self.PAGE)

    def search_drafts(self, query: str | None) -> None:
        if self.locked:
            return
        self.draft_query = query or ""
        self.reload_drafts(self.PAGE)

    def more_drafts(self) -> None:
        self.reload_drafts(len(self.drafts) + self.PAGE)

    def select_lyric(self, lyric_id: str) -> None:
        if self._guard():
            return
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
        if self._guard() or not self._ensure_saved():
            return
        self._submit(JobKind.PROMPT_GENERATE, {"draft_id": self.draft_id, "wishes": wishes or self.wishes,
                                               "ui": self._snapshot()})

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
    async def generate_music(self, options: dict[str, Any] | None = None) -> None:
        if self._guard() or not self._ensure_saved():
            return
        if options:                        # (예전 호출 방식 호환) 화면 옵션을 VM 에 반영
            for key in ("seed", "duration", "audio_format", "ab_compare"):
                if key in options:
                    setattr(self, key, options[key])
        # 음악 프롬프트는 선택 사항 — 비어 있으면 가사와 음악 모델만으로 생성
        prompt = self.prompt_text.strip()
        if prompt and (self.music_prompt is None or self.music_prompt.prompt != prompt) \
                and not self.apply_manual_prompt():
            return
        if self.output_dir.strip() and self.sync_command(
                lambda: self.c.music.resolve_output_dir(self.output_dir)) is None:
            return
        seed = self.seed
        spec = {
            "title": self.song_title.strip() or self.title, "draft_id": self.draft_id, "prompt": prompt,
            "vocal_language": VOCAL_LANGUAGE.get(self.language, "ko"), "bpm": self.bpm,
            "use_lora": self.music_model_id is not None, "model_id": self.music_model_id,
            "output_dir": self.output_dir.strip(),
            "audio_duration": float(self.duration) if self.duration else None,
            "seed": int(seed) if seed not in (None, "") else None,
            "audio_format": self.audio_format or "mp3",
            "prompt_meta": {"source": (self.music_prompt.source if self.music_prompt else "manual") if prompt
                            else "none",
                            "notes": self.music_prompt.notes if (prompt and self.music_prompt) else []},
        }
        ab = bool(self.ab_compare) and self.music_model_id is not None
        self._submit(JobKind.MUSIC_GENERATE, {"spec": spec, "ab_compare": ab, "ui": self._snapshot()})

    async def cancel_job(self) -> None:
        if self.job:
            await self.run_command(lambda: self.c.jobs.cancel(self.job.id))

    # --- 화면 상태 저장·복원·초기화 ---
    _UI_FIELDS = ("draft_id", "lyric_filter", "draft_query", "prompt_text", "wishes", "bpm", "music_filter",
                  "music_model_id", "language", "song_title", "output_dir", "seed", "duration", "audio_format",
                  "ab_compare")

    def _snapshot(self) -> dict[str, Any]:
        """버튼을 누르는 순간의 화면 상태 (작업 기록에 같이 저장 → 다시 들어와도 그대로 보여 줌)."""
        return {k: getattr(self, k) for k in self._UI_FIELDS}

    def _restore(self, ui: dict[str, Any]) -> None:
        for key in ("lyric_filter", "draft_query", "music_filter"):
            if key in ui:
                setattr(self, key, ui[key] or "")
        self.reload_drafts(self.PAGE)
        self._load_music_models()
        draft = self.c.lyrics.get(ui["draft_id"]) if ui.get("draft_id") else None
        if draft:
            self.draft_id, self.title, self.lyrics_text = draft.id, draft.title, draft.lyrics
        for key in ("prompt_text", "wishes", "bpm", "music_model_id", "language", "song_title", "output_dir",
                    "seed", "duration", "audio_format", "ab_compare"):
            if key in ui:
                setattr(self, key, ui[key])

    def reset_form(self) -> None:
        """처음 들어온 화면처럼 전부 초기화 (방금 만든 곡 결과만 남김)."""
        self.clear_lyric()
        self.lyric_filter, self.draft_query = ALL, ""
        self.music_prompt, self.prompt_text, self.wishes, self.bpm = None, "", "", None
        self.music_filter, self.music_model_id = ALL, None
        self.language, self.song_title, self.output_dir = "ko", "", ""
        self.seed, self.duration, self.audio_format, self.ab_compare = None, 60, "mp3", False
        self.reload_drafts(self.PAGE)
        self._load_music_models()

    UNSEEN_MINUTES = 60

    def _show_unseen_results(self) -> None:
        last = next((j for j in self.c.job_repo.list_recent(30) if j.kind == JobKind.MUSIC_GENERATE), None)
        if last is None or last.status != JobStatus.SUCCEEDED or last.id in self.c.seen_results or not last.ended_at:
            return
        ended = last.ended_at
        now = datetime.now(ended.tzinfo) if ended.tzinfo else datetime.now()
        if now - ended > timedelta(minutes=self.UNSEEN_MINUTES):
            return
        songs = [s for sid in (last.result or {}).get("song_ids", []) if (s := self.c.song_repo.get(sid))]
        if not songs:
            return
        self.c.seen_results.add(last.id)
        self.songs = songs
        self.info = f"완성된 곡 {len(songs)}개 — 화면을 보고 있지 않는 동안 완성되었습니다."

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
            songs = [s for sid in job.result.get("song_ids", []) if (s := self.c.song_repo.get(sid))]
            if self.is_live():                 # 지금 화면에 보여 줌 → 다시 들어와도 또 보여 주지 않음
                self.c.seen_results.add(job.id)
            self.reset_form()                  # 다 만들어지면 처음 들어온 화면처럼
            self.songs = songs
            folder = str(Path(songs[0].audio_path).parent) if songs else ""
            self.info = f"음원 {len(songs)}개 생성 완료 → {folder}  (만든 곡에서도 볼 수 있습니다)"
