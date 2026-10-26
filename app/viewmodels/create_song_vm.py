"""곡 만들기 VM: 가사 모델·음악 모델 선택 → 작사 → 편집/검수 → 음악 프롬프트 → 음악 생성 → 재생.

카테고리를 고르지 않는다. 사용자가 직접 고른
- 가사 모델 (Qwen 기본 또는 학습한 가사 LoRA) 로 가사를 만들고
- 음악 모델 (ACE 기본 또는 학습한 음악 LoRA) 로 그 가사를 ACE-Step 에 보내 곡을 만든다.
음악 모델 목록은 [음악 LoRA 학습 → 모델·카테고리] 에서 나눈 카테고리로 걸러 볼 수 있다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.core.observer import ObservableProperty, Topics
from app.domain.entities import Category, Job, LyricsDraft, ModelRecord, Song
from app.domain.enums import Engine, JobKind, JobStatus
from app.services.lyrics_service import LYRICS_LANGUAGES, VOCAL_LANGUAGE
from app.services.prompt_service import MusicPrompt, free_category
from app.viewmodels.base import BaseViewModel, JobTracker

ALL = ""   # '전체' 카테고리 필터 값


class CreateSongViewModel(BaseViewModel):
    LANGUAGES = LYRICS_LANGUAGES   # {"ko": "한국어", "en": "영어", "ja": "일본어"}

    # --- 모델 선택 ---
    lyrics_filter = ObservableProperty(ALL)                                # 가사 모델 카테고리 필터
    lyrics_models = ObservableProperty[list[ModelRecord]](factory=list)   # 필터된 가사 LoRA 목록
    lyrics_model_id = ObservableProperty[str | None](None)                # None = Qwen 기본
    categories = ObservableProperty[list[Category]](factory=list)          # 음악 모델 분류
    music_filter = ObservableProperty(ALL)                                 # 음악 모델 카테고리 필터
    music_models = ObservableProperty[list[ModelRecord]](factory=list)     # 필터된 음악 LoRA 목록
    music_model_id = ObservableProperty[str | None](None)                  # None = ACE 기본
    output_dir = ObservableProperty("")                                    # 결과 저장 폴더
    language = ObservableProperty("ko")                                    # 가사 언어 = 노래(보컬) 언어
    song_title = ObservableProperty("")                                    # 곡 제목 (비우면 가사 제목)

    # --- 가사 ---
    drafts = ObservableProperty[list[LyricsDraft]](factory=list)
    draft_id = ObservableProperty[str | None](None)
    title = ObservableProperty("")
    lyrics_text = ObservableProperty("")
    history = ObservableProperty[list[LyricsDraft]](factory=list)

    # --- 음악 ---
    music_prompt = ObservableProperty[MusicPrompt | None](None)
    prompt_text = ObservableProperty("")
    bpm = ObservableProperty[int | None](None)

    job = ObservableProperty[Job | None](None)
    job_logs = ObservableProperty[list[str]](factory=list)
    songs = ObservableProperty[list[Song]](factory=list)
    step = ObservableProperty(1)

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.tracker = JobTracker(self, self._on_job_update, self._on_job_log)
        self.listen(Topics.CATEGORY_CHANGED, lambda e: self.load_models())
        self.load_models()
        self.drafts = self.c.lyrics.list_recent()
        # 다른 화면에 다녀와도 실행 중인 작사/음악 생성 작업을 다시 표시 (끝난 작업은 다시 불러오지 않음)
        self.tracker.resume([JobKind.LYRICS_GENERATE, JobKind.PROMPT_GENERATE, JobKind.MUSIC_GENERATE],
                            recent_minutes=0)

    # --- 조회 ---
    @property
    def job_running(self) -> bool:
        return self.job is not None and not self.job.status.is_terminal

    @property
    def lyrics_model(self) -> ModelRecord | None:
        return next((m for m in self.lyrics_models if m.id == self.lyrics_model_id), None)

    @property
    def music_model(self) -> ModelRecord | None:
        if not self.music_model_id:
            return None
        return self.c.categories.models.get(self.music_model_id)

    @property
    def selection_summary(self) -> str:
        lm = self.lyrics_model.label if self.lyrics_model else "Qwen 기본"
        mm = self.music_model.label if self.music_model else "ACE 기본 (LoRA 없음)"
        return f"가사: {lm}  →  음악: {mm}"

    # --- 모델 선택 ---
    def load_models(self) -> None:
        self.categories = self.c.categories.list()
        ids = {c.id for c in self.categories}
        if self.music_filter and self.music_filter not in ids:
            self.music_filter = ALL
        if self.lyrics_filter and self.lyrics_filter not in ids:
            self.lyrics_filter = ALL
        self._load_lyrics_models()
        self._load_music_models()

    def _load_lyrics_models(self) -> None:
        if self.lyrics_filter:
            self.lyrics_models = self.c.categories.category_models(self.lyrics_filter, Engine.QWEN)
        else:
            self.lyrics_models = self.c.categories.list_models(engine=Engine.QWEN)
        if self.lyrics_model_id not in {m.id for m in self.lyrics_models}:
            active = (self.c.categories.active_model(self.lyrics_filter, Engine.QWEN)
                      if self.lyrics_filter else None)
            self.lyrics_model_id = active.id if active else None

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

    def set_lyrics_filter(self, category_id: str | None) -> None:
        self.lyrics_filter = category_id or ALL
        self._load_lyrics_models()

    def set_lyrics_model(self, model_id: str | None) -> None:
        self.lyrics_model_id = model_id or None

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

    # --- 1) 작사 ---
    async def generate_lyrics(self, form: dict[str, Any]) -> None:
        params = {
            "model_id": self.lyrics_model_id, "topic": form.get("topic", "").strip(), "language": self.language,
            "speaker": form.get("speaker", ""), "mood": form.get("mood", ""), "viewpoint": form.get("viewpoint", ""),
            "forbidden": [t.strip() for t in form.get("forbidden", "").split(",") if t.strip()],
            "variants": int(form.get("variants") or 1),
            "syllable_min": form.get("syllable_min") or None, "syllable_max": form.get("syllable_max") or None,
        }
        if form.get("structure"):
            params["structure"] = [s.strip() for s in form["structure"].split(",") if s.strip()]
        self._submit(JobKind.LYRICS_GENERATE, params)

    def load_draft(self, draft_id: str) -> None:
        draft = self.sync_command(lambda: self.c.lyrics.get(draft_id))
        if not draft:
            return
        self.draft_id = draft.id
        self.title = draft.title
        self.lyrics_text = draft.lyrics
        self.history = self.c.lyrics.history(draft.id)
        self.step = max(self.step, 2)

    # --- 2) 가사 편집 ---
    def _auto_title(self) -> str:
        """직접 쓴 가사는 제목 입력칸이 없으므로 첫 가사 줄([Verse] 같은 표시 제외)로 이름을 붙인다."""
        for line in self.lyrics_text.splitlines():
            text = line.strip()
            if text and not (text.startswith("[") and text.endswith("]")):
                return text[:30]
        return "무제"

    def save_lyrics(self) -> None:
        if not self.title.strip():
            self.title = self._auto_title()
        if self.draft_id is None:
            draft = self.sync_command(lambda: self.c.lyrics.create_manual(
                None, self.title or "무제", self.lyrics_text), success="가사를 저장했습니다.")
        else:
            draft = self.sync_command(lambda: self.c.lyrics.save_edit(self.draft_id, self.lyrics_text, self.title),
                                      success="새 버전으로 저장했습니다. (원본 보존)")
        if draft:
            self.draft_id = draft.id
            self.history = self.c.lyrics.history(draft.id)
            self.drafts = self.c.lyrics.list_recent()

    # --- 3) 음악 프롬프트 ---
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

    # --- 4) 음악 생성 ---
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
        if not self.lyrics_text.strip():
            self.error = "가사가 비어 있습니다."
            return False
        current = self.c.lyrics.get(self.draft_id) if self.draft_id else None
        if current is None or current.lyrics.strip() != self.lyrics_text.strip() or current.title != self.title:
            self.save_lyrics()
        return self.draft_id is not None and not self.error

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
        if job.kind == JobKind.LYRICS_GENERATE:
            self.drafts = self.c.lyrics.list_recent()
            ids = job.result.get("draft_ids", [])
            if ids:
                self.load_draft(ids[0])
            self.info = f"가사 {len(ids)}개를 생성했습니다."
        elif job.kind == JobKind.PROMPT_GENERATE:
            mp = MusicPrompt(**job.result["music_prompt"])
            self.music_prompt = mp
            self.prompt_text = mp.prompt
            self.bpm = mp.bpm
            self.step = max(self.step, 3)
            self.info = f"음악 프롬프트 생성 완료 (출처: {mp.source})"
        elif job.kind == JobKind.MUSIC_GENERATE:
            self.songs = [s for sid in job.result.get("song_ids", []) if (s := self.c.song_repo.get(sid))]
            self.step = 4
            folder = str(Path(self.songs[0].audio_path).parent) if self.songs else ""
            self.info = f"음원 {len(self.songs)}개 생성 완료 → {folder}"
