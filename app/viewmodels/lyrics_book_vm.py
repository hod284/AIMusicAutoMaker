"""가사집 ViewModel.

- LyricsBookViewModel  : [가사 목록] 탭 — 카테고리 필터(기본 '전체')·검색·목록 + 에디터(덮어쓰기 저장) + AI 작사
- LyricCategoriesViewModel : [카테고리 편집] 탭 — 카테고리 만들기/삭제, 가사 넣기/빼기, 가사 삭제
가사 하나 = DB 기록 하나. 저장하면 같은 기록에 덮어쓴다 (v1/v2 를 만들지 않음).
"""

from __future__ import annotations

from typing import Any

from app.core.observer import ObservableProperty, Topics
from app.domain.entities import Job, LyricsDraft, ModelRecord
from app.domain.enums import Engine, JobKind, JobStatus
from app.repositories.lyric_category_repo import LyricCategory
from app.services.lyrics_service import LYRICS_LANGUAGES
from app.viewmodels.base import BaseViewModel, JobTracker

ALL = ""


class LyricsBookViewModel(BaseViewModel):
    LANGUAGES = LYRICS_LANGUAGES
    PAGE = 50

    # --- 목록 ---
    categories = ObservableProperty[list[LyricCategory]](factory=list)
    category_filter = ObservableProperty(ALL)                 # 들어올 때는 항상 '전체'
    query = ObservableProperty("")
    items = ObservableProperty[list[LyricsDraft]](factory=list)
    total = ObservableProperty(0)
    membership = ObservableProperty[dict[str, list[str]]](factory=dict)

    # --- 에디터 ---
    selected_id = ObservableProperty[str | None](None)        # None = 새 가사 (아직 저장 안 됨)
    title = ObservableProperty("")
    text = ObservableProperty("")
    selected_categories = ObservableProperty[list[str]](factory=list)
    editor_open = ObservableProperty(False)

    # --- AI 작사 ---
    language = ObservableProperty("ko")
    lyrics_models = ObservableProperty[list[ModelRecord]](factory=list)
    lyrics_model_id = ObservableProperty[str | None](None)
    job = ObservableProperty[Job | None](None)
    job_logs = ObservableProperty[list[str]](factory=list)

    def __init__(self, container=None, open_id: str | None = None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.svc = self.c.lyrics_book
        self.tracker = JobTracker(self, self._on_job, self._on_log)
        self.listen(Topics.LYRICS_SAVED, lambda e: self.reload())
        self.listen(Topics.CATEGORY_CHANGED, lambda e: self._load_models())
        self._load_models()
        self.reload(self.PAGE)
        self.tracker.resume([JobKind.LYRICS_GENERATE], recent_minutes=0)
        if open_id:
            self.open(open_id)

    # --- 조회 ---
    @property
    def job_running(self) -> bool:
        return self.job is not None and not self.job.status.is_terminal

    @property
    def dirty(self) -> bool:
        """에디터 내용이 저장된 것과 다른지."""
        if not self.editor_open:
            return False
        if self.selected_id is None:
            return bool(self.text.strip())
        saved = next((d for d in self.items if d.id == self.selected_id), None)
        return saved is None or saved.lyrics.strip() != self.text.strip() or saved.title != self.title

    def category_names(self, lyric_id: str) -> list[str]:
        names = {c.id: c.name for c in self.categories}
        return [names[c] for c in self.membership.get(lyric_id, []) if c in names]

    # --- 목록 ---
    def reload(self, limit: int | None = None) -> None:
        self.categories = self.svc.categories()
        if self.category_filter and self.category_filter not in {c.id for c in self.categories}:
            self.category_filter = ALL
        self.membership = self.svc.membership()
        limit = limit or max(self.PAGE, len(self.items))
        self.items, self.total = self.svc.list(self.category_filter or None, self.query, limit)

    def set_filter(self, category_id: str | None) -> None:
        self.category_filter = category_id or ALL
        self.reload(self.PAGE)

    def set_query(self, query: str | None) -> None:
        self.query = query or ""
        self.reload(self.PAGE)

    def more(self) -> None:
        self.reload(len(self.items) + self.PAGE)

    # --- 에디터 ---
    def open(self, lyric_id: str) -> None:
        d = self.sync_command(lambda: self.svc.get(lyric_id))
        if d is None:
            return
        self.selected_id = d.id
        self.title = d.title
        self.text = d.lyrics
        self.selected_categories = list(self.svc.membership().get(d.id, []))
        self.editor_open = True

    def new(self) -> None:
        self.selected_id = None
        self.title = ""
        self.text = ""
        self.selected_categories = [self.category_filter] if self.category_filter else []
        self.editor_open = True

    def close_editor(self) -> None:
        self.editor_open = False
        self.selected_id = None

    def save(self) -> None:
        if self.selected_id is None:
            d = self.sync_command(lambda: self.svc.create(self.title, self.text))
        else:
            d = self.sync_command(lambda: self.svc.save(self.selected_id, self.title, self.text))
        if d is None:
            return
        self.sync_command(lambda: self.svc.set_categories(d.id, self.selected_categories))
        self.selected_id, self.title, self.text = d.id, d.title, d.lyrics
        self.reload()
        if not self.error:
            self.info = f"'{d.title}' 저장했습니다."

    def delete(self) -> None:
        if self.selected_id is None:
            self.close_editor()
            return
        title = self.title
        self.sync_command(lambda: self.svc.delete(self.selected_id))
        if not self.error:
            self.close_editor()
            self.reload()
            self.info = f"'{title}' 가사를 삭제했습니다."

    # --- AI 작사 ---
    def _load_models(self) -> None:
        self.lyrics_models = self.c.categories.list_models(engine=Engine.QWEN)
        if self.lyrics_model_id not in {m.id for m in self.lyrics_models}:
            self.lyrics_model_id = None

    def generate(self, form: dict[str, Any]) -> None:
        if self.job_running:
            self.error = "AI 작사가 아직 진행 중입니다."
            return
        params: dict[str, Any] = {
            "model_id": self.lyrics_model_id, "language": self.language,
            "topic": (form.get("topic") or "").strip(), "speaker": form.get("speaker", ""),
            "mood": form.get("mood", ""), "viewpoint": form.get("viewpoint", ""),
        }
        if form.get("structure"):
            params["structure"] = [s.strip() for s in str(form["structure"]).split(",") if s.strip()]
        if not params["topic"]:
            self.error = "주제를 입력하세요."
            return
        self.job_logs = []
        job = self.sync_command(lambda: self.c.jobs.submit(JobKind.LYRICS_GENERATE, params))
        if job:
            self.tracker.track(job)

    async def cancel_job(self) -> None:
        if self.job:
            await self.run_command(lambda: self.c.jobs.cancel(self.job.id))

    def _on_log(self, line: str) -> None:
        self.job_logs = (self.job_logs + [line])[-100:]

    def _on_job(self, job: Job) -> None:
        self.job = job
        if job.status == JobStatus.FAILED:
            self.error = job.error or "AI 작사가 실패했습니다."
        if job.status == JobStatus.SUCCEEDED:
            ids = job.result.get("draft_ids", [])
            self.reload()
            if ids:
                self.open(ids[0])
            self.info = "AI 가사를 만들어 가사집에 저장했습니다. 고친 뒤 [저장]을 누르세요."


class LyricCategoriesViewModel(BaseViewModel):
    """[카테고리 편집] 탭: 왼쪽 카테고리(만들기/삭제/안의 가사), 오른쪽 전체 가사(넣기/빼기/삭제)."""

    categories = ObservableProperty[list[LyricCategory]](factory=list)
    category_id = ObservableProperty[str | None](None)
    members = ObservableProperty[list[LyricsDraft]](factory=list)
    all_items = ObservableProperty[list[LyricsDraft]](factory=list)
    total = ObservableProperty(0)
    query = ObservableProperty("")

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.svc = self.c.lyrics_book
        self.listen(Topics.LYRICS_SAVED, lambda e: self.load())
        self.load()

    @property
    def member_ids(self) -> set[str]:
        return {d.id for d in self.members}

    @property
    def category_name(self) -> str:
        return next((c.name for c in self.categories if c.id == self.category_id), "")

    def load(self) -> None:
        self.categories = self.svc.categories()
        if self.category_id not in {c.id for c in self.categories}:
            self.category_id = self.categories[0].id if self.categories else None
        self.members = self.svc.list(self.category_id, "", 1000)[0] if self.category_id else []
        self.all_items, self.total = self.svc.list(None, self.query, 500)

    def select(self, category_id: str | None) -> None:
        self.category_id = category_id
        self.load()

    def set_query(self, query: str | None) -> None:
        self.query = query or ""
        self.load()

    def create(self, name: str) -> None:
        cat = self.sync_command(lambda: self.svc.create_category(name), success=f"카테고리 '{name.strip()}' 를 만들었습니다.")
        if cat:
            self.category_id = cat.id
            self.load()

    def delete_category(self) -> None:
        if not self.category_id:
            return
        name = self.category_name
        self.sync_command(lambda: self.svc.delete_category(self.category_id))
        if not self.error:
            self.info = f"카테고리 '{name}' 를 삭제했습니다. (가사는 그대로 남습니다)"
            self.category_id = None
            self.load()

    def add(self, lyric_id: str) -> None:
        if not self.category_id:
            self.error = "왼쪽에서 카테고리를 먼저 만들거나 고르세요."
            return
        self.sync_command(lambda: self.svc.add_to_category(self.category_id, lyric_id), success="카테고리에 넣었습니다.")

    def remove(self, lyric_id: str) -> None:
        if self.category_id:
            self.sync_command(lambda: self.svc.remove_from_category(self.category_id, lyric_id),
                              success="카테고리에서 뺐습니다. (가사는 그대로)")

    def delete_lyric(self, lyric_id: str) -> None:
        title = next((d.title for d in self.all_items if d.id == lyric_id), "")
        self.sync_command(lambda: self.svc.delete(lyric_id))
        if not self.error:
            self.info = f"'{title}' 가사를 삭제했습니다."
