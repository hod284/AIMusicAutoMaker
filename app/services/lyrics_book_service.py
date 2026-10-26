"""가사집 서비스: 가사 한 개 = 기록 한 개. 목록·검색·덮어쓰기 저장·삭제·카테고리 분류."""

from __future__ import annotations

from app.core.errors import NotFoundError, ValidationError
from app.core.observer import EventBus, Topics
from app.domain.entities import LyricsDraft
from app.domain.enums import LyricsSource
from app.repositories.content_repo import LyricsRepository
from app.repositories.lyric_category_repo import LyricCategory, LyricCategoryRepository


class LyricsBookService:
    def __init__(self, lyrics: LyricsRepository, categories: LyricCategoryRepository,
                 bus: EventBus | None = None) -> None:
        self.lyrics = lyrics
        self.cats = categories
        self.bus = bus or EventBus()

    def _changed(self, **info: object) -> None:
        self.bus.publish(Topics.LYRICS_SAVED, **info)   # 가사 목록을 보는 화면들이 새로고침

    # --- 가사 ---
    def list(self, category_id: str | None = None, query: str = "", limit: int = 50) -> tuple[list[LyricsDraft], int]:
        return (self.lyrics.list_latest(query, limit, category_id or None),
                self.lyrics.count_latest(query, category_id or None))

    def get(self, lyric_id: str) -> LyricsDraft:
        d = self.lyrics.get(lyric_id)
        if d is None:
            raise NotFoundError("가사를 찾을 수 없습니다.", detail=lyric_id)
        return d

    def create(self, title: str, lyrics: str) -> LyricsDraft:
        if not lyrics.strip():
            raise ValidationError("가사가 비어 있습니다.")
        d = LyricsDraft(title=title.strip() or _first_line(lyrics), lyrics=lyrics.strip(),
                        source=LyricsSource.USER_ORIGINAL)
        self.lyrics.add(d)
        self._changed(action="created", lyric_id=d.id)
        return d

    def save(self, lyric_id: str, title: str, lyrics: str) -> LyricsDraft:
        """덮어쓰기 저장. 가사 하나는 항상 기록 하나."""
        d = self.get(lyric_id)
        if not lyrics.strip():
            raise ValidationError("가사가 비어 있습니다.")
        self.lyrics.update_content(d.id, title.strip() or _first_line(lyrics), lyrics.strip())
        self._changed(action="saved", lyric_id=d.id)
        return self.get(d.id)

    def delete(self, lyric_id: str) -> None:
        d = self.get(lyric_id)
        self.lyrics.delete_lineage(d.lineage_id)
        self._changed(action="deleted", lyric_id=lyric_id)

    # --- 카테고리 ---
    def categories(self) -> list[LyricCategory]:
        return self.cats.list()

    def create_category(self, name: str) -> LyricCategory:
        name = name.strip()
        if not name:
            raise ValidationError("카테고리 이름을 입력하세요.")
        if len(name) > 40:
            raise ValidationError("카테고리 이름은 40자 이하로 입력하세요.")
        if any(c.name == name for c in self.cats.list()):
            raise ValidationError(f"'{name}' 카테고리가 이미 있습니다.")
        cat = self.cats.add(name)
        self._changed(action="category_created")
        return cat

    def delete_category(self, category_id: str) -> None:
        self.cats.delete(category_id)
        self._changed(action="category_deleted")

    def add_to_category(self, category_id: str, lyric_id: str) -> None:
        if not any(c.id == category_id for c in self.cats.list()):
            raise NotFoundError("카테고리를 찾을 수 없습니다.")
        self.cats.add_member(category_id, lyric_id)
        self._changed(action="member_added")

    def remove_from_category(self, category_id: str, lyric_id: str) -> None:
        self.cats.remove_member(category_id, lyric_id)
        self._changed(action="member_removed")

    def set_categories(self, lyric_id: str, category_ids: list[str]) -> None:
        self.get(lyric_id)
        valid = {c.id for c in self.cats.list()}
        self.cats.set_members_of(lyric_id, [c for c in category_ids if c in valid])
        self._changed(action="members_set", lyric_id=lyric_id)

    def membership(self) -> dict[str, list[str]]:
        return self.cats.membership()


def _first_line(lyrics: str) -> str:
    """제목이 없으면 [Verse] 표시를 뺀 첫 가사 줄로."""
    for line in lyrics.splitlines():
        t = line.strip()
        if t and not (t.startswith("[") and t.endswith("]")):
            return t[:30]
    return "무제"
