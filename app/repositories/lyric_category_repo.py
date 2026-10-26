"""가사집 카테고리 리포지토리 (가사 분류 전용, 곡·모델 카테고리와 별개)."""

from __future__ import annotations

import sqlite3

from pydantic import BaseModel, Field

from app.domain.entities import new_id, utcnow
from app.repositories.base import BaseRepository, iso


class LyricCategory(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    count: int = 0


class LyricCategoryRepository(BaseRepository):
    def _cat(self, r: sqlite3.Row) -> LyricCategory:
        return LyricCategory(id=r["id"], name=r["name"], count=r["n"])

    def list(self) -> list[LyricCategory]:
        with self.db.read() as conn:
            return [self._cat(r) for r in conn.execute(
                "SELECT c.id, c.name, COUNT(m.lyric_id) AS n FROM lyric_categories c"
                " LEFT JOIN lyric_category_members m ON m.category_id=c.id"
                " GROUP BY c.id ORDER BY c.created_at")]

    def add(self, name: str) -> LyricCategory:
        cat = LyricCategory(name=name)
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO lyric_categories (id, name, created_at) VALUES (?,?,?)",
                         (cat.id, cat.name, iso(utcnow())))
        return cat

    def delete(self, category_id: str) -> None:
        """카테고리 묶음만 지운다. 가사는 남는다."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM lyric_category_members WHERE category_id=?", (category_id,))
            conn.execute("DELETE FROM lyric_categories WHERE id=?", (category_id,))

    def add_member(self, category_id: str, lyric_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO lyric_category_members (category_id, lyric_id, added_at)"
                         " VALUES (?,?,?)", (category_id, lyric_id, iso(utcnow())))

    def remove_member(self, category_id: str, lyric_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM lyric_category_members WHERE category_id=? AND lyric_id=?",
                         (category_id, lyric_id))

    def set_members_of(self, lyric_id: str, category_ids: list[str]) -> None:
        """한 가사가 속한 카테고리를 통째로 바꾼다 (에디터의 카테고리 선택)."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM lyric_category_members WHERE lyric_id=?", (lyric_id,))
            for cid in dict.fromkeys(category_ids):
                conn.execute("INSERT OR IGNORE INTO lyric_category_members (category_id, lyric_id, added_at)"
                             " VALUES (?,?,?)", (cid, lyric_id, iso(utcnow())))

    def membership(self) -> dict[str, list[str]]:
        """{lyric_id: [카테고리 id, ...]}"""
        result: dict[str, list[str]] = {}
        with self.db.read() as conn:
            for r in conn.execute("SELECT lyric_id, category_id FROM lyric_category_members"):
                result.setdefault(r["lyric_id"], []).append(r["category_id"])
        return result
