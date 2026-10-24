"""만든 곡 카테고리 리포지토리 (곡 분류 전용, 모델 카테고리와 별개)."""

from __future__ import annotations

import sqlite3

from pydantic import BaseModel, Field

from app.domain.entities import Song, new_id, utcnow
from app.repositories.base import BaseRepository, iso
from app.repositories.content_repo import SongRepository


class SongCategory(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    song_count: int = 0


class SongCategoryRepository(BaseRepository):
    def _cat(self, r: sqlite3.Row) -> SongCategory:
        return SongCategory(id=r["id"], name=r["name"], song_count=r["song_count"])

    def list(self) -> list[SongCategory]:
        with self.db.read() as conn:
            return [self._cat(r) for r in conn.execute(
                "SELECT c.id, c.name, COUNT(m.song_id) AS song_count FROM song_categories c"
                " LEFT JOIN song_category_members m ON m.category_id=c.id"
                " GROUP BY c.id ORDER BY c.created_at")]

    def get_by_name(self, name: str) -> SongCategory | None:
        return next((c for c in self.list() if c.name == name), None)

    def add(self, name: str) -> SongCategory:
        cat = SongCategory(name=name)
        with self.db.transaction() as conn:
            conn.execute("INSERT INTO song_categories (id, name, created_at) VALUES (?,?,?)",
                         (cat.id, cat.name, iso(utcnow())))
        return cat

    def delete(self, category_id: str) -> None:
        """카테고리 묶음만 지운다. 곡은 남는다."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM song_category_members WHERE category_id=?", (category_id,))
            conn.execute("DELETE FROM song_categories WHERE id=?", (category_id,))

    def add_member(self, category_id: str, song_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO song_category_members (category_id, song_id, added_at)"
                         " VALUES (?,?,?)", (category_id, song_id, iso(utcnow())))

    def remove_member(self, category_id: str, song_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM song_category_members WHERE category_id=? AND song_id=?",
                         (category_id, song_id))

    def categories_of(self) -> dict[str, list[str]]:
        """{song_id: [카테고리 id, ...]} — 목록 화면에서 배지 표시용 (한 번에 조회)."""
        result: dict[str, list[str]] = {}
        with self.db.read() as conn:
            for r in conn.execute("SELECT song_id, category_id FROM song_category_members"):
                result.setdefault(r["song_id"], []).append(r["category_id"])
        return result

    def songs_in(self, songs: SongRepository, category_id: str, query: str = "", limit: int = 1000) -> list[Song]:
        sql = ("SELECT s.* FROM song_category_members m JOIN songs s ON s.id=m.song_id"
               " WHERE m.category_id=?")
        args: list[object] = [category_id]
        if query:
            sql += " AND s.title LIKE ?"
            args.append(f"%{query}%")
        sql += " ORDER BY s.created_at DESC LIMIT ?"
        args.append(limit)
        with self.db.read() as conn:
            return [songs._row(r) for r in conn.execute(sql, args)]

    def delete_song(self, song_id: str) -> None:
        """곡 기록 삭제 (카테고리 소속·앨범 트랙 포함)."""
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM song_category_members WHERE song_id=?", (song_id,))
            conn.execute("DELETE FROM album_tracks WHERE song_id=?", (song_id,))
            conn.execute("DELETE FROM songs WHERE id=?", (song_id,))
