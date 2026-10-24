"""가사 초안 / 곡 / 데이터셋 리포지토리."""

from __future__ import annotations

import sqlite3

from app.domain.entities import Dataset, LyricsDraft, Song
from app.domain.enums import DatasetKind, LyricsSource
from app.repositories.base import BaseRepository, dumps, iso, loads, parse_dt


class LyricsRepository(BaseRepository):
    def _row(self, r: sqlite3.Row) -> LyricsDraft:
        return LyricsDraft(
            id=r["id"], category_id=r["category_id"], lineage_id=r["lineage_id"], title=r["title"],
            topic=r["topic"], lyrics=r["lyrics"], version=r["version"], source=LyricsSource(r["source"]),
            model_id=r["model_id"], prompt=loads(r["prompt_json"], {}), created_at=parse_dt(r["created_at"]),
        )

    def add(self, d: LyricsDraft) -> LyricsDraft:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO lyrics_drafts (id, category_id, lineage_id, title, topic, lyrics, version, source,"
                " model_id, prompt_json, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (d.id, d.category_id, d.lineage_id, d.title, d.topic, d.lyrics, d.version, d.source.value,
                 d.model_id, dumps(d.prompt), iso(d.created_at)),
            )
        return d

    def get(self, draft_id: str) -> LyricsDraft | None:
        with self.db.read() as conn:
            r = conn.execute("SELECT * FROM lyrics_drafts WHERE id=?", (draft_id,)).fetchone()
        return self._row(r) if r else None

    def latest_version(self, lineage_id: str) -> int:
        with self.db.read() as conn:
            r = conn.execute(
                "SELECT COALESCE(MAX(version),0) FROM lyrics_drafts WHERE lineage_id=?", (lineage_id,)
            ).fetchone()
        return int(r[0])

    def history(self, lineage_id: str) -> list[LyricsDraft]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT * FROM lyrics_drafts WHERE lineage_id=? ORDER BY version", (lineage_id,)
            )
            return [self._row(r) for r in rows]

    def list_recent(self, limit: int = 50) -> list[LyricsDraft]:
        with self.db.read() as conn:
            return [self._row(r) for r in conn.execute(
                "SELECT * FROM lyrics_drafts ORDER BY created_at DESC LIMIT ?", (limit,))]

    def list_for_category(self, category_id: str, limit: int = 100) -> list[LyricsDraft]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT * FROM lyrics_drafts WHERE category_id=? ORDER BY created_at DESC LIMIT ?",
                (category_id, limit),
            )
            return [self._row(r) for r in rows]


class SongRepository(BaseRepository):
    def _row(self, r: sqlite3.Row) -> Song:
        return Song(
            id=r["id"], category_id=r["category_id"], lyrics_draft_id=r["lyrics_draft_id"],
            music_model_id=r["music_model_id"], qwen_model_id=r["qwen_model_id"], title=r["title"],
            audio_path=r["audio_path"], prompt=r["prompt"], generation=loads(r["generation_json"], {}),
            status=r["status"], created_at=parse_dt(r["created_at"]),
        )

    def add(self, s: Song) -> Song:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO songs (id, category_id, lyrics_draft_id, music_model_id, qwen_model_id, title,"
                " audio_path, prompt, generation_json, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (s.id, s.category_id, s.lyrics_draft_id, s.music_model_id, s.qwen_model_id, s.title,
                 s.audio_path, s.prompt, dumps(s.generation), s.status, iso(s.created_at)),
            )
        return s

    def save(self, s: Song) -> Song:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE songs SET title=?, audio_path=?, prompt=?, generation_json=?, status=? WHERE id=?",
                (s.title, s.audio_path, s.prompt, dumps(s.generation), s.status, s.id),
            )
        return s

    def get(self, song_id: str) -> Song | None:
        with self.db.read() as conn:
            r = conn.execute("SELECT * FROM songs WHERE id=?", (song_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, category_id: str | None = None, query: str = "", limit: int = 200, *,
             title_only: bool = False) -> list[Song]:
        sql, args = "SELECT * FROM songs WHERE 1=1", []
        if category_id:
            sql += " AND category_id=?"
            args.append(category_id)
        if query and title_only:
            sql += " AND title LIKE ?"
            args.append(f"%{query}%")
        elif query:
            sql += " AND (title LIKE ? OR prompt LIKE ?)"
            args += [f"%{query}%", f"%{query}%"]
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self.db.read() as conn:
            return [self._row(r) for r in conn.execute(sql, args)]


class DatasetRepository(BaseRepository):
    def _row(self, r: sqlite3.Row) -> Dataset:
        return Dataset(
            id=r["id"], category_id=r["category_id"], kind=DatasetKind(r["kind"]), version=r["version"],
            status=r["status"], source_path=r["source_path"], json_path=r["json_path"],
            tensor_path=r["tensor_path"], rights_status=r["rights_status"], created_at=parse_dt(r["created_at"]),
        )

    def next_version(self, category_id: str | None, kind: DatasetKind) -> int:
        with self.db.read() as conn:
            r = conn.execute(
                "SELECT COALESCE(MAX(version),0) FROM datasets WHERE category_id IS ? AND kind=?",
                (category_id, kind.value),
            ).fetchone()
        return int(r[0]) + 1

    def add(self, d: Dataset) -> Dataset:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO datasets (id, category_id, kind, version, status, source_path, json_path,"
                " tensor_path, rights_status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (d.id, d.category_id, d.kind.value, d.version, d.status, d.source_path, d.json_path,
                 d.tensor_path, d.rights_status, iso(d.created_at)),
            )
        return d

    def save(self, d: Dataset) -> Dataset:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE datasets SET status=?, source_path=?, json_path=?, tensor_path=?, rights_status=? WHERE id=?",
                (d.status, d.source_path, d.json_path, d.tensor_path, d.rights_status, d.id),
            )
        return d

    def get(self, dataset_id: str) -> Dataset | None:
        with self.db.read() as conn:
            r = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return self._row(r) if r else None

    def list_kind(self, kind: DatasetKind) -> list[Dataset]:
        with self.db.read() as conn:
            return [self._row(r) for r in conn.execute(
                "SELECT * FROM datasets WHERE kind=? ORDER BY created_at", (kind.value,))]

    def list(self, category_id: str | None, kind: DatasetKind | None = None) -> list[Dataset]:
        sql, args = "SELECT * FROM datasets WHERE category_id IS ?", [category_id]
        if kind:
            sql += " AND kind=?"
            args.append(kind.value)
        with self.db.read() as conn:
            return [self._row(r) for r in conn.execute(sql + " ORDER BY version", args)]
