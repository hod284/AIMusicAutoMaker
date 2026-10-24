from __future__ import annotations

import sqlite3

from app.domain.entities import Category, StyleConstraints, utcnow
from app.domain.enums import CategoryStatus
from app.repositories.base import BaseRepository, dumps, iso, loads, parse_dt

_ACTIVE_JOB_STATUSES = ("created", "validating", "queued", "running", "exporting", "validating_adapter")


class CategoryRepository(BaseRepository):
    def _row(self, r: sqlite3.Row) -> Category:
        return Category(
            id=r["id"],
            slug=r["slug"],
            name=r["name"],
            description=r["description"],
            language=r["language"],
            tags=loads(r["tags_json"], []),
            status=CategoryStatus(r["status"]),
            style=StyleConstraints(**loads(r["style_constraints_json"], {})),
            created_at=parse_dt(r["created_at"]),
            updated_at=parse_dt(r["updated_at"]),
        )

    def add(self, c: Category) -> Category:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO categories (id, slug, name, description, language, tags_json, status,"
                " style_constraints_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (c.id, c.slug, c.name, c.description, c.language, dumps(c.tags), c.status.value,
                 c.style.model_dump_json(), iso(c.created_at), iso(c.updated_at)),
            )
        return c

    def update(self, c: Category) -> Category:
        c.updated_at = utcnow()
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE categories SET name=?, description=?, language=?, tags_json=?, status=?,"
                " style_constraints_json=?, updated_at=? WHERE id=?",
                (c.name, c.description, c.language, dumps(c.tags), c.status.value,
                 c.style.model_dump_json(), iso(c.updated_at), c.id),
            )
        return c

    def get(self, category_id: str) -> Category | None:
        with self.db.read() as conn:
            r = conn.execute("SELECT * FROM categories WHERE id=?", (category_id,)).fetchone()
        return self._row(r) if r else None

    def get_by_slug(self, slug: str) -> Category | None:
        with self.db.read() as conn:
            r = conn.execute("SELECT * FROM categories WHERE slug=?", (slug,)).fetchone()
        return self._row(r) if r else None

    def list(self) -> list[Category]:
        """전체 카테고리 (보관 기능 제거 → 상태 구분 없이 모두 반환)."""
        with self.db.read() as conn:
            return [self._row(r) for r in conn.execute("SELECT * FROM categories ORDER BY created_at")]

    def reference_counts(self, category_id: str) -> dict[str, int]:
        """삭제 확인 창에 보여줄 참조 개수."""
        counts: dict[str, int] = {}
        with self.db.read() as conn:
            for table in ("datasets", "models", "lyrics_drafts", "songs", "jobs"):
                counts[table] = conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE category_id=?", (category_id,)  # noqa: S608 - 고정 테이블명
                ).fetchone()[0]
        return counts

    def active_job_count(self, category_id: str) -> int:
        q = ",".join("?" * len(_ACTIVE_JOB_STATUSES))
        with self.db.read() as conn:
            return conn.execute(
                f"SELECT COUNT(*) FROM jobs WHERE category_id=? AND status IN ({q})",  # noqa: S608
                (category_id, *_ACTIVE_JOB_STATUSES),
            ).fetchone()[0]

    def delete_grouping(self, category_id: str) -> dict[str, int]:
        """카테고리 '묶음'만 삭제. 모델·곡·가사·작업 기록은 카테고리 없음(NULL)으로 남기고 파일은 건드리지 않는다."""
        cid = (category_id,)
        in_ds = "(SELECT id FROM datasets WHERE category_id=?)"
        steps = [
            ("models_dataset_detached", f"UPDATE models SET dataset_id=NULL WHERE dataset_id IN {in_ds}"),
            ("jobs_dataset_detached", f"UPDATE jobs SET dataset_id=NULL WHERE dataset_id IN {in_ds}"),
            ("models", "UPDATE models SET category_id=NULL WHERE category_id=?"),
            ("jobs", "UPDATE jobs SET category_id=NULL WHERE category_id=?"),
            ("songs", "UPDATE songs SET category_id=NULL WHERE category_id=?"),
            ("lyrics_drafts", "UPDATE lyrics_drafts SET category_id=NULL WHERE category_id=?"),
            ("dataset_items", f"DELETE FROM dataset_items WHERE dataset_id IN {in_ds}"),
            ("datasets", "DELETE FROM datasets WHERE category_id=?"),
            ("bindings", "DELETE FROM category_model_bindings WHERE category_id=?"),
            ("members", "DELETE FROM category_model_members WHERE category_id=?"),
            ("categories", "DELETE FROM categories WHERE id=?"),
        ]
        counts: dict[str, int] = {}
        with self.db.transaction() as conn:
            for name, sql in steps:
                counts[name] = conn.execute(sql, cid).rowcount
        return counts

    def delete_cascade(self, category_id: str) -> dict[str, int]:
        """카테고리와 소속 기록을 한 트랜잭션으로 삭제 (외래키 순서 준수). 공용 모델은 남긴다."""
        cid = (category_id,)
        steps = [
            ("job_events", "DELETE FROM job_events WHERE job_id IN (SELECT id FROM jobs WHERE category_id=?)"),
            ("jobs", "DELETE FROM jobs WHERE category_id=?"),
            ("album_tracks", "DELETE FROM album_tracks WHERE song_id IN (SELECT id FROM songs WHERE category_id=?)"),
            ("songs", "DELETE FROM songs WHERE category_id=?"),
            ("lyrics_drafts", "DELETE FROM lyrics_drafts WHERE category_id=?"),
            ("bindings", "DELETE FROM category_model_bindings WHERE category_id=?"),
            ("members", "DELETE FROM category_model_members WHERE category_id=?"),
            ("member_refs", "DELETE FROM category_model_members WHERE model_id IN "
                            "(SELECT id FROM models WHERE category_id=?)"),
            ("binding_refs", "DELETE FROM category_model_bindings WHERE model_id IN "
                             "(SELECT id FROM models WHERE category_id=?)"),
            ("models", "DELETE FROM models WHERE category_id=?"),
            ("dataset_items",
             "DELETE FROM dataset_items WHERE dataset_id IN (SELECT id FROM datasets WHERE category_id=?)"),
            ("datasets", "DELETE FROM datasets WHERE category_id=?"),
            ("categories", "DELETE FROM categories WHERE id=?"),
        ]
        counts: dict[str, int] = {}
        with self.db.transaction() as conn:          # 중간에 실패하면 전부 롤백
            for name, sql in steps:
                counts[name] = conn.execute(sql, cid).rowcount
        return counts
