from __future__ import annotations

import sqlite3

from app.domain.entities import ModelRecord
from app.domain.enums import Engine, ValidationStatus
from app.repositories.base import BaseRepository, dumps, iso, loads, parse_dt


class ModelRepository(BaseRepository):
    """CAT-002 모델 레지스트리 + 카테고리별 활성 바인딩."""

    def _row(self, r: sqlite3.Row) -> ModelRecord:
        return ModelRecord(
            id=r["id"],
            category_id=r["category_id"],
            engine=Engine(r["engine"]),
            base_model=r["base_model"],
            adapter_path=r["adapter_path"],
            version=r["version"],
            dataset_id=r["dataset_id"],
            train_config=loads(r["train_config_json"], {}),
            file_hash=r["file_hash"],
            validation_status=ValidationStatus(r["validation_status"]),
            notes=r["notes"],
            display_name=r["display_name"] if "display_name" in r.keys() else "",
            created_at=parse_dt(r["created_at"]),
        )

    def next_version(self, category_id: str | None, engine: Engine) -> int:
        with self.db.read() as conn:
            r = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM models WHERE category_id IS ? AND engine=?",
                (category_id, engine.value),
            ).fetchone()
        return int(r[0]) + 1

    def add(self, m: ModelRecord) -> ModelRecord:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO models (id, category_id, engine, base_model, adapter_path, version, dataset_id,"
                " train_config_json, file_hash, validation_status, notes, display_name, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (m.id, m.category_id, m.engine.value, m.base_model, m.adapter_path, m.version, m.dataset_id,
                 dumps(m.train_config), m.file_hash, m.validation_status.value, m.notes, m.display_name,
                 iso(m.created_at)),
            )
        return m

    def set_validation(self, model_id: str, status: ValidationStatus, notes: str | None = None) -> None:
        with self.db.transaction() as conn:
            if notes is None:
                conn.execute("UPDATE models SET validation_status=? WHERE id=?", (status.value, model_id))
            else:
                conn.execute(
                    "UPDATE models SET validation_status=?, notes=? WHERE id=?", (status.value, notes, model_id)
                )

    def rename(self, model_id: str, display_name: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("UPDATE models SET display_name=? WHERE id=?", (display_name, model_id))

    def update_location(self, model_id: str, adapter_path: str, train_config: dict) -> None:
        with self.db.transaction() as conn:
            conn.execute("UPDATE models SET adapter_path=?, train_config_json=? WHERE id=?",
                         (adapter_path, dumps(train_config), model_id))

    def get(self, model_id: str) -> ModelRecord | None:
        with self.db.read() as conn:
            r = conn.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        return self._row(r) if r else None

    def list(self, category_id: str | None = None, engine: Engine | None = None) -> list[ModelRecord]:
        sql, args = "SELECT * FROM models WHERE 1=1", []
        if category_id is not None:
            sql += " AND category_id=?"
            args.append(category_id)
        if engine is not None:
            sql += " AND engine=?"
            args.append(engine.value)
        sql += " ORDER BY created_at DESC"
        with self.db.read() as conn:
            return [self._row(r) for r in conn.execute(sql, args)]

    def delete(self, model_id: str) -> None:
        """모델 기록 삭제. 이 모델로 만든 곡·가사·작업 기록은 남기고 모델 참조만 비운다."""
        with self.db.transaction() as conn:
            conn.execute("UPDATE songs SET music_model_id=NULL WHERE music_model_id=?", (model_id,))
            conn.execute("UPDATE songs SET qwen_model_id=NULL WHERE qwen_model_id=?", (model_id,))
            conn.execute("UPDATE lyrics_drafts SET model_id=NULL WHERE model_id=?", (model_id,))
            conn.execute("UPDATE jobs SET model_id=NULL WHERE model_id=?", (model_id,))
            conn.execute("DELETE FROM category_model_bindings WHERE model_id=?", (model_id,))
            conn.execute("DELETE FROM category_model_members WHERE model_id=?", (model_id,))
            conn.execute("DELETE FROM models WHERE id=?", (model_id,))

    # --- 카테고리 멤버십 ---
    def add_member(self, category_id: str, model_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO category_model_members (category_id, model_id) VALUES (?,?)",
                         (category_id, model_id))

    def remove_member(self, category_id: str, model_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM category_model_bindings WHERE category_id=? AND model_id=?",
                         (category_id, model_id))
            conn.execute("DELETE FROM category_model_members WHERE category_id=? AND model_id=?",
                         (category_id, model_id))

    def members(self, category_id: str, engine: Engine | None = None) -> list[ModelRecord]:
        sql = ("SELECT m.* FROM category_model_members c JOIN models m ON m.id=c.model_id"
               " WHERE c.category_id=?")
        args: list[str] = [category_id]
        if engine is not None:
            sql += " AND m.engine=?"
            args.append(engine.value)
        with self.db.read() as conn:
            return [self._row(r) for r in conn.execute(sql + " ORDER BY m.created_at DESC", args)]

    def categories_of(self, model_id: str) -> list[str]:
        with self.db.read() as conn:
            return [r[0] for r in conn.execute(
                "SELECT category_id FROM category_model_members WHERE model_id=?", (model_id,))]

    def is_member(self, category_id: str, model_id: str) -> bool:
        with self.db.read() as conn:
            return conn.execute("SELECT 1 FROM category_model_members WHERE category_id=? AND model_id=?",
                                (category_id, model_id)).fetchone() is not None

    # --- 활성 바인딩 ---
    def bind_active(self, category_id: str, engine: Engine, model_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO category_model_bindings (category_id, engine, model_id) VALUES (?,?,?)"
                " ON CONFLICT(category_id, engine) DO UPDATE SET model_id=excluded.model_id",
                (category_id, engine.value, model_id),
            )

    def unbind(self, category_id: str, engine: Engine) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "DELETE FROM category_model_bindings WHERE category_id=? AND engine=?",
                (category_id, engine.value),
            )

    def active_for(self, category_id: str, engine: Engine) -> ModelRecord | None:
        with self.db.read() as conn:
            r = conn.execute(
                "SELECT m.* FROM category_model_bindings b JOIN models m ON m.id=b.model_id"
                " WHERE b.category_id=? AND b.engine=?",
                (category_id, engine.value),
            ).fetchone()
        return self._row(r) if r else None
