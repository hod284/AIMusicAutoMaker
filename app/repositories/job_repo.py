from __future__ import annotations

import sqlite3

from app.domain.entities import Job, JobEvent
from app.domain.enums import JobKind, JobStatus
from app.repositories.base import BaseRepository, dumps, iso, loads, parse_dt

_ACTIVE = (
    JobStatus.CREATED.value,
    JobStatus.VALIDATING.value,
    JobStatus.QUEUED.value,
    JobStatus.RUNNING.value,
    JobStatus.EXPORTING.value,
    JobStatus.VALIDATING_ADAPTER.value,
)


class JobRepository(BaseRepository):
    def _row(self, r: sqlite3.Row) -> Job:
        return Job(
            id=r["id"],
            kind=JobKind(r["kind"]),
            status=JobStatus(r["status"]),
            category_id=r["category_id"],
            dataset_id=r["dataset_id"],
            model_id=r["model_id"],
            progress=r["progress"],
            params=loads(r["params_json"], {}),
            result=loads(r["result_json"], {}),
            error=r["error"],
            retries=r["retries"],
            heartbeat_at=parse_dt(r["heartbeat_at"]),
            created_at=parse_dt(r["created_at"]),
            started_at=parse_dt(r["started_at"]),
            ended_at=parse_dt(r["ended_at"]),
        )

    def add(self, j: Job) -> Job:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs (id, kind, category_id, dataset_id, model_id, status, progress, params_json,"
                " result_json, error, retries, heartbeat_at, created_at, started_at, ended_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (j.id, j.kind.value, j.category_id, j.dataset_id, j.model_id, j.status.value, j.progress,
                 dumps(j.params), dumps(j.result), j.error, j.retries, iso(j.heartbeat_at), iso(j.created_at),
                 iso(j.started_at), iso(j.ended_at)),
            )
        return j

    def save(self, j: Job) -> Job:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE jobs SET status=?, progress=?, params_json=?, result_json=?, error=?, retries=?,"
                " heartbeat_at=?, started_at=?, ended_at=?, model_id=? WHERE id=?",
                (j.status.value, j.progress, dumps(j.params), dumps(j.result), j.error, j.retries,
                 iso(j.heartbeat_at), iso(j.started_at), iso(j.ended_at), j.model_id, j.id),
            )
        return j

    def get(self, job_id: str) -> Job | None:
        with self.db.read() as conn:
            r = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row(r) if r else None

    def list_recent(self, limit: int = 50) -> list[Job]:
        with self.db.read() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            return [self._row(r) for r in rows]

    def list_active(self) -> list[Job]:
        q = ",".join("?" * len(_ACTIVE))
        with self.db.read() as conn:
            rows = conn.execute(f"SELECT * FROM jobs WHERE status IN ({q}) ORDER BY created_at", _ACTIVE)  # noqa: S608
            return [self._row(r) for r in rows]

    # --- 이벤트 로그 ---
    def add_event(self, e: JobEvent) -> JobEvent:
        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO job_events (job_id, timestamp, level, message, details_json) VALUES (?,?,?,?,?)",
                (e.job_id, iso(e.timestamp), e.level, e.message, dumps(e.details)),
            )
            e.id = cur.lastrowid
        return e

    def events(self, job_id: str, limit: int = 500) -> list[JobEvent]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT * FROM job_events WHERE job_id=? ORDER BY id DESC LIMIT ?", (job_id, limit)
            ).fetchall()
        return [
            JobEvent(id=r["id"], job_id=r["job_id"], timestamp=parse_dt(r["timestamp"]), level=r["level"],
                     message=r["message"], details=loads(r["details_json"], {}))
            for r in reversed(rows)
        ]
