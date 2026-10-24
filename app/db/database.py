"""SQLite 접근 (싱글톤).

- WAL 모드, FK 활성화, 버전 테이블 기반 SQL 마이그레이션.
- 연결은 호출마다 짧게 열고 닫는다 (NiceGUI 이벤트 루프 / 워커 간 스레드 안전).
- 대용량 오디오/모델 파일은 DB 에 저장하지 않는다 (경로·해시만).
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.core.singleton import SingletonMeta

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class Database(metaclass=SingletonMeta):
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """쓰기 트랜잭션. 예외 시 롤백."""
        with self._write_lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    def migrate(self) -> list[str]:
        applied: list[str] = []
        with self._write_lock:
            conn = self._connect()
            # 테이블 재생성(003 등) 중 외래키 검사로 막히지 않게 끄고, 끝난 뒤 무결성 검사
            conn.execute("PRAGMA foreign_keys = OFF")
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_migrations ("
                    "name TEXT PRIMARY KEY, applied_at TEXT DEFAULT CURRENT_TIMESTAMP)"
                )
                done = {r["name"] for r in conn.execute("SELECT name FROM schema_migrations")}
                for sql_file in sorted(MIGRATIONS_DIR.glob("*.sql")):
                    if sql_file.name in done:
                        continue
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        for statement in _split_sql(sql_file.read_text(encoding="utf-8")):
                            conn.execute(statement)
                        conn.execute("INSERT INTO schema_migrations(name) VALUES (?)", (sql_file.name,))
                        conn.execute("COMMIT")
                    except Exception:
                        conn.execute("ROLLBACK")
                        raise
                    applied.append(sql_file.name)
                broken = conn.execute("PRAGMA foreign_key_check").fetchall()
                if broken:
                    raise RuntimeError(f"마이그레이션 후 외래키 불일치: {[tuple(r) for r in broken[:5]]}")
            finally:
                conn.close()
        return applied


def _split_sql(script: str) -> list[str]:
    """주석 제거 후 ';' 기준 분리 (트리거/문자열 내 세미콜론은 사용하지 않는 규칙)."""
    lines = [ln for ln in script.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
