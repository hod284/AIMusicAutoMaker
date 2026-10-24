"""리포지토리 패턴 베이스.

서비스는 SQL 을 모른다. 리포지토리 인터페이스만 의존하므로
나중에 SQLAlchemy/다른 저장소로 교체할 때 서비스 코드를 바꾸지 않아도 된다.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from app.db.database import Database


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    return json.loads(value)


def iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class BaseRepository:
    def __init__(self, db: Database) -> None:
        self.db = db
