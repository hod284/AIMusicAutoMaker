"""원자적 파일 IO / 해시 / slug."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any


def atomic_write_bytes(path: Path, data: bytes, *, overwrite: bool = False) -> Path:
    """임시파일에 쓴 뒤 rename. ``overwrite=False`` 면 기존 파일을 절대 덮어쓰지 않는다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"이미 존재하는 파일은 덮어쓰지 않습니다: {path}")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)  # 대상이 생기는 경쟁 상황에서도 덮어쓰지 않음
            os.unlink(tmp)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def atomic_write_text(path: Path, text: str, *, overwrite: bool = False) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"), overwrite=overwrite)


def atomic_write_json(path: Path, data: Any, *, overwrite: bool = False) -> Path:
    return atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2), overwrite=overwrite)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    """디렉터리 내 파일들의 (상대경로, 내용) 해시. LoRA 어댑터 지문용."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


_SLUG_RE = re.compile(r"[^a-z0-9가-힣]+")


def slugify(text: str) -> str:
    """사람이 읽을 수 있는 slug. 폴더 참조는 slug 가 아니라 category_id(UUID)를 쓴다."""
    text = unicodedata.normalize("NFC", text).strip().lower()
    slug = _SLUG_RE.sub("-", text).strip("-")
    return slug or "category"
