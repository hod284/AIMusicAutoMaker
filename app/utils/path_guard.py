"""경로 격리 (보안 11.3, MUS-001).

- ``../`` 탈출, 심볼릭 링크 탈출, 허용 루트 바깥 접근을 차단한다.
- 기존 ACE 자산(test01, lora_output, checkpoints)과 .venv 는 쓰기 금지 경로로 보호한다.
- 안전한 파일명 생성.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from app.core.errors import PathSecurityError

_SAFE_CHARS = re.compile(r"[^0-9A-Za-z가-힣ㄱ-ㅎㅏ-ㅣ\u1100-\u11ff\ua960-\ua97f\ud7b0-\ud7ff._-]+")


def safe_filename(name: str, max_len: int = 120) -> str:
    """업로드 파일명을 안전한 이름으로 정규화한다 (디렉터리 구성요소 제거)."""
    name = unicodedata.normalize("NFC", Path(name).name)
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = _SAFE_CHARS.sub("_", stem).strip("._") or "file"
    ext = _SAFE_CHARS.sub("", ext).lower()
    result = f"{stem[:max_len]}.{ext}" if ext else stem[:max_len]
    if result in {".", ".."}:
        raise PathSecurityError(detail=name)
    return result


class PathGuard:
    """허용 루트 집합과 보호 경로 집합을 받아 경로를 검증한다."""

    def __init__(self, allowed_roots: list[Path], protected: list[Path] | None = None) -> None:
        self.allowed_roots = [Path(p).resolve() for p in allowed_roots]
        self.protected = [Path(p).resolve() for p in (protected or [])]

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def resolve(self, path: str | Path, *, base: Path | None = None) -> Path:
        """``base`` 기준 상대경로(또는 절대경로)를 해석하고 허용 루트 안인지 검사.

        ``resolve()`` 는 심볼릭 링크를 따라가므로 링크를 통한 탈출도 걸러진다.
        """
        raw = Path(path)
        if base is not None and not raw.is_absolute():
            raw = Path(base) / raw
        resolved = raw.resolve()
        if not any(self._is_within(resolved, root) for root in self.allowed_roots):
            raise PathSecurityError(detail=f"허용 루트 바깥: {resolved}")
        return resolved

    def ensure_writable(self, path: str | Path, *, base: Path | None = None) -> Path:
        resolved = self.resolve(path, base=base)
        for protected in self.protected:
            if resolved == protected or self._is_within(resolved, protected):
                raise PathSecurityError("기존 자산은 수정할 수 없습니다.", detail=str(resolved))
        return resolved

    def child(self, root: Path, *parts: str) -> Path:
        """``root`` 아래에 안전한 하위 경로를 만든다. 각 구성요소는 이름 하나여야 한다."""
        for part in parts:
            if part in {"", ".", ".."} or "/" in part or "\\" in part:
                raise PathSecurityError(detail=f"잘못된 경로 구성요소: {part!r}")
        return self.ensure_writable(Path(root).joinpath(*parts))
