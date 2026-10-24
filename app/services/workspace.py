"""카테고리별 작업 폴더 레이아웃 (REQUIREMENTS 3.2).

폴더 이름은 slug 가 아니라 **category_id(UUID)** 를 사용한다 → 이름을 바꿔도 참조가 유지된다.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.utils.path_guard import PathGuard

LAYOUT = (
    "lyrics/source", "lyrics/training", "lyrics/adapters", "lyrics/drafts",
    "music/source", "music/metadata", "music/tensors", "music/adapters",
    "songs", "albums",
)


@dataclass(frozen=True)
class CategoryWorkspace:
    root: Path

    @property
    def lyrics_training(self) -> Path:
        return self.root / "lyrics" / "training"

    @property
    def lyrics_adapters(self) -> Path:
        return self.root / "lyrics" / "adapters"

    @property
    def music_source(self) -> Path:
        return self.root / "music" / "source"

    @property
    def music_metadata(self) -> Path:
        return self.root / "music" / "metadata"

    @property
    def music_tensors(self) -> Path:
        return self.root / "music" / "tensors"

    @property
    def music_adapters(self) -> Path:
        return self.root / "music" / "adapters"

    @property
    def songs(self) -> Path:
        return self.root / "songs"


class WorkspaceManager:
    def __init__(self, categories_dir: Path, guard: PathGuard, trash_dir: Path) -> None:
        self.categories_dir = Path(categories_dir)
        self.guard = guard
        self.trash_dir = Path(trash_dir)

    def for_category(self, category_id: str) -> CategoryWorkspace:
        return CategoryWorkspace(self.guard.child(self.categories_dir, category_id))

    def ensure(self, category_id: str) -> CategoryWorkspace:
        ws = self.for_category(category_id)
        for rel in LAYOUT:
            (ws.root / rel).mkdir(parents=True, exist_ok=True)
        return ws

    def move_to_trash(self, category_id: str) -> Path | None:
        """바로 지우지 않고 data/trash/<시각>_<id> 로 옮긴다. 폴더가 없으면 None."""
        src = self.for_category(category_id).root
        if not src.exists():
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = self.guard.child(self.trash_dir, f"{stamp}_{category_id}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        return dest
