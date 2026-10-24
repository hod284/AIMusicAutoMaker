"""모델·카테고리 탭 ViewModel (음악/가사 공통)."""

from __future__ import annotations

from pathlib import Path

from app.core.observer import ObservableProperty, Topics
from app.domain.entities import Category, ModelRecord
from app.domain.enums import Engine
from app.viewmodels.base import BaseViewModel


class ModelsViewModel(BaseViewModel):
    """'모델·카테고리' 탭 공통: 만든 LoRA 모델을 카테고리에 넣고 빼고, 카테고리를 만들고, 모델을 삭제한다.

    음악(ACE)·가사(Qwen) 모두 같은 화면을 쓰고 엔진만 다르다 (Template Method).
    카테고리는 두 엔진이 같이 쓰고, '사용 중' 모델은 카테고리×엔진마다 하나다.
    """

    engine: Engine = Engine.ACE
    kind_label = "음악 LoRA"

    models = ObservableProperty[list[ModelRecord]](factory=list)          # 전체 음악 LoRA
    categories = ObservableProperty[list[Category]](factory=list)
    category_id = ObservableProperty[str | None](None)
    members = ObservableProperty[list[ModelRecord]](factory=list)         # 선택한 카테고리의 모델
    active_id = ObservableProperty[str | None](None)                      # 그중 곡 만들기에 쓰는 모델
    model_search = ObservableProperty("")

    def __init__(self, container=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(container)
        self.listen(Topics.CATEGORY_CHANGED, lambda e: self.load())
        self.load()

    @property
    def visible_models(self) -> list[ModelRecord]:
        q = self.model_search.strip().lower()
        if not q:
            return self.models
        return [m for m in self.models if q in m.label.lower() or q in m.adapter_path.lower()]

    @property
    def member_ids(self) -> set[str]:
        return {m.id for m in self.members}

    def category_names_of(self, model_id: str) -> list[str]:
        names = {c.id: c.name for c in self.categories}
        return [names[c] for c in self.c.categories.categories_of_model(model_id) if c in names]

    def load(self) -> None:
        self.models = self.c.categories.list_models(engine=self.engine)
        self.categories = self.c.categories.list()
        if self.category_id not in {c.id for c in self.categories}:
            self.category_id = self.categories[0].id if self.categories else None
        self._load_members()

    def _load_members(self) -> None:
        if not self.category_id:
            self.members, self.active_id = [], None
            return
        self.members = self.c.categories.category_models(self.category_id, self.engine)
        active = self.c.categories.active_model(self.category_id, self.engine)
        self.active_id = active.id if active else None

    def select_category(self, category_id: str | None) -> None:
        self.category_id = category_id
        self._load_members()

    def set_search(self, text: str | None) -> None:
        self.model_search = text or ""

    def create_category(self, name: str) -> None:
        if not name.strip():
            self.error = "카테고리 이름을 입력하세요."
            return
        cat = self.sync_command(lambda: self.c.categories.create(name.strip()),
                                success=f"카테고리 '{name.strip()}' 를 만들었습니다.")
        if cat:
            self.load()
            self.select_category(cat.id)

    def add(self, model_id: str) -> None:
        if not self.category_id:
            self.error = "왼쪽에서 카테고리를 먼저 고르세요."
            return
        self.sync_command(lambda: self.c.categories.add_model_to_category(self.category_id, model_id),
                          success="카테고리에 넣었습니다.")
        self._load_members()

    def remove(self, model_id: str) -> None:
        if self.category_id:
            self.sync_command(lambda: self.c.categories.remove_model_from_category(self.category_id, model_id),
                              success="카테고리에서 뺐습니다. (모델 파일은 그대로)")
            self._load_members()

    def use(self, model_id: str | None) -> None:
        if self.category_id:
            self.sync_command(lambda: self.c.categories.set_active_model(self.category_id, self.engine, model_id),
                              success="곡 만들기에 쓸 모델을 바꿨습니다." if model_id else "LoRA 없이 기본 모델을 씁니다.")
            self._load_members()

    def delete_category(self) -> None:
        if not self.category_id:
            return
        name = next((c.name for c in self.categories if c.id == self.category_id), "")
        result = self.sync_command(lambda: self.c.categories.delete_grouping(self.category_id))
        if result is not None:
            self.info = f"카테고리 '{name}' 를 삭제했습니다. (모델은 그대로 남습니다)"
            self.category_id = None
            self.load()

    def delete_targets(self, model_id: str) -> list[str] | None:
        """삭제 확인 창에 보여줄 '같이 지워질 폴더'. 오류면 None."""
        result = self.sync_command(lambda: self.c.categories.model_file_targets(model_id))
        return None if result is None else [str(p) for p in result]

    def delete_model(self, model_id: str, delete_files: bool = True) -> None:
        name = next((m.label for m in self.models if m.id == model_id), "")
        removed = self.sync_command(lambda: self.c.categories.delete_model(model_id, delete_files=delete_files))
        if removed is not None:
            if self.engine == Engine.ACE:
                self.c.music.invalidate_lora_state()  # ACE 에 올라가 있던 LoRA 상태를 다시 맞추게
            self.info = (f"'{name}' 모델을 삭제했습니다. 지운 폴더 {len(removed)}개" if removed
                         else f"'{name}' 모델 기록을 삭제했습니다. (지울 파일 없음)")
            self.load()

    @property
    def browse_root(self) -> Path:
        return Path.home()

    def browse(self, path: str | None = None) -> tuple[Path, list[tuple[str, bool]]]:
        result = self.sync_command(lambda: self.c.categories.browse_dirs(self.browse_root, path))
        return result if result else (self.browse_root, [])

    def rename(self, model_id: str, name: str) -> None:
        self._rename_message(self.sync_command(lambda: self.c.categories.rename_model(model_id, name)))
        self.load()

    def _rename_message(self, result: tuple | None) -> None:
        if not result:
            return
        _, new_folder = result
        self.info = (f"이름과 폴더를 바꿨습니다 → {new_folder}" if new_folder
                     else "이름을 바꿨습니다. (가져온 어댑터라 원본 폴더 이름은 그대로 둡니다)")


class MusicModelsViewModel(ModelsViewModel):
    engine = Engine.ACE
    kind_label = "음악 LoRA"


class LyricsModelsViewModel(ModelsViewModel):
    engine = Engine.QWEN
    kind_label = "가사 LoRA"
