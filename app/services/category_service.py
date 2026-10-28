"""카테고리 + 모델 레지스트리 서비스 (CAT-001/002/003)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from app.core.errors import NotFoundError, ValidationError
from app.core.observer import EventBus, Topics
from app.domain.entities import Category, ModelRecord, StyleConstraints
from app.domain.enums import Engine, ValidationStatus
from app.repositories.category_repo import CategoryRepository
from app.repositories.model_repo import ModelRepository
from app.services.workspace import CategoryWorkspace, WorkspaceManager
from app.utils.fileio import sha256_dir, slugify
from app.utils.path_guard import safe_filename

ACE_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
# mlx_lm.lora 의 어댑터 산출물 (설치 버전에 따라 다를 수 있어 둘 중 하나 허용)
QWEN_ADAPTER_FILES = ("adapter_config.json",)
QWEN_WEIGHT_CANDIDATES = ("adapters.safetensors", "adapter_model.safetensors")


VOICE_REFERENCE = "reference.wav"     # 목소리 모델 폴더: 참고 녹음 (필수)
VOICE_CHECKPOINT = "ft_model.pth"     # Seed-VC 학습 결과 (학습한 모델만)


def voice_model_files(path: Path) -> tuple[Path, Path | None, Path | None]:
    """(참고 녹음, 학습 체크포인트 또는 None, 설정 yml 또는 None)"""
    path = Path(path)
    ckpt = path / VOICE_CHECKPOINT
    cfg = next(iter(sorted(path.glob("*.yml"))), None)
    return path / VOICE_REFERENCE, (ckpt if ckpt.is_file() else None), cfg


def validate_adapter_dir(engine: Engine, path: Path) -> tuple[bool, str]:
    """어댑터 폴더 구조 검사. (통과 여부, 메시지)"""
    path = Path(path)
    if not path.is_dir():
        return False, f"폴더가 없습니다: {path}"
    if engine == Engine.VOICE:
        ref, ckpt, cfg = voice_model_files(path)
        if not ref.is_file():
            return False, f"목소리 모델 참고 녹음({VOICE_REFERENCE})이 없습니다."
        if ckpt is not None and cfg is None:
            return False, "학습한 목소리 모델의 설정 파일(yml)이 없습니다."
        return True, "학습한 목소리 모델 (Seed-VC)" if ckpt else "참고 녹음만 쓰는 목소리 (학습 없음)"
    if engine == Engine.ACE:
        missing = [f for f in ACE_ADAPTER_FILES if not (path / f).is_file()]
        if missing:
            return False, f"ACE 어댑터 필수 파일 누락: {', '.join(missing)}"
        return True, "adapter_config.json / adapter_model.safetensors 확인"
    missing = [f for f in QWEN_ADAPTER_FILES if not (path / f).is_file()]
    if missing or not any((path / w).is_file() for w in QWEN_WEIGHT_CANDIDATES):
        return False, "MLX 어댑터 파일(adapter_config.json + adapters.safetensors) 누락"
    return True, "MLX 어댑터 구조 확인"


class CategoryService:
    def __init__(self, repo: CategoryRepository, models: ModelRepository, workspace: WorkspaceManager,
                 bus: EventBus | None = None) -> None:
        self.repo = repo
        self.models = models
        self.workspace = workspace
        self.bus = bus or EventBus()

    # --- 카테고리 ---
    def create(self, name: str, description: str = "", *, language: str = "ko", tags: list[str] | None = None,
               style: StyleConstraints | dict[str, Any] | None = None, slug: str | None = None) -> Category:
        base_slug = slugify(slug or name)
        unique = base_slug
        n = 2
        while self.repo.get_by_slug(unique) is not None:
            unique = f"{base_slug}-{n}"
            n += 1
        style_obj = style if isinstance(style, StyleConstraints) else StyleConstraints(**(style or {}))
        category = Category(slug=unique, name=name, description=description, language=language,
                            tags=tags or [], style=style_obj)
        self.workspace.ensure(category.id)
        self.repo.add(category)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=category.id, action="created")
        return category

    def get(self, category_id: str) -> Category:
        category = self.repo.get(category_id)
        if category is None:
            raise NotFoundError("카테고리를 찾을 수 없습니다.", detail=category_id)
        return category

    def list(self) -> list[Category]:
        return self.repo.list()

    def update(self, category_id: str, **changes: Any) -> Category:
        """이름 변경해도 slug/폴더(UUID)는 유지된다."""
        category = self.get(category_id)
        allowed = {"name", "description", "language", "tags", "style"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValidationError(f"수정할 수 없는 필드: {', '.join(sorted(unknown))}")
        if "style" in changes and isinstance(changes["style"], dict):
            changes["style"] = StyleConstraints(**changes["style"])
        updated = category.model_copy(update=changes)
        Category.model_validate(updated.model_dump())  # 재검증
        self.repo.update(updated)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=category_id, action="updated")
        return updated

    def delete_permanently(self, category_id: str) -> dict[str, object]:
        """카테고리 삭제. DB 기록은 삭제, 파일은 휴지통(data/trash)으로 이동."""
        self.get(category_id)
        if self.repo.active_job_count(category_id):
            raise ValidationError("이 카테고리로 실행 중인 작업이 있습니다. 작업을 멈춘 뒤 삭제하세요.")
        counts = self.repo.delete_cascade(category_id)
        trash = self.workspace.move_to_trash(category_id)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=category_id, action="deleted")
        return {"deleted": counts, "trash": str(trash) if trash else None}

    def delete_grouping(self, category_id: str) -> dict[str, int]:
        """카테고리 묶음만 삭제 (모델·곡·가사·파일은 남김)."""
        self.get(category_id)
        if self.repo.active_job_count(category_id):
            raise ValidationError("이 카테고리로 실행 중인 작업이 있습니다. 작업이 끝난 뒤 삭제하세요.")
        counts = self.repo.delete_grouping(category_id)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=category_id, action="deleted")
        return counts

    def workspace_of(self, category_id: str) -> CategoryWorkspace:
        self.get(category_id)
        return self.workspace.ensure(category_id)

    # --- 모델 레지스트리 ---
    def register_model(self, category_id: str | None, engine: Engine, base_model: str, adapter_path: Path, *,
                       dataset_id: str | None = None, train_config: dict[str, Any] | None = None,
                       notes: str = "", display_name: str = "") -> ModelRecord:
        """기존 어댑터(예: final_lora)를 '참조'로 등록한다. 파일을 복사/수정하지 않는다."""
        if category_id is not None:
            self.get(category_id)
        ok, message = validate_adapter_dir(engine, adapter_path)
        if not ok:
            raise ValidationError(message, detail=str(adapter_path))
        record = ModelRecord(
            category_id=category_id, engine=engine, base_model=base_model, adapter_path=str(Path(adapter_path)),
            version=self.models.next_version(category_id, engine), dataset_id=dataset_id,
            train_config=train_config or {}, file_hash=sha256_dir(Path(adapter_path)),
            validation_status=ValidationStatus.STRUCTURE_OK, notes=notes or message,
            display_name=display_name.strip(),
        )
        self.models.add(record)
        if category_id is not None:
            self.models.add_member(category_id, record.id)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=category_id, action="model_registered")
        return record

    # --- 카테고리 ↔ 모델 멤버십 ---
    def add_model_to_category(self, category_id: str, model_id: str) -> None:
        self.get(category_id)
        model = self.models.get(model_id)
        if model is None:
            raise NotFoundError("모델을 찾을 수 없습니다.", detail=model_id)
        self.models.add_member(category_id, model_id)
        # 카테고리에 사용 중인 같은 엔진 모델이 없으면 바로 사용 모델로
        if self.models.active_for(category_id, model.engine) is None:
            self.models.bind_active(category_id, model.engine, model_id)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=category_id, action="member_added")

    def remove_model_from_category(self, category_id: str, model_id: str) -> None:
        """카테고리에서 빼기만 한다. 모델 파일·등록은 그대로 남는다."""
        self.get(category_id)
        self.models.remove_member(category_id, model_id)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=category_id, action="member_removed")

    def category_models(self, category_id: str, engine: Engine | None = None) -> list[ModelRecord]:
        return self.models.members(category_id, engine)

    def categories_of_model(self, model_id: str) -> list[str]:
        return self.models.categories_of(model_id)

    def rename_model(self, model_id: str, display_name: str) -> tuple[ModelRecord, str | None]:
        """이름 변경. 이 앱이 학습으로 만든 모델이면 결과 폴더 이름도 같이 바꾼다.

        반환: (모델, 바뀐 폴더 경로 또는 None). '기존 어댑터 등록'으로 가져온 폴더는 건드리지 않는다.
        """
        model = self.models.get(model_id)
        if model is None:
            raise NotFoundError("모델을 찾을 수 없습니다.", detail=model_id)
        name = display_name.strip()
        if not name:
            raise ValidationError("모델 이름을 입력하세요.")
        if len(name) > 80:
            raise ValidationError("모델 이름은 80자 이하로 입력하세요.")

        new_root: Path | None = None
        root_str = model.train_config.get("export_root")
        if not root_str and model.dataset_id:
            # 이 기능 이전에 앱에서 학습한 모델: .../runs/lora_vNNN/export/... 의 export 폴더를 결과 폴더로 본다
            parts = Path(model.adapter_path).parts
            if "runs" in parts and len(parts) > parts.index("runs") + 2:
                root_str = str(Path(*parts[: parts.index("runs") + 3]))
        if root_str:
            old_root = Path(root_str)
            adapter = Path(model.adapter_path)
            if old_root.is_dir() and (adapter == old_root or old_root in adapter.parents):
                candidate = old_root.parent / safe_filename(name)
                if candidate != old_root:
                    for protected in self.workspace.guard.protected:
                        if candidate == protected or protected in candidate.parents:
                            raise ValidationError("보호된 ACE 폴더 안으로는 바꿀 수 없습니다.", detail=str(candidate))
                    if candidate.exists():
                        raise ValidationError(f"'{candidate.name}' 폴더가 이미 있습니다. 다른 이름을 쓰세요.",
                                              detail=str(candidate))
                    old_root.rename(candidate)          # 같은 폴더 안에서 이름만 변경
                    new_adapter = candidate / adapter.relative_to(old_root)
                    self.models.update_location(
                        model_id, str(new_adapter), {**model.train_config, "export_root": str(candidate)})
                    new_root = candidate

        self.models.rename(model_id, name)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=model.category_id, action="model_renamed",
                         model_id=model_id)
        return self.models.get(model_id), (str(new_root) if new_root else None)  # type: ignore[return-value]

    # --- 모델 삭제 ---
    def model_file_targets(self, model_id: str) -> list[Path]:
        """모델 삭제 시 함께 지울 폴더. 이 앱이 학습으로 만든 폴더만 포함한다.

        - 결과 폴더 (결과 저장 폴더/모델 이름) : 최종 LoRA
        - ACE 학습 폴더 (datasets/studio/.../runs/lora_vNNN) : 체크포인트
        가져와 등록한 어댑터(예: final_lora)나 판단할 수 없는 경로는 지우지 않는다.
        """
        model = self.models.get(model_id)
        if model is None:
            raise NotFoundError("모델을 찾을 수 없습니다.", detail=model_id)
        adapter = Path(model.adapter_path).expanduser()
        targets: list[Path] = []

        root_str = model.train_config.get("export_root")
        if root_str:
            root = Path(root_str).expanduser()
            if adapter == root or root in adapter.parents:
                targets.append(root)

        run_dir: Path | None = None
        out = model.train_config.get("output_dir")
        if out and Path(out).name == "output" and Path(out).parent.parent.name == "runs":
            run_dir = Path(out).parent
        elif model.dataset_id and "runs" in adapter.parts:
            i = adapter.parts.index("runs")
            if len(adapter.parts) > i + 1:
                run_dir = Path(*adapter.parts[: i + 2])
        if run_dir is not None and run_dir.name.startswith("lora_v") and _under_studio_datasets(run_dir):
            targets.append(run_dir)

        result: list[Path] = []
        for t in targets:
            if not t.exists() and not t.is_symlink():
                continue
            self._check_deletable(t)
            if not any(t == r or r in t.parents for r in result):   # 겹치면 바깥 폴더 하나만
                result = [r for r in result if t not in r.parents] + [t]
        return result

    def _check_deletable(self, path: Path) -> None:
        resolved = path.resolve()
        home = Path.home().resolve()
        if len(resolved.parts) < 4 or resolved in (home, home.parent, Path("/")):
            raise ValidationError("너무 위쪽 폴더라 삭제하지 않습니다.", detail=str(resolved))
        for protected in self.workspace.guard.protected:
            if resolved == protected or protected in resolved.parents or resolved in protected.parents:
                raise ValidationError("보호된 기존 자산이 포함된 폴더라 삭제하지 않습니다.", detail=str(resolved))

    def delete_model(self, model_id: str, *, delete_files: bool = True) -> list[str]:
        """모델 삭제: (선택) 결과 파일을 실제로 지우고 → DB 기록 삭제. 곡은 남는다.

        파일 삭제가 실패하면 DB 기록을 남겨 다시 시도할 수 있게 한다. 반환: 지운 폴더 목록.
        """
        model = self.models.get(model_id)
        if model is None:
            raise NotFoundError("모델을 찾을 수 없습니다.", detail=model_id)
        removed: list[str] = []
        if delete_files:
            for target in self.model_file_targets(model_id):
                try:
                    if target.is_symlink() or target.is_file():
                        target.unlink()
                    else:
                        shutil.rmtree(target)
                except OSError as exc:
                    raise ValidationError(f"파일 삭제 실패: {exc}",
                                          detail=f"{target} (이미 지운 폴더: {removed or '없음'})") from exc
                removed.append(str(target))
        self.models.delete(model_id)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=model.category_id, action="model_deleted",
                         model_id=model_id)
        return removed

    def browse_dirs(self, root: Path, path: str | None = None) -> tuple[Path, list[tuple[str, bool]]]:
        """어댑터 폴더 고르기용 읽기 전용 탐색. root 바깥으로는 못 나간다.

        반환: (현재 폴더, [(하위 폴더 이름, 어댑터 폴더 여부)])
        """
        root = Path(root).expanduser().resolve()
        current = Path(path).expanduser().resolve() if path else root
        try:
            current.relative_to(root)
        except ValueError:
            current = root
        if not current.is_dir():
            current = root
        entries: list[tuple[str, bool]] = []
        for child in sorted(current.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir() and not child.name.startswith(".") and child.name != "__pycache__":
                is_adapter = (child / "adapter_config.json").is_file()
                entries.append((child.name, is_adapter))
        return current, entries

    def set_active_model(self, category_id: str, engine: Engine, model_id: str | None) -> None:
        """한 카테고리×엔진에 활성 버전은 하나. None 이면 기본 모델 fallback."""
        self.get(category_id)
        if model_id is None:
            self.models.unbind(category_id, engine)
        else:
            model = self.models.get(model_id)
            if model is None or model.engine != engine:
                raise ValidationError("엔진이 일치하는 모델이 아닙니다.")
            if not self.models.is_member(category_id, model_id):
                self.models.add_member(category_id, model_id)
            self.models.bind_active(category_id, engine, model_id)
        self.bus.publish(Topics.CATEGORY_CHANGED, category_id=category_id, action="binding")

    def active_model(self, category_id: str, engine: Engine) -> ModelRecord | None:
        return self.models.active_for(category_id, engine)

    def list_models(self, category_id: str | None = None, engine: Engine | None = None) -> list[ModelRecord]:
        return self.models.list(category_id, engine)


def _under_studio_datasets(path: Path) -> bool:
    """ACE-Step-1.5/datasets/studio/... 아래(이 앱이 만든 학습 폴더)인지."""
    parts = Path(path).parts
    return any(parts[i] == "datasets" and parts[i + 1] == "studio" for i in range(len(parts) - 1))
