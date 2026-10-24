"""학습 워크플로 (Phase 4/5 확장 지점).

- 가사: ``LyricsDatasetBuilder`` (JSONL 검증·분할·버전 스냅샷) + ``LyricsTrainHandler`` (mlx_lm.lora)
- 음악: ``app/services/music_training_service.py`` 참고

학습 산출물은 항상 새 버전 폴더에 쓰고, 기존 파일은 덮어쓰지 않는다.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.adapters.ports import LoraTrainSpec, LyricsEnginePort, TrainProgress
from app.core.errors import ValidationError
from app.domain.entities import Dataset
from app.domain.enums import DatasetKind, Engine, JobKind, JobStatus
from app.repositories.content_repo import DatasetRepository
from app.services.category_service import CategoryService, validate_adapter_dir
from app.services.job_service import JobContext, JobHandler
from app.utils.fileio import atomic_write_text

MIN_RECOMMENDED_EXAMPLES = 50


class JsonlReport(BaseModel):
    total: int = 0
    valid: int = 0
    duplicates: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_chat_record(obj: Any) -> str | None:
    """MLX-LM chat 형식 {"messages":[{"role","content"},...]} 검사. 오류 메시지 또는 None."""
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        return "messages 배열이 없습니다."
    roles = [m.get("role") for m in obj["messages"] if isinstance(m, dict)]
    if "assistant" not in roles or "user" not in roles:
        return "user / assistant 메시지가 모두 필요합니다."
    for m in obj["messages"]:
        if not isinstance(m, dict) or not str(m.get("content", "")).strip():
            return "빈 content 가 있습니다."
    return None


class LyricsDatasetBuilder:
    def __init__(self, categories: CategoryService, datasets: DatasetRepository) -> None:
        self.categories = categories
        self.datasets = datasets

    def parse_jsonl(self, text: str) -> tuple[list[dict[str, Any]], JsonlReport]:
        report = JsonlReport()
        seen: set[str] = set()
        records: list[dict[str, Any]] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            report.total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                report.errors.append(f"{lineno}행: JSON 오류 ({exc.msg})")
                continue
            if err := validate_chat_record(obj):
                report.errors.append(f"{lineno}행: {err}")
                continue
            answer = next(m["content"] for m in obj["messages"] if m["role"] == "assistant").strip()
            if answer in seen:
                report.duplicates += 1
                continue
            seen.add(answer)
            records.append(obj)
        report.valid = len(records)
        if report.valid < MIN_RECOMMENDED_EXAMPLES:
            report.warnings.append(
                f"예제 {report.valid}개 — 파이프라인 동작 테스트용입니다. 스타일 학습에는 {MIN_RECOMMENDED_EXAMPLES}개 이상 권장."
            )
        return records, report

    def build(self, category_id: str, jsonl_text: str, valid_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, JsonlReport]:
        records, report = self.parse_jsonl(jsonl_text)
        if report.errors:
            raise ValidationError("JSONL 검증 실패", detail=report.errors[:20])
        if len(records) < 2:
            raise ValidationError("train/valid 분할을 위해 최소 2개 예제가 필요합니다.")
        rng = random.Random(seed)
        rng.shuffle(records)
        n_valid = max(1, int(len(records) * valid_ratio))
        valid, train = records[:n_valid], records[n_valid:]
        ws = self.categories.workspace_of(category_id)
        version = self.datasets.next_version(category_id, DatasetKind.LYRICS)
        out = ws.lyrics_training / f"v{version:03d}"
        if out.exists():
            raise ValidationError("이미 존재하는 데이터셋 버전 폴더입니다.", detail=str(out))
        dump = lambda rows: "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"  # noqa: E731
        atomic_write_text(out / "train.jsonl", dump(train))
        atomic_write_text(out / "valid.jsonl", dump(valid))
        dataset = Dataset(category_id=category_id, kind=DatasetKind.LYRICS, version=version, status="ready",
                          source_path=str(out))
        self.datasets.add(dataset)
        return dataset, report


class LyricsTrainHandler(JobHandler):
    kind = JobKind.LYRICS_TRAIN

    def __init__(self, engine: LyricsEnginePort, categories: CategoryService, datasets: DatasetRepository,
                 base_model: str) -> None:
        self.engine = engine
        self.categories = categories
        self.datasets = datasets
        self.base_model = base_model

    def validate(self, params: dict[str, Any]) -> None:
        if not params.get("category_id") or not params.get("dataset_id"):
            raise ValidationError("category_id, dataset_id 가 필요합니다.")

    def describe(self, params: dict[str, Any]) -> str:
        return "Qwen 가사 LoRA 학습"

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        p = ctx.params
        dataset = next((d for d in self.datasets.list(p["category_id"], DatasetKind.LYRICS) if d.id == p["dataset_id"]), None)
        if dataset is None:
            raise ValidationError("해당 카테고리의 가사 데이터셋이 아닙니다.")
        ws = self.categories.workspace_of(p["category_id"])
        version = self.categories.models.next_version(p["category_id"], Engine.QWEN)
        adapter_dir = ws.lyrics_adapters / f"v{version:03d}"
        if adapter_dir.exists():
            raise ValidationError("어댑터 출력 폴더가 이미 존재합니다.", detail=str(adapter_dir))
        spec = LoraTrainSpec(model=p.get("base_model", self.base_model), data_dir=Path(dataset.source_path),
                             adapter_path=adapter_dir, iters=p.get("iters", 100), batch_size=p.get("batch_size", 1),
                             learning_rate=p.get("learning_rate"), max_seq_length=p.get("max_seq_length"),
                             num_layers=p.get("num_layers"), save_every=p.get("save_every"))
        losses: list[dict[str, Any]] = []

        def on_progress(tp: TrainProgress) -> None:
            ctx.check_cancelled()
            if tp.step is not None:
                losses.append(tp.model_dump(exclude={"line"}))
                if tp.total:
                    ctx.progress(tp.step / tp.total)
            if tp.line:
                ctx.log(tp.line, "debug" if tp.step is None else "info")

        handle = await self.engine.start_training(spec, on_progress)
        try:
            code = await handle.wait()
        except BaseException:
            await handle.cancel()
            raise
        if code != 0:
            raise ValidationError(f"mlx_lm.lora 종료 코드 {code}")
        ctx.set_status(JobStatus.EXPORTING)
        ctx.set_status(JobStatus.VALIDATING_ADAPTER)
        ok, msg = validate_adapter_dir(Engine.QWEN, adapter_dir)
        if not ok:
            raise ValidationError(f"학습은 끝났지만 어댑터 검증 실패: {msg}")
        model = self.categories.register_model(p["category_id"], Engine.QWEN, spec.model, adapter_dir,
                                               dataset_id=dataset.id, train_config=spec.model_dump(mode="json"))
        return {"model_id": model.id, "adapter_path": str(adapter_dir), "losses": losses[-200:]}
