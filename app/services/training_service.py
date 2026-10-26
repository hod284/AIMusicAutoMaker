"""학습 워크플로 (Phase 4/5 확장 지점).

- 가사: ``LyricsDatasetBuilder`` (JSONL 검증·분할·버전 스냅샷) + ``LyricsTrainHandler`` (mlx_lm.lora)
- 음악: ``app/services/music_training_service.py`` 참고

학습 산출물은 항상 새 버전 폴더에 쓰고, 기존 파일은 덮어쓰지 않는다.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.adapters.ports import LoraTrainSpec, LyricsEnginePort, TrainProgress
from app.core.errors import ValidationError
from app.domain.entities import Dataset
from app.domain.enums import DatasetKind, Engine, JobKind, JobStatus
from app.repositories.content_repo import DatasetRepository
from app.services.category_service import CategoryService, validate_adapter_dir
from app.services.lyrics_service import lyrics_system, prompt_labels, write_in_line
from app.services.job_service import JobContext, JobHandler
from app.utils.fileio import atomic_write_text
from app.utils.path_guard import safe_filename

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


# --- 가사 .txt → 학습 예제 변환 ---
_MARKER_RE = re.compile(r"^\s*\[([^\]\n]{1,40})\]\s*$")
# .txt 맨 위 머리말 이름 (한국어·영어·일본어 모두 인식) → 공통 키
_HEADER_KEYS = {
    "제목": "title", "title": "title", "タイトル": "title",
    "주제": "topic", "topic": "topic", "テーマ": "topic",
    "화자": "speaker", "narrator": "speaker", "語り手": "speaker",
    "분위기": "mood", "mood": "mood", "雰囲気": "mood",
    "시점": "viewpoint", "point of view": "viewpoint", "視点": "viewpoint",
    "구조": "structure", "structure": "structure", "構成": "structure",
}
_TITLE_LINE = {"ko": "제목", "ko_en": "제목", "en": "Title", "ja": "タイトル"}
TXT_TAIL = write_in_line("ko")
_HANGUL_RE = re.compile("[\uAC00-\uD7AF]")
_KANA_RE = re.compile("[\u3040-\u30FF]")
_LATIN_RE = re.compile("[A-Za-z]")


def detect_language(text: str) -> str:
    """가사에 가장 많이 쓰인 글자로 언어를 고른다: 한글 → ko, 가나 → ja, 그 외 영문 → en."""
    body = "\n".join(ln for ln in text.splitlines() if not _MARKER_RE.match(ln))
    hangul, kana, latin = (len(r.findall(body)) for r in (_HANGUL_RE, _KANA_RE, _LATIN_RE))
    # 한글·가나는 1글자=1음절, 영문은 단어 하나가 여러 글자 → 영문 글자 수는 4로 나눠 비교
    if hangul >= max(kana, 1) and hangul * 4 >= latin:
        # 한글이 주인데 영어 단어도 꽤 섞였으면(영어 단어 ≈ 영문 글자/4 가 한글 글자의 15% 이상) K-pop 스타일
        return "ko_en" if latin / 4 >= hangul * 0.15 else "ko"
    if kana > hangul and kana * 4 >= latin:
        return "ja"
    return "en" if latin else "ko"


def decode_text(data: bytes) -> str:
    """UTF-8 → (실패 시) 윈도우 한글 CP949 순서로 읽는다."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValidationError("텍스트 인코딩을 읽을 수 없습니다. UTF-8 로 저장해 주세요.")


def txt_to_record(filename: str, text: str) -> dict[str, Any]:
    """가사만 적힌 .txt 를 MLX chat 예제 1개로 바꾼다.

    - 제목: 파일 안의 '제목: ...' 줄, 없으면 파일 이름
    - 맨 위에 '주제:/분위기:/화자:/시점:/구조:' 줄이 있으면 요청(user)으로 사용 (선택)
    - [Verse 1] 같은 구조 표시가 있으면 '구조:' 를 자동으로 만든다
    요청(user) 형식은 곡 만들기 화면이 실제로 보내는 형식과 같다 → 학습한 모델이 그 요청을 잘 따른다.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")
    header: dict[str, str] = {}
    body: list[str] = []
    in_header = True
    for line in lines:
        stripped = line.strip()
        key, sep, value = stripped.replace("：", ":").partition(":")
        canon = _HEADER_KEYS.get(key.strip().lower()) if sep else None
        if canon == "title" and "title" not in header:
            header["title"] = value.strip()
            continue
        if in_header and canon:
            header[canon] = value.strip()
            continue
        if in_header and not stripped:
            continue
        in_header = False
        body.append(line.rstrip())
    lyrics = "\n".join(body).strip()
    if not lyrics:
        raise ValidationError(f"{filename}: 가사가 비어 있습니다.")
    title = header.get("title") or Path(filename).stem.strip() or "무제"
    lang = detect_language(lyrics)                 # 팝송(영어)·일본어 가사면 그 언어의 지시·형식으로 학습
    L = prompt_labels(lang)
    markers = [m.group(1).strip() for ln in body if (m := _MARKER_RE.match(ln))]
    user = [f"{L['topic']}: {header.get('topic') or title}"]
    for key in ("speaker", "mood", "viewpoint"):
        if header.get(key):
            user.append(f"{L[key]}: {header[key]}")
    structure = header.get("structure") or ", ".join(markers)
    if structure:
        user.append(f"{L['structure']}: {structure}")
    user.append(write_in_line(lang))
    return {"messages": [
        {"role": "system", "content": lyrics_system(lang)},   # 생성할 때와 똑같은 system 지시로 학습
        {"role": "user", "content": "\n".join(user)},
        {"role": "assistant", "content": f"{_TITLE_LINE[lang]}: {title}\n{lyrics}"},
    ]}


def txt_files_to_jsonl(files: list[tuple[str, str]]) -> str:
    rows = [json.dumps(txt_to_record(name, text), ensure_ascii=False) for name, text in files]
    return "\n".join(rows) + "\n"


class LyricsTrainConfig(BaseModel):
    """가사 LoRA 학습 설정 (mlx_lm.lora). 비운 값(None)은 mlx_lm 기본값을 쓴다."""

    model_name: str = Field(min_length=1, max_length=80)
    export_dir: str = ""
    iters: int = Field(default=200, ge=1, le=100_000)
    batch_size: int = Field(default=1, ge=1, le=64)
    learning_rate: float | None = Field(default=None, gt=0, le=1)
    num_layers: int | None = Field(default=None, ge=1, le=64)
    max_seq_length: int | None = Field(default=None, ge=128, le=32_768)
    save_every: int | None = Field(default=None, ge=1)


class LyricsDatasetBuilder:
    """카테고리 없이 만드는 가사 데이터셋 + 결과 폴더 규칙.

    - 데이터셋: workspace/library/lyrics/datasets/vNNN/{train,valid}.jsonl
    - 결과(어댑터): <결과 저장 폴더 또는 workspace/library/lyrics/adapters>/<모델 이름>
    """

    def __init__(self, categories: CategoryService, datasets: DatasetRepository, library_dir: Path) -> None:
        self.categories = categories
        self.datasets = datasets
        self.library_dir = Path(library_dir)

    @property
    def guard(self):  # type: ignore[no-untyped-def]
        return self.categories.workspace.guard

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

    def list(self) -> list[Dataset]:
        """가사 데이터셋 전체 (예전에 카테고리 안에서 만든 것도 포함)."""
        return self.datasets.list_kind(DatasetKind.LYRICS)

    def get(self, dataset_id: str) -> Dataset:
        ds = self.datasets.get(dataset_id)
        if ds is None or ds.kind != DatasetKind.LYRICS:
            raise ValidationError("가사 데이터셋을 찾을 수 없습니다.", detail=dataset_id)
        return ds

    @staticmethod
    def label(ds: Dataset) -> str:
        return f"{'L' if ds.category_id is None else 'C'}-v{ds.version:03d}"

    def build(self, jsonl_text: str, valid_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, JsonlReport]:
        records, report = self.parse_jsonl(jsonl_text)
        for rec in records:   # system 이 없는 예제에는 생성 때 쓰는 system 지시를 넣어 학습/생성 조건을 맞춘다
            if not any(m.get("role") == "system" for m in rec["messages"]):
                answer = next((m["content"] for m in rec["messages"] if m.get("role") == "assistant"), "")
                rec["messages"].insert(0, {"role": "system", "content": lyrics_system(detect_language(answer))})
        if report.errors:
            raise ValidationError("JSONL 검증 실패", detail=report.errors[:20])
        if len(records) < 2:
            raise ValidationError("train/valid 분할을 위해 최소 2개 예제가 필요합니다.")
        rng = random.Random(seed)
        rng.shuffle(records)
        n_valid = max(1, int(len(records) * valid_ratio))
        valid, train = records[:n_valid], records[n_valid:]
        root = self.guard.child(self.library_dir, "datasets")
        # DB 를 지워도 예전 vNNN 폴더가 남아 있을 수 있다 → 덮어쓰지 않고 빈 번호로 건너뛴다
        version = self.datasets.next_version(None, DatasetKind.LYRICS)
        while (root / f"v{version:03d}").exists():
            version += 1
        out = root / f"v{version:03d}"
        dump = lambda rows: "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"  # noqa: E731
        atomic_write_text(out / "train.jsonl", dump(train))
        atomic_write_text(out / "valid.jsonl", dump(valid))
        dataset = Dataset(category_id=None, kind=DatasetKind.LYRICS, version=version, status="ready",
                          source_path=str(out), rights_status="confirmed")
        self.datasets.add(dataset)
        return dataset, report

    # --- 결과 저장 위치 ---
    def resolve_export_dir(self, model_name: str, export_dir: str = "") -> Path:
        """<결과 저장 폴더>/<모델 이름>. 이미 있는 폴더·보호 경로는 거부 (덮어쓰지 않음)."""
        name = safe_filename(model_name.strip())
        if not name:
            raise ValidationError("결과 모델 이름을 입력하세요.")
        if export_dir.strip():
            base = Path(export_dir).expanduser().resolve()
            if not base.is_dir():
                raise ValidationError("결과 저장 폴더가 없습니다.", detail=str(base))
        else:
            base = self.library_dir / "adapters"
        target = base / name
        for protected in self.guard.protected:
            if target == protected or protected in target.parents:
                raise ValidationError("보호된 기존 폴더(.venv 등) 안에는 저장할 수 없습니다.", detail=str(target))
        if target.exists():
            raise ValidationError(f"'{target}' 폴더가 이미 있습니다. 덮어쓰지 않습니다. 모델 이름을 바꾸세요.")
        return target


class LyricsTrainHandler(JobHandler):
    kind = JobKind.LYRICS_TRAIN

    def __init__(self, engine: LyricsEnginePort, builder: LyricsDatasetBuilder, base_model: str) -> None:
        self.engine = engine
        self.builder = builder
        self.categories = builder.categories
        self.base_model = base_model

    def validate(self, params: dict[str, Any]) -> None:
        if not params.get("dataset_id"):
            raise ValidationError("데이터셋을 선택하세요.")
        ds = self.builder.get(params["dataset_id"])
        if not (Path(ds.source_path) / "train.jsonl").is_file():
            raise ValidationError("데이터셋 폴더에 train.jsonl 이 없습니다.", detail=ds.source_path)
        cfg = LyricsTrainConfig(**params.get("config", {}))
        self.builder.resolve_export_dir(cfg.model_name, cfg.export_dir)

    def describe(self, params: dict[str, Any]) -> str:
        return "Qwen 가사 LoRA 학습"

    async def run(self, ctx: JobContext) -> dict[str, Any]:
        ds = self.builder.get(ctx.params["dataset_id"])
        cfg = LyricsTrainConfig(**ctx.params.get("config", {}))
        adapter_dir = self.builder.resolve_export_dir(cfg.model_name, cfg.export_dir)
        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        spec = LoraTrainSpec(model=ctx.params.get("base_model") or self.base_model, data_dir=Path(ds.source_path),
                             adapter_path=adapter_dir, iters=cfg.iters, batch_size=cfg.batch_size,
                             learning_rate=cfg.learning_rate, max_seq_length=cfg.max_seq_length,
                             num_layers=cfg.num_layers, save_every=cfg.save_every)
        ctx.log(f"학습 시작: 데이터셋 {LyricsDatasetBuilder.label(ds)}, iters={cfg.iters}, 결과={adapter_dir}")
        losses: list[dict[str, Any]] = []

        def on_progress(tp: TrainProgress) -> None:
            ctx.check_cancelled()
            if tp.step is not None:
                losses.append(tp.model_dump(exclude={"line"}))
                if tp.total:
                    ctx.progress(min(tp.step / tp.total, 1.0))
            if tp.line:
                ctx.log(tp.line, "debug" if tp.step is None else "info")

        handle = await self.engine.start_training(spec, on_progress)
        try:
            code = await handle.wait()
        except BaseException:
            await handle.cancel()
            raise
        if code != 0:
            raise ValidationError(f"mlx_lm.lora 가 오류로 끝났습니다 (종료 코드 {code}). 위 로그를 확인하세요.",
                                  detail=str(adapter_dir))
        ctx.set_status(JobStatus.EXPORTING)            # 상태 순서: running → exporting → validating_adapter
        ctx.set_status(JobStatus.VALIDATING_ADAPTER)
        ok, msg = validate_adapter_dir(Engine.QWEN, adapter_dir)
        if not ok:
            raise ValidationError(f"학습은 끝났지만 어댑터 검증 실패: {msg}", detail=str(adapter_dir))
        model = self.categories.register_model(
            None, Engine.QWEN, spec.model, adapter_dir, dataset_id=ds.id,
            # export_root: 모델 삭제 시 함께 지울 '이 앱이 만든' 결과 폴더
            train_config={**spec.model_dump(mode="json"), "export_root": str(adapter_dir)},
            notes=f"앱에서 학습 (데이터셋 {LyricsDatasetBuilder.label(ds)})",
            display_name=cfg.model_name.strip(),
        )
        ctx.log(f"모델 등록: {model.label} — 곡 만들기의 '가사 모델'에서 고를 수 있습니다.")
        return {"model_id": model.id, "adapter_path": str(adapter_dir), "losses": losses[-500:]}
