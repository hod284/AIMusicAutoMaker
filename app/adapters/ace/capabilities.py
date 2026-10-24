"""ACE-Step Capability Registry (REQUIREMENTS 7.2 동적 발견 원칙).

동작:
1. ``config/ace_routes.json`` 의 바인딩(기능 → method/path)을 읽는다.
2. 실행 중인 서버의 ``/openapi.json`` 과 대조한다.
   - 바인딩 있음 + OpenAPI 에 존재 → AVAILABLE
   - 바인딩 있음 + OpenAPI 에 없음 → MISSING (버튼 비활성)
   - 바인딩 없음 → UNBOUND, 키워드로 찾은 '후보 경로'만 제안 (자동 사용 안 함)
3. 요청 본문 스키마의 필드 목록을 추출해, 페이로드에서 **스키마에 없는 키는 제거**한다.
   (추측한 JSON 을 서버로 보내지 않기 위함)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.adapters.ports import Capability, CapabilityReport, CapabilityState
from app.core.errors import CapabilityUnsupportedError, ValidationError

CAPABILITY_LABELS: dict[str, str] = {
    "health": "서버 상태",
    "models": "모델 목록",
    "stats": "서버 통계",
    "generate": "음악 생성",
    "query_result": "결과 조회",
    "audio": "음원 다운로드",
    "init_model": "모델 초기화",
    "lora_load": "LoRA 로드",
    "lora_unload": "LoRA 언로드",
    "lora_toggle": "LoRA 켜기/끄기",
    "lora_status": "LoRA 상태",
    "dataset_scan": "데이터 스캔",
    "dataset_auto_label": "자동 라벨링",
    "dataset_save": "데이터셋 저장",
    "dataset_samples": "샘플 목록",
    "dataset_sample_update": "샘플 수정",
    "auto_label_status": "라벨링 상태",
    "auto_label_status_latest": "라벨링 상태(최근)",
    "preprocess": "텐서 전처리",
    "preprocess_status": "전처리 상태",
    "preprocess_status_latest": "전처리 상태(최근)",
    "training_start": "LoRA 학습 시작",
    "training_status": "학습 상태",
    "training_stop": "학습 중지",
    "training_export": "LoRA Export",
}

# UNBOUND 기능의 후보 탐색 키워드 (제안만 하고 자동 바인딩하지 않음)
_SUGGEST_KEYWORDS: dict[str, tuple[str, ...]] = {
    "lora_load": ("lora", "load"),
    "lora_unload": ("lora", "unload"),
    "lora_toggle": ("lora", "toggle"),
    "lora_status": ("lora", "status"),
    "dataset_scan": ("dataset", "scan"),
    "dataset_auto_label": ("label",),
    "dataset_save": ("dataset", "save"),
    "preprocess": ("preprocess",),
    "preprocess_status": ("preprocess", "status"),
    "training_start": ("train", "start"),
    "training_status": ("train", "status"),
    "training_stop": ("train", "stop"),
    "training_export": ("export",),
}

# 공식 API 문서(docs/en/API.md)에 명시된 /release_task 필드.
# 설치본 OpenAPI 가 본문 스키마를 선언하지 않는 경우(멀티 콘텐츠 타입 수동 파싱)에만 사용한다.
DOCUMENTED_GENERATE_FIELDS: frozenset[str] = frozenset({
    "prompt", "caption", "lyrics", "thinking", "vocal_language", "audio_format", "sample_mode", "sample_query",
    "use_format", "model", "bpm", "key_scale", "time_signature", "audio_duration", "inference_steps",
    "guidance_scale", "use_random_seed", "seed", "batch_size", "shift", "infer_method", "timesteps", "use_adg",
    "cfg_interval_start", "cfg_interval_end", "lm_model_path", "lm_backend", "lm_temperature", "lm_cfg_scale",
    "lm_negative_prompt", "lm_top_k", "lm_top_p", "lm_repetition_penalty", "use_cot_caption",
    "use_cot_language", "constrained_decoding", "allow_lm_batch", "task_type", "ai_token",
})
DOCUMENTED_FALLBACK: dict[str, frozenset[str]] = {
    "generate": DOCUMENTED_GENERATE_FIELDS,
    "query_result": frozenset({"task_id_list", "ai_token"}),
}


class RouteBinding(BaseModel):
    method: str
    path: str
    documented: bool = False
    field_map: dict[str, str] = Field(default_factory=dict)  # 앱 필드명 → 서버 필드명


class OpenApiSpec:
    """openapi.json 을 다루는 얇은 래퍼. $ref 해석과 본문 필드 추출."""

    def __init__(self, doc: dict[str, Any]) -> None:
        self.doc = doc
        self.paths: dict[str, dict[str, Any]] = doc.get("paths", {}) or {}

    def has(self, method: str, path: str) -> bool:
        return method.lower() in {m.lower() for m in self.paths.get(path, {})}

    def all_routes(self) -> list[str]:
        return sorted(f"{m.upper()} {p}" for p, ms in self.paths.items() for m in ms if m.lower() != "parameters")

    def _deref(self, schema: dict[str, Any], depth: int = 0) -> dict[str, Any]:
        if depth > 10:
            return schema
        ref = schema.get("$ref")
        if ref and ref.startswith("#/"):
            node: Any = self.doc
            for part in ref[2:].split("/"):
                node = node.get(part, {}) if isinstance(node, dict) else {}
            return self._deref(node, depth + 1)
        for combo in ("allOf", "anyOf", "oneOf"):
            if combo in schema:
                merged: dict[str, Any] = {"properties": {}, "required": []}
                for sub in schema[combo]:
                    sub_d = self._deref(sub, depth + 1)
                    merged["properties"].update(sub_d.get("properties", {}))
                    merged["required"] += sub_d.get("required", [])
                return merged
        return schema

    def body_fields(self, method: str, path: str) -> tuple[set[str], set[str]]:
        """(전체 필드, 필수 필드). 스키마가 없으면 빈 집합."""
        op = self.paths.get(path, {}).get(method.lower(), {})
        fields: set[str] = set()
        required: set[str] = set()
        body = op.get("requestBody", {}).get("content", {})
        for content in body.values():
            schema = self._deref(content.get("schema", {}) or {})
            fields |= set(schema.get("properties", {}).keys())
            required |= set(schema.get("required", []))
        for param in op.get("parameters", []) or []:
            param = self._deref(param)
            if param.get("in") == "query":
                fields.add(param.get("name", ""))
                if param.get("required"):
                    required.add(param.get("name", ""))
        fields.discard("")
        return fields, required


class AceCapabilityRegistry:
    def __init__(self, bindings_file: Path) -> None:
        self.bindings_file = Path(bindings_file)
        self.bindings: dict[str, RouteBinding | None] = self._load_bindings()
        self.spec: OpenApiSpec | None = None
        self.report = CapabilityReport(engine="ace", source="unavailable")
        self._mark_unknown("서버 OpenAPI 를 아직 확인하지 않았습니다.")

    def _load_bindings(self) -> dict[str, RouteBinding | None]:
        result: dict[str, RouteBinding | None] = {name: None for name in CAPABILITY_LABELS}
        if not self.bindings_file.is_file():
            return result
        raw = json.loads(self.bindings_file.read_text(encoding="utf-8"))
        for name, value in raw.items():
            if name.startswith("_"):
                continue
            result[name] = RouteBinding(**value) if value else None
        return result

    def _mark_unknown(self, reason: str) -> None:
        caps = {}
        for name, label in CAPABILITY_LABELS.items():
            b = self.bindings.get(name)
            caps[name] = Capability(
                name=name, label_ko=label, state=CapabilityState.UNKNOWN if b else CapabilityState.UNBOUND,
                method=b.method if b else None, path=b.path if b else None, reason=reason,
            )
        self.report = CapabilityReport(engine="ace", capabilities=caps, source="unavailable")

    def mark_unavailable(self, reason: str) -> CapabilityReport:
        self.spec = None
        self._mark_unknown(reason)
        return self.report

    def apply_openapi(self, doc: dict[str, Any]) -> CapabilityReport:
        self.spec = OpenApiSpec(doc)
        routes = self.spec.all_routes()
        caps: dict[str, Capability] = {}
        for name, label in CAPABILITY_LABELS.items():
            binding = self.bindings.get(name)
            if binding is None:
                keywords = _SUGGEST_KEYWORDS.get(name, ())
                suggestions = [r for r in routes if keywords and all(k in r.lower() for k in keywords)]
                caps[name] = Capability(
                    name=name, label_ko=label, state=CapabilityState.UNBOUND,
                    reason="라우트가 매핑되지 않았습니다. 설치본 router 확인 후 config/ace_routes.json 에 등록하세요.",
                    suggestions=suggestions,
                )
                continue
            if self.spec.has(binding.method, binding.path):
                fields, _ = self.spec.body_fields(binding.method, binding.path)
                caps[name] = Capability(
                    name=name, label_ko=label, state=CapabilityState.AVAILABLE, method=binding.method,
                    path=binding.path, request_fields=sorted(fields),
                )
            else:
                caps[name] = Capability(
                    name=name, label_ko=label, state=CapabilityState.MISSING, method=binding.method,
                    path=binding.path, reason="설치본 OpenAPI 에 해당 라우트가 없습니다.",
                )
        self.report = CapabilityReport(engine="ace", capabilities=caps, discovered_paths=routes, source="openapi")
        return self.report

    # ------------------------------------------------------------------
    def require(self, name: str) -> RouteBinding:
        cap = self.report.capabilities.get(name)
        binding = self.bindings.get(name)
        if binding is None or cap is None:
            raise CapabilityUnsupportedError(detail=f"{name}: 라우트 미매핑")
        if cap.state == CapabilityState.MISSING:
            raise CapabilityUnsupportedError(detail=f"{name}: 설치본에 {binding.method} {binding.path} 없음")
        # UNKNOWN(openapi 미확인)이라도 문서화된 핵심 생성 라우트는 시도 허용
        if cap.state == CapabilityState.UNKNOWN and not binding.documented:
            raise CapabilityUnsupportedError(detail=f"{name}: OpenAPI 미확인")
        return binding

    def build_payload(self, name: str, desired: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """스키마에 존재하는 필드만 남긴 페이로드와 제거된 키 목록을 반환한다."""
        binding = self.require(name)
        mapped = {binding.field_map.get(k, k): v for k, v in desired.items() if v is not None}
        allowed: set[str] = set()
        required: set[str] = set()
        if self.spec is not None:
            allowed, required = self.spec.body_fields(binding.method, binding.path)
        if not allowed and not mapped:
            # 보낼 값도 없고 스키마도 없는 라우트(언로드/중지 등)는 빈 본문으로 호출
            return {}, []
        if not allowed:
            fallback = DOCUMENTED_FALLBACK.get(name)
            if fallback is None:
                raise CapabilityUnsupportedError(
                    detail=f"{name}: 요청 스키마를 확인할 수 없어 본문을 추측하지 않습니다."
                )
            allowed = set(fallback)
        payload = {k: v for k, v in mapped.items() if k in allowed}
        dropped = sorted(set(mapped) - set(payload))
        missing = sorted(required - set(payload))
        if missing:
            raise ValidationError(f"필수 필드 누락: {', '.join(missing)}", detail=name)
        return payload, dropped
