"""ACE-Step REST 어댑터 (어댑터 패턴: ACE HTTP API → MusicEnginePort).

검증 규칙 (REQUIREMENTS 7.3):
- HTTP 200 과 JSON ``code == 200`` 둘 다 확인
- ``task_id`` 없으면 실패
- ``/query_result`` 의 ``result`` 는 JSON 문자열일 수 있으므로 decode 후 검증
- API 키는 Authorization 헤더로만 전송하고 로그에 남기지 않는다
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import httpx

from app.adapters.ace.capabilities import AceCapabilityRegistry
from app.adapters.ports import (
    CapabilityReport,
    EngineHealth,
    GeneratedAudio,
    LoraStatus,
    MusicEnginePort,
    MusicGenerationRequest,
    TaskResult,
    TaskState,
)
from app.core.errors import EngineResponseError, EngineUnavailableError, ValidationError
from app.utils.fileio import atomic_write_bytes

log = logging.getLogger(__name__)

ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".flac", ".opus", ".aac", ".ogg", ".m4a"}


def unwrap_envelope(resp: httpx.Response) -> Any:
    """ACE 응답 envelope ``{data, code, error, timestamp, extra}`` 검증 후 data 반환."""
    if resp.status_code != 200:
        detail = resp.text[:500]
        raise EngineResponseError(f"ACE 서버 HTTP {resp.status_code}", detail=detail)
    try:
        body = resp.json()
    except ValueError as exc:
        raise EngineResponseError("ACE 응답이 JSON 이 아닙니다.", detail=resp.text[:200]) from exc
    if not isinstance(body, dict):
        raise EngineResponseError(detail=f"envelope 형식 아님: {type(body).__name__}")
    if "code" in body and body.get("code") != 200:
        raise EngineResponseError(f"ACE 오류 code={body.get('code')}", detail=body.get("error"))
    if body.get("error"):
        raise EngineResponseError("ACE 가 오류를 반환했습니다.", detail=body.get("error"))
    return body.get("data", body)


def parse_query_item(item: dict[str, Any]) -> TaskResult:
    """``/query_result`` 의 data 원소 하나를 ``TaskResult`` 로 변환."""
    task_id = str(item.get("task_id") or "")
    if not task_id:
        raise EngineResponseError(detail="query_result 항목에 task_id 없음")
    try:
        state = TaskState(int(item.get("status", 0)))
    except (TypeError, ValueError) as exc:
        raise EngineResponseError(detail=f"알 수 없는 status: {item.get('status')}") from exc

    raw_result = item.get("result")
    parsed: Any = raw_result
    if isinstance(raw_result, str) and raw_result.strip():
        try:
            parsed = json.loads(raw_result)
        except json.JSONDecodeError:
            parsed = raw_result  # 실패 메시지 문자열일 수 있음

    audios: list[GeneratedAudio] = []
    error = ""
    if isinstance(parsed, dict):
        parsed = [parsed]
    if isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            if entry.get("file"):
                audios.append(
                    GeneratedAudio(
                        file=str(entry["file"]),
                        seed=str(entry.get("seed_value", "")),
                        metas=entry.get("metas") or {},
                        dit_model=str(entry.get("dit_model", "")),
                        lm_model=str(entry.get("lm_model", "")),
                        raw=entry,
                    )
                )
            if entry.get("error"):
                error = str(entry["error"])
    elif isinstance(parsed, str) and state == TaskState.FAILED:
        error = parsed

    if state == TaskState.SUCCEEDED and not audios:
        # 성공이라고 했지만 파일이 없으면 성공으로 취급하지 않는다
        return TaskResult(task_id=task_id, state=TaskState.FAILED, error="성공 응답이지만 음원 파일이 없습니다.",
                          raw=item)
    return TaskResult(task_id=task_id, state=state, audios=audios, error=error, raw=item)


def audio_server_path(file_ref: str) -> str:
    """result.file 이 '/v1/audio?path=...' URL 이거나 서버 경로일 수 있어 둘 다 처리."""
    if file_ref.startswith("/v1/audio") or file_ref.startswith("http"):
        query = parse_qs(urlparse(file_ref).query)
        values = query.get("path")
        if not values:
            raise EngineResponseError(detail=f"오디오 URL 에 path 없음: {file_ref}")
        return values[0]
    return file_ref


class AceRestAdapter(MusicEnginePort):
    name = "ace"

    def __init__(
        self,
        base_url: str,
        registry: AceCapabilityRegistry,
        *,
        api_key: str = "",
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.registry = registry
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_s, headers=headers, transport=transport,
            trust_env=False,  # 로컬 루프백 전용. 시스템 프록시 경유 금지
        )
        self._capabilities_loaded = False

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.request(method, path, **kwargs)
        except httpx.ConnectError as exc:
            raise EngineUnavailableError(
                "ACE-Step 서버에 연결할 수 없습니다. 서버 실행 여부와 포트를 확인하세요.",
                detail=str(self._client.base_url),
            ) from exc
        except httpx.TimeoutException as exc:
            raise EngineUnavailableError("ACE-Step 서버 응답 시간 초과", detail=path) from exc

    async def _call(self, capability: str, payload: dict[str, Any] | None = None, *,
                    path_params: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        binding = self.registry.require(capability)
        path = binding.path
        if path_params:
            path = path.format(**{k: quote(str(v), safe="") for k, v in path_params.items()})
        if "{" in path:
            raise ValidationError("경로 파라미터가 비어 있습니다.", detail=path)
        if payload is not None:
            if binding.method.upper() == "GET":
                kwargs["params"] = payload
            else:
                kwargs["json"] = payload
        resp = await self._request(binding.method, path, **kwargs)
        return unwrap_envelope(resp)

    # ------------------------------------------------------------------
    async def health(self) -> EngineHealth:
        try:
            data = await self._call("health")
        except EngineUnavailableError as exc:
            self.registry.mark_unavailable(exc.user_message)
            return EngineHealth(engine="ace", ok=False, message=exc.user_message)
        except Exception as exc:  # noqa: BLE001
            return EngineHealth(engine="ace", ok=False, message=f"상태 확인 실패: {exc}")
        if not self._capabilities_loaded:
            await self.capabilities(refresh=True)
        details = data if isinstance(data, dict) else {"raw": data}
        return EngineHealth(engine="ace", ok=True, message="연결됨", version=str(details.get("version", "")),
                            details=details)

    async def capabilities(self, refresh: bool = False) -> CapabilityReport:
        if self._capabilities_loaded and not refresh:
            return self.registry.report
        try:
            resp = await self._request("GET", "/openapi.json")
            if resp.status_code != 200:
                return self.registry.mark_unavailable(f"openapi.json HTTP {resp.status_code}")
            report = self.registry.apply_openapi(resp.json())
            self._capabilities_loaded = True
            return report
        except EngineUnavailableError as exc:
            return self.registry.mark_unavailable(exc.user_message)

    async def list_models(self) -> dict[str, Any]:
        return await self._call("models")

    async def stats(self) -> dict[str, Any]:
        return await self._call("stats")

    # ------------------------------------------------------------------
    async def submit_generation(self, req: MusicGenerationRequest) -> str:
        desired: dict[str, Any] = {
            "prompt": req.prompt,
            "lyrics": req.lyrics,
            "thinking": req.thinking,
            "vocal_language": req.vocal_language,
            "audio_format": req.audio_format,
            "batch_size": req.batch_size,
            "audio_duration": req.audio_duration,
            "bpm": req.bpm,
            "key_scale": req.key_scale,
            "time_signature": req.time_signature,
            "inference_steps": req.inference_steps,
            "model": req.model,
        }
        if req.seed is not None and req.seed >= 0:
            desired["use_random_seed"] = False
            desired["seed"] = req.seed
        desired.update(req.extra)
        payload, dropped = self.registry.build_payload("generate", desired)
        if dropped:
            log.warning("ACE generate: 스키마에 없는 필드 제외 %s", dropped)
        data = await self._call("generate", payload)
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not task_id:
            raise EngineResponseError("ACE 가 task_id 를 반환하지 않았습니다.", detail=data)
        return str(task_id)

    async def query_results(self, task_ids: list[str]) -> list[TaskResult]:
        payload, _ = self.registry.build_payload("query_result", {"task_id_list": task_ids})
        data = await self._call("query_result", payload)
        if not isinstance(data, list):
            raise EngineResponseError(detail=f"query_result data 가 list 가 아님: {type(data).__name__}")
        return [parse_query_item(item) for item in data if isinstance(item, dict)]

    async def download_audio(self, audio: GeneratedAudio, dest: Path) -> Path:
        server_path = audio_server_path(audio.file)
        ext = Path(server_path).suffix.lower()
        if ext not in ALLOWED_AUDIO_EXT:
            raise ValidationError("지원하지 않는 오디오 형식입니다.", detail=server_path)
        binding = self.registry.require("audio")
        resp = await self._request(binding.method, binding.path, params={"path": server_path})
        if resp.status_code != 200 or not resp.content:
            raise EngineResponseError(f"음원 다운로드 실패 HTTP {resp.status_code}", detail=server_path)
        ctype = resp.headers.get("content-type", "")
        if "json" in ctype:
            raise EngineResponseError("음원 대신 JSON 응답을 받았습니다.", detail=resp.text[:200])
        dest = dest.with_suffix(ext)
        return atomic_write_bytes(dest, resp.content)

    # ------------------------------------------------------------------
    # LoRA: 문서에 라우트가 없으므로 바인딩이 채워져야만 동작한다.
    # LoRA 올리기/내리기는 음악 모델 가중치를 GPU 에서 다시 옮기는 작업이라, 메모리가 빠듯하면(다른 AI 작업 직후 등)
    # 수십 초 이상 걸릴 수 있다. 일반 요청(30초)보다 넉넉하게 기다린다.
    LORA_TIMEOUT_S = 300.0

    async def load_lora(self, adapter_path: Path) -> LoraStatus:
        payload, _ = self.registry.build_payload("lora_load", {"adapter_path": str(adapter_path)})
        data = await self._call("lora_load", payload, timeout=self.LORA_TIMEOUT_S)
        status = await self._status_after(data)
        # 로드는 됐지만 '사용' 이 꺼져 있으면 toggle 로 켠다 (설치본에 /v1/lora/toggle 존재)
        if status.enabled is False and self.registry.report.is_usable("lora_toggle"):
            toggle, _ = self.registry.build_payload("lora_toggle", {"use_lora": True})
            await self._call("lora_toggle", toggle, timeout=self.LORA_TIMEOUT_S)
            status = await self._status_after(data)
        status.adapter_path = status.adapter_path or str(adapter_path)
        return status

    async def unload_lora(self) -> LoraStatus:
        binding = self.registry.require("lora_unload")
        payload: dict[str, Any] | None = None
        if binding.method.upper() != "GET":
            payload, _ = self.registry.build_payload("lora_unload", {})
        data = await self._call("lora_unload", payload, timeout=self.LORA_TIMEOUT_S)
        return await self._status_after(data)

    async def lora_status(self) -> LoraStatus:
        if not self.registry.report.is_usable("lora_status"):
            return LoraStatus(loaded=None, enabled=None, verified_by="", raw="lora_status 라우트 미지원")
        data = await self._call("lora_status")
        return self._parse_lora_status(data, verified_by="api")

    async def _status_after(self, response_data: Any) -> LoraStatus:
        if self.registry.report.is_usable("lora_status"):
            return await self.lora_status()
        # 상태 라우트가 없으면 '확인 불가' 로 둔다 (성공으로 단정하지 않음)
        return LoraStatus(loaded=None, enabled=None, verified_by="assumed", raw=response_data)

    @staticmethod
    def _parse_lora_status(data: Any, verified_by: str) -> LoraStatus:
        if not isinstance(data, dict):
            return LoraStatus(verified_by=verified_by, raw=data)

        def pick(*keys: str) -> Any:
            for k in keys:
                if k in data:
                    return data[k]
            return None

        loaded = pick("loaded", "lora_loaded", "is_loaded")
        enabled = pick("enabled", "use_lora", "lora_enabled")
        path = pick("adapter_path", "lora_path", "path")
        return LoraStatus(
            loaded=bool(loaded) if loaded is not None else None,
            enabled=bool(enabled) if enabled is not None else None,
            adapter_path=str(path) if path else None,
            verified_by=verified_by,
            raw=data,
        )

    async def invoke(self, capability: str, payload: dict[str, Any] | None = None, *,
                     path_params: dict[str, Any] | None = None, timeout: float | None = None) -> Any:
        """데이터셋/전처리/학습 라우트를 Capability 이름으로 호출. 본문은 설치본 스키마로 필터링."""
        binding = self.registry.require(capability)
        extra: dict[str, Any] = {"timeout": timeout} if timeout else {}
        if binding.method.upper() == "GET":
            return await self._call(capability, None, path_params=path_params, **extra)
        body, dropped = self.registry.build_payload(capability, payload or {})
        if dropped:
            log.warning("ACE %s: 스키마에 없는 필드 제외 %s", capability, dropped)
        return await self._call(capability, body, path_params=path_params, **extra)
