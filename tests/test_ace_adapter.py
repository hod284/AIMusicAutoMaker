"""ACE REST 어댑터: envelope 검증, query_result 파싱, Capability Registry (httpx MockTransport)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.adapters.ace.capabilities import AceCapabilityRegistry
from app.adapters.ace.rest_adapter import AceRestAdapter, audio_server_path, parse_query_item, unwrap_envelope
from app.adapters.ports import CapabilityState, MusicGenerationRequest, TaskState
from app.core.config import STUDIO_ROOT
from app.core.errors import CapabilityUnsupportedError, EngineResponseError, EngineUnavailableError

BINDINGS = STUDIO_ROOT / "config" / "ace_routes.json"

OPENAPI = {
    "openapi": "3.1.0",
    "paths": {
        "/health": {"get": {}},
        "/release_task": {"post": {}},  # 본문 스키마 미선언 (멀티 콘텐츠 수동 파싱) → 문서 필드 fallback
        "/query_result": {"post": {}},
        "/v1/audio": {"get": {"parameters": [{"name": "path", "in": "query", "required": True}]}},
        "/v1/models": {"get": {}},
        "/v1/lora/load": {"post": {"requestBody": {"content": {"application/json": {
            "schema": {"$ref": "#/components/schemas/LoadLoraRequest"}}}}}},
        "/v1/training/start": {"post": {}},
    },
    "components": {"schemas": {"LoadLoraRequest": {
        "properties": {"lora_path": {"type": "string"}}, "required": ["lora_path"]}}},
}


def envelope(data, code=200, error=None):
    return {"data": data, "code": code, "error": error, "timestamp": 1, "extra": None}


def make_adapter(handler, bindings=BINDINGS) -> AceRestAdapter:
    return AceRestAdapter("http://ace.test", AceCapabilityRegistry(bindings), transport=httpx.MockTransport(handler))


def test_unwrap_envelope_checks_http_and_code():
    assert unwrap_envelope(httpx.Response(200, json=envelope({"a": 1}))) == {"a": 1}
    with pytest.raises(EngineResponseError):
        unwrap_envelope(httpx.Response(500, text="boom"))
    with pytest.raises(EngineResponseError):
        unwrap_envelope(httpx.Response(200, json=envelope(None, code=500, error="bad")))


def test_parse_query_item_decodes_json_string_result():
    item = {"task_id": "t1", "status": 1, "result": json.dumps([{
        "file": "/v1/audio?path=%2Ftmp%2Fout%2Fa.mp3", "seed_value": "42", "metas": {"bpm": 76},
        "dit_model": "acestep-v15-turbo"}])}
    r = parse_query_item(item)
    assert r.state == TaskState.SUCCEEDED
    assert r.audios[0].seed == "42"
    assert audio_server_path(r.audios[0].file) == "/tmp/out/a.mp3"


def test_parse_query_item_success_without_file_is_failure():
    r = parse_query_item({"task_id": "t1", "status": 1, "result": "[]"})
    assert r.state == TaskState.FAILED


def test_parse_query_item_requires_task_id():
    with pytest.raises(EngineResponseError):
        parse_query_item({"status": 1, "result": "[]"})


async def test_full_generation_roundtrip(tmp_path: Path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=OPENAPI)
        if request.url.path == "/health":
            return httpx.Response(200, json=envelope({"status": "ok", "version": "1.0"}))
        if request.url.path == "/release_task":
            body = json.loads(request.content)
            assert "not_a_field" not in body  # 문서에 없는 필드는 제거됨
            assert body["vocal_language"] == "ko" and body["use_random_seed"] is False and body["seed"] == 7
            return httpx.Response(200, json=envelope({"task_id": "abc", "status": "queued"}))
        if request.url.path == "/query_result":
            assert json.loads(request.content) == {"task_id_list": ["abc"]}
            return httpx.Response(200, json=envelope([{"task_id": "abc", "status": 1, "result": json.dumps(
                [{"file": "/v1/audio?path=/srv/out/abc.mp3", "seed_value": "7"}])}]))
        if request.url.path == "/v1/audio":
            assert request.url.params["path"] == "/srv/out/abc.mp3"
            return httpx.Response(200, content=b"ID3fake", headers={"content-type": "audio/mpeg"})
        return httpx.Response(404)

    ace = make_adapter(handler)
    health = await ace.health()
    assert health.ok and health.version == "1.0"
    task_id = await ace.submit_generation(MusicGenerationRequest(
        prompt="rock ballad", lyrics="[Verse]\n가사", seed=7, extra={"not_a_field": 1}))
    results = await ace.query_results([task_id])
    path = await ace.download_audio(results[0].audios[0], tmp_path / "song" / "audio")
    assert path.suffix == ".mp3" and path.read_bytes() == b"ID3fake"
    await ace.aclose()


async def test_capabilities_from_openapi_and_payload_filtering(tmp_path: Path):
    bindings = json.loads(BINDINGS.read_text())
    bindings["lora_load"] = {"method": "POST", "path": "/v1/lora/load", "field_map": {"adapter_path": "lora_path"}}
    custom = tmp_path / "routes.json"
    custom.write_text(json.dumps(bindings))
    ace = make_adapter(lambda r: httpx.Response(200, json=OPENAPI), custom)
    report = await ace.capabilities(refresh=True)
    caps = report.capabilities
    assert caps["generate"].state == CapabilityState.AVAILABLE
    assert caps["lora_load"].state == CapabilityState.AVAILABLE
    assert caps["lora_load"].request_fields == ["lora_path"]
    assert caps["stats"].state == CapabilityState.MISSING           # 바인딩은 있으나 설치본에 없음
    assert caps["lora_unload"].state == CapabilityState.MISSING     # 바인딩은 있으나 이 OpenAPI 에 없음 → 비활성
    payload, dropped = ace.registry.build_payload("lora_load", {"adapter_path": "/x", "junk": 1})
    assert payload == {"lora_path": "/x"} and dropped == ["junk"]
    with pytest.raises(CapabilityUnsupportedError):
        ace.registry.build_payload("lora_unload", {})
    # training_start: 라우트는 있으나 본문 스키마가 없으면 JSON 을 추측하지 않는다
    with pytest.raises(CapabilityUnsupportedError):
        ace.registry.build_payload("training_start", {"tensor_dir": "/t"})
    await ace.aclose()


async def test_unbound_suggestions_listed_but_not_used(tmp_path: Path):
    # 바인딩이 없는(null) 기능은 비슷한 경로를 '제안'만 하고 자동으로 쓰지 않는다
    bindings = json.loads(BINDINGS.read_text())
    bindings["lora_load"] = None
    custom = tmp_path / "routes.json"
    custom.write_text(json.dumps(bindings))
    ace = make_adapter(lambda r: httpx.Response(200, json=OPENAPI), custom)
    report = await ace.capabilities(refresh=True)
    assert "POST /v1/lora/load" in report.capabilities["lora_load"].suggestions
    assert report.capabilities["lora_load"].state == CapabilityState.UNBOUND
    with pytest.raises(CapabilityUnsupportedError):
        ace.registry.build_payload("lora_load", {"adapter_path": "/x"})
    await ace.aclose()


async def test_connection_failure_is_friendly():
    def handler(request):
        raise httpx.ConnectError("refused")

    ace = make_adapter(handler)
    health = await ace.health()
    assert not health.ok and "연결할 수 없습니다" in health.message
    with pytest.raises(EngineUnavailableError):
        await ace.submit_generation(MusicGenerationRequest(prompt="x x x", lyrics="y"))
    await ace.aclose()


async def test_missing_task_id_is_error():
    def handler(request):
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=OPENAPI)
        return httpx.Response(200, json=envelope({"status": "queued"}))

    ace = make_adapter(handler)
    await ace.capabilities(refresh=True)
    with pytest.raises(EngineResponseError):
        await ace.submit_generation(MusicGenerationRequest(prompt="abc", lyrics="y"))
    await ace.aclose()
