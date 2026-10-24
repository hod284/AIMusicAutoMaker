"""Mock 가사 엔진. Qwen 없이 UI/파이프라인을 개발할 때 사용. 출력에 [MOCK] 표시."""

from __future__ import annotations

import asyncio
import json

from app.adapters.ports import EngineHealth, LLMRequest, LLMResult, LyricsEnginePort

MOCK_LYRICS = """[Verse 1]
비가 내리는 밤 창가에 앉아
너의 이름을 조용히 불러봐
[Chorus]
돌아와 줘 그날의 우리로
아직 나는 여기 서 있어
[Verse 2]
젖은 거리 위 남겨진 발자국
지워지지 않는 너의 목소리
[Chorus]
돌아와 줘 그날의 우리로
아직 나는 여기 서 있어"""


class MockLyricsAdapter(LyricsEnginePort):
    name = "qwen-mock"
    is_mock = True

    def __init__(self, latency_s: float = 0.05) -> None:
        self.latency_s = latency_s
        self.requests: list[LLMRequest] = []
        self.next_text: str | None = None  # 테스트에서 응답 주입

    async def health(self) -> EngineHealth:
        return EngineHealth(engine="qwen", ok=True, message="Mock 엔진 (실제 Qwen 아님)", is_mock=True)

    async def generate(self, req: LLMRequest) -> LLMResult:
        await asyncio.sleep(self.latency_s)
        self.requests.append(req)
        if self.next_text is not None:
            text = self.next_text
        elif "JSON" in req.system or "json" in req.user.lower():
            text = json.dumps({
                "prompt": "[MOCK] emotional korean rock ballad, warm piano, electric guitar, expressive male vocals",
                "genre": "rock ballad", "bpm": 76, "vocal_language": "ko",
                "instruments": ["piano", "electric guitar", "drums"], "mood": ["nostalgic"],
                "structure_hint": "verse-chorus-verse-chorus",
            }, ensure_ascii=False)
        else:
            text = MOCK_LYRICS
        return LLMResult(text=text, model="mock-qwen", adapter_path=req.adapter_path, is_mock=True)
