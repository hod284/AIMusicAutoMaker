"""NiceGUI 사용자 시뮬레이션: 실제 View ↔ ViewModel ↔ EventBus 갱신 확인 (Mock 엔진).

브라우저 없이 NiceGUI 의 ``user`` 픽스처로 페이지를 열고 버튼을 클릭한다.
"""

from __future__ import annotations

import asyncio

import pytest
from nicegui import ui
from nicegui.testing import User


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDIO_MUSIC_ENGINE", "mock")
    monkeypatch.setenv("STUDIO_LYRICS_ENGINE", "mock")
    monkeypatch.setenv("STUDIO_WORKSPACE_DIR", str(tmp_path / "ws"))
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("STUDIO_ACE_ROOT", str(tmp_path / "ace"))
    monkeypatch.setenv("ACE_POLL_INTERVAL_S", "0.01")


async def _wait_for(user: User, text: str, timeout: float = 15.0) -> None:
    for _ in range(int(timeout / 0.05)):
        try:
            await user.should_see(text, retries=1)
            return
        except AssertionError:
            await asyncio.sleep(0.05)
    await user.should_see(text)


def _el(user: User, kind, label: str):
    return next(e for e in user.find(kind).elements if e.props.get("label") == label)


def _click_item(user: User, text: str) -> None:
    from nicegui.testing.user_interaction import UserInteraction

    item = next(e for e in user.find(ui.item).elements
                if any(text in getattr(x, "text", "") for x in e.descendants()))
    UserInteraction(user, {item}, None).click()


async def test_pages_render(user: User) -> None:
    for path, text in [("/", "ACE-Step (음악)"), ("/lyrics", "AI로 가사 만들기"), ("/create", "가사집에 가사가 없습니다"),
                       ("/library", "곡이 없습니다"), ("/jobs", "로그"), ("/training", "가사 LoRA 학습"),
                       ("/music-training", "음악 LoRA 학습"), ("/settings", "현재 설정")]:
        await user.open(path)
        await user.should_see(text)


async def test_lyrics_book_then_full_song_flow(user: User) -> None:
    """가사집에서 AI 작사 → 곡 만들기에서 가사 고르기 → 프롬프트 → 음악 생성 → 만든 곡."""
    await user.open("/lyrics")
    user.find("AI로 가사 만들기").click()
    _el(user, ui.input, "주제 *").set_value("비 오는 밤의 그리움")
    btn = next(b for b in user.find(ui.button).elements if b.text == "만들기")
    from nicegui.testing.user_interaction import UserInteraction
    UserInteraction(user, {btn}, None).click()
    await _wait_for(user, "가사 편집", timeout=30)      # 만든 가사가 가사집에 저장되고 에디터에 열림

    await user.open("/create")
    await user.should_see("1 / 1개")
    from app.core.container import ServiceContainer
    (draft,), _ = ServiceContainer().lyrics_book.list()
    _click_item(user, draft.title or "무제")
    await _wait_for(user, draft.lyrics.splitlines()[0])

    user.find(marker="gen-prompt").click()
    await _wait_for(user, "음악 프롬프트 생성 완료")

    user.find(marker="gen-music").click()
    await _wait_for(user, "음원 1개 생성 완료")
    await user.should_see("MOCK")                    # Mock 결과는 반드시 표시

    await user.open("/library")
    await user.should_see("MOCK")


async def test_create_song_locks_while_generating_and_resets(user: User) -> None:
    """생성 중: 버튼 비활성+로딩, 입력 잠금 / 다른 화면 다녀와도 그대로 / 끝나면 처음 화면."""
    from app.core.container import ServiceContainer

    c = ServiceContainer()
    c.lyrics_book.create("빗소리", "[Verse 1]\n비가 내려\n[Chorus]\n너를 불러")
    c.music_engine.polls_until_done = 10**9           # 끝나지 않는 생성

    await user.open("/create")
    _click_item(user, "빗소리")
    await _wait_for(user, "너를 불러")
    prompt = "음악 프롬프트 (영어 권장)"
    _el(user, ui.textarea, prompt).set_value("husky male vocal, piano")
    _el(user, ui.number, "Seed (빈칸=랜덤)").set_value(4321)
    _el(user, ui.input, "곡 제목").set_value("테스트곡")

    user.find(marker="gen-music").click()
    await _wait_for(user, "음악 생성 중…")
    assert not _el(user, ui.textarea, prompt).enabled
    assert not _el(user, ui.input, "곡 제목").enabled
    assert not _el(user, ui.select, "음악 모델 *").enabled
    assert all(e.props.get("disable") for e in user.find(ui.item).elements)

    await user.open("/")
    await user.should_see("대시보드")
    await user.open("/create")
    await _wait_for(user, "음악 생성 중…")
    assert _el(user, ui.textarea, prompt).value == "husky male vocal, piano"
    assert _el(user, ui.number, "Seed (빈칸=랜덤)").value == 4321
    assert _el(user, ui.input, "곡 제목").value == "테스트곡"
    assert not _el(user, ui.textarea, prompt).enabled
    await user.should_see("너를 불러")

    c.music_engine.polls_until_done = 1
    await _wait_for(user, "음원 1개 생성 완료")
    await _wait_for(user, "왼쪽에서 가사를 고르면")
    assert _el(user, ui.textarea, prompt).value == "" and _el(user, ui.textarea, prompt).enabled
    assert _el(user, ui.number, "Seed (빈칸=랜덤)").value is None
    assert _el(user, ui.input, "곡 제목").value == ""
    btn = next(iter(user.find(marker="gen-music").elements))
    assert btn.text == "음악 생성" and not btn.props.get("disable")
    await user.should_see("테스트곡")                 # 결과 카드

    await user.open("/create")
    await user.should_see("왼쪽에서 가사를 고르면")
    await user.should_not_see("음악 생성 중…")


async def test_dashboard_opens_fast_while_ace_is_busy(user: User) -> None:
    """ACE 가 음악을 만드는 중이라 /health 응답이 늦어도 대시보드는 바로 열린다 (response_timeout 경고 없음)."""
    from app.core.container import ServiceContainer

    engine = ServiceContainer().music_engine
    original = engine.health

    async def slow_health():
        await asyncio.sleep(5)
        return await original()

    engine.health = slow_health
    loop = asyncio.get_running_loop()
    start = loop.time()
    await user.open("/")
    await user.should_see("ACE-Step (음악)")
    assert loop.time() - start < 2.0
    await user.should_see("확인 중…")                # 상태는 뒤에서 확인
