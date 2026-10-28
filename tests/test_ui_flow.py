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
    monkeypatch.setenv("STUDIO_VOICE_ENGINE", "mock")
    monkeypatch.setenv("STUDIO_VOICE_ROOT", str(tmp_path / "voice-ai"))
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


async def _marked(user: User, marker: str, timeout: float = 5.0):  # type: ignore[no-untyped-def]
    """화면에 보이게 된 표식 요소 (bind_visibility 는 잠깐 뒤에 반영된다)."""
    for _ in range(int(timeout / 0.05)):
        try:
            return next(iter(user.find(marker=marker).elements))
        except AssertionError:
            await asyncio.sleep(0.05)
    return next(iter(user.find(marker=marker).elements))


def _click_item(user: User, text: str) -> None:
    from nicegui.testing.user_interaction import UserInteraction

    item = next(e for e in user.find(ui.item).elements
                if any(text in getattr(x, "text", "") for x in e.descendants()))
    UserInteraction(user, {item}, None).click()


async def test_pages_render(user: User) -> None:
    for path, text in [("/", "ACE-Step (음악)"), ("/lyrics", "AI로 가사 만들기"), ("/create", "가사집에 가사가 없습니다"),
                       ("/library", "곡이 없습니다"), ("/jobs", "로그"), ("/training", "가사 LoRA 학습"),
                       ("/music-training", "음악 LoRA 학습"), ("/settings", "현재 설정"),
                       ("/voice-models", "녹음 올리기"), ("/voice-models?tab=models", "목소리 모델 전체"),
                       ("/vocal-convert", "바꿀 목소리")]:
        await user.open(path)
        await user.should_see(text)


async def test_lyrics_book_then_full_song_flow(user: User) -> None:
    """가사집에서 AI 작사 → 곡 만들기에서 가사 고르기 → 프롬프트 → 음악 생성 → 만든 곡."""
    await user.open("/lyrics")
    user.find("AI로 가사 만들기").click()
    _el(user, ui.input, "주제 *").set_value("비 오는 밤의 그리움")
    user.find(marker="ai-lyrics-go").click()       # (같은 이름의 '만들기' 버튼이 카테고리 탭에도 있어 표식으로 찾음)
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


async def test_voice_model_then_convert_song_flow(user: User) -> None:
    """목소리 모델 학습 → 보컬 변환(만든 곡) → 새 곡이 만든 곡 목록에 나옴. 실행 중에는 입력 잠김."""
    from app.adapters.voice.mock_adapter import write_tone
    from app.core.container import ServiceContainer
    from app.services.music_service import MusicGenerationSpec

    c = ServiceContainer()
    c.voice.stage_upload("my_song.wav", write_tone(c.settings.data_dir / "t.wav", 3.0).read_bytes())
    (song,) = await c.music.generate(MusicGenerationSpec(lyrics="[Verse]\n가사", prompt="ballad", use_lora=False,
                                                         title="빗소리"))

    await user.open("/voice-models")
    await user.should_see("my_song.wav")
    _el(user, ui.input, "목소리 모델 이름 *").set_value("내 목소리")
    user.find(marker="voice-train").click()
    await _wait_for(user, "목소리 모델 '내 목소리' 를 만들었습니다", timeout=30)
    (model,) = c.voice.voice_models()

    await user.open("/library")
    await user.should_see("빗소리")
    c.voice_engine.latency_s = 0.5                               # 실행 중 잠금을 보기 위해 천천히
    await user.open(f"/vocal-convert?song={song.id}")
    await user.should_see("원곡: 빗소리")
    assert _el(user, ui.select, "목소리 모델 *").value == model.id     # 모델이 하나면 바로 골라짐
    user.find(marker="voice-convert").click()
    await _wait_for(user, "보컬 변환 중…")
    assert not _el(user, ui.input, "제목").enabled
    await _wait_for(user, "보컬 변환 완료", timeout=30)
    await user.should_see("빗소리 · 내 목소리")

    await user.open("/library")
    await user.should_see("빗소리 · 내 목소리")
    await user.should_see("목소리 · 내 목소리")


async def test_vocal_convert_external_file_with_base_model(user: User) -> None:
    """외부 곡 파일 올리기 + 기본 모델(참고 녹음 올리기) → 제목·저장 폴더로 저장."""
    from nicegui.elements.upload_files import SmallFileUpload

    from app.adapters.voice.mock_adapter import write_tone
    from app.core.container import ServiceContainer

    c = ServiceContainer()
    data = write_tone(c.settings.data_dir / "x.wav", 3.0).read_bytes()
    out = c.settings.data_dir / "my_out"
    out.mkdir()

    await user.open("/vocal-convert")
    next(t for t in user.find(ui.toggle).elements if "file" in t.options).set_value("file")   # 원곡: 외부 파일
    await (await _marked(user, "upload-source")).handle_uploads(
        [SmallFileUpload(name="외부곡.wav", content_type="audio/wav", _data=data)])
    await _wait_for(user, "원곡 파일: 외부곡.wav")
    toggles = list(user.find(ui.toggle).elements)
    next(t for t in toggles if "reference" in t.options).set_value("reference")
    await (await _marked(user, "upload-reference")).handle_uploads(
        [SmallFileUpload(name="목소리.wav", content_type="audio/wav", _data=data)])
    await _wait_for(user, "참고 녹음: 목소리.wav")
    _el(user, ui.input, "제목").set_value("내가 부른 외부곡")
    _el(user, ui.input, "저장 폴더").set_value(str(out))
    user.find(marker="voice-convert").click()
    await _wait_for(user, "보컬 변환 완료", timeout=30)
    assert (out / "내가_부른_외부곡.mp3").is_file()
    await user.open("/library")
    await user.should_see("내가 부른 외부곡")


async def test_music_training_page_step_buttons_follow_dataset_state(user: User) -> None:
    from app.core.container import ServiceContainer
    from app.domain.entities import Dataset
    from app.domain.enums import DatasetKind

    c = ServiceContainer()
    c.music_datasets.datasets.add(Dataset(category_id=None, kind=DatasetKind.MUSIC, version=1, status="labeled",
                                          source_path=str(c.settings.data_dir)))
    await user.open("/music-training")
    enabled = {m: not next(iter(user.find(marker=m).elements)).props.get("disable")
               and next(iter(user.find(marker=m).elements)).enabled
               for m in ("step-music_scan", "step-music_label", "step-music_preprocess", "step-music_train")}
    assert enabled == {"step-music_scan": True, "step-music_label": True, "step-music_preprocess": True,
                       "step-music_train": False}                       # 라벨링까지 했으면 전처리까지 열림


async def test_rvc_option_selects_saved_model(user: User) -> None:
    from app.core.container import ServiceContainer
    c = ServiceContainer()
    await user.open('/vocal-convert')
    toggle = next(e for e in user.find(ui.toggle).elements if 'rvc' in e.options)
    toggle.set_value('rvc')
    await user.should_see('RVC 모델 넣기 (.pth)')
    c.voice.store_rvc('singer.pth', b'model')
    await user.open('/vocal-convert')
    toggle = next(e for e in user.find(ui.toggle).elements if 'rvc' in e.options)
    toggle.set_value('rvc')
    picker = _el(user, ui.select, '업로드한 모델 중 선택 (.pth) *')
    assert any('singer.pth' in n for n in picker.options)
    picker.set_value(next(iter(picker.options)))
    await asyncio.sleep(0.2)  # NiceGUI's visibility binding propagation.
    await user.should_not_see("변환 스텝")
