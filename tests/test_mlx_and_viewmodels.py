"""MLX CLI 옵션 매핑 + ViewModel (UI 없이 MVVM 동작 검증)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.adapters.mlx.process_adapter import build_train_args, parse_help_flags, parse_train_line
from app.adapters.ports import LoraTrainSpec
from app.core.errors import CapabilityUnsupportedError
from app.domain.enums import JobKind, JobStatus

HELP = """usage: mlx_lm.lora [-h] [--model MODEL] [--train] [--data DATA] [--adapter-path ADAPTER_PATH]
  --iters ITERS  --batch-size BATCH_SIZE  --learning-rate LEARNING_RATE  --save-every SAVE_EVERY"""


def test_train_args_only_use_supported_flags():
    flags = parse_help_flags(HELP)
    spec = LoraTrainSpec(model="m", data_dir=Path("/d"), adapter_path=Path("/a"), iters=50,
                         max_seq_length=2048, learning_rate=1e-5)
    args, skipped = build_train_args(spec, flags)
    assert args[:7] == ["--model", "m", "--train", "--data", "/d", "--adapter-path", "/a"]
    assert "--iters" in args and "--learning-rate" in args
    assert "--max-seq-length" in skipped and "--max-seq-length" not in args
    with pytest.raises(CapabilityUnsupportedError):
        build_train_args(spec, {"--model"})


def test_parse_train_lines():
    assert parse_train_line("Iter 10: Train loss 2.345, Learning Rate 1e-5").train_loss == 2.345
    assert parse_train_line("Iter 20: Val loss 1.9, Val took 3s").val_loss == 1.9
    assert parse_train_line("Loading model") is None


async def _until(predicate, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > end:
            raise TimeoutError
        await asyncio.sleep(0.01)


async def test_create_song_viewmodel_end_to_end(container):
    """곡 만들기: 가사집에서 가사 고르기 → AI 프롬프트 → 음악 생성 (생성 중 잠금, 끝나면 초기화)."""
    from app.viewmodels.create_song_vm import CreateSongViewModel

    lyric = container.lyrics_book.create("빗소리", "[Verse 1]\n비가 내리는 밤\n[Chorus]\n돌아와 줘")
    vm = CreateSongViewModel(container)
    changes = []
    vm.subscribe("*", lambda name, value: changes.append(name))
    assert [d.id for d in vm.drafts] == [lyric.id] and vm.draft_total == 1

    vm.select_lyric(lyric.id)
    assert vm.title == "빗소리" and vm.lyrics_text.startswith("[Verse 1]") and vm.language == "ko"
    assert "lyrics_text" in changes

    await vm.generate_prompt()
    await _until(lambda: vm.job and vm.job.status.is_terminal)
    assert vm.job.status == JobStatus.SUCCEEDED, vm.error
    assert vm.music_prompt is not None and vm.prompt_text and vm.job_logs

    vm.song_title, vm.seed, vm.duration = "비 오는 밤", 3, 30
    await vm.generate_music()
    assert vm.locked                                   # 생성 중: 입력 잠김
    vm.select_lyric(lyric.id)
    vm.set_music_model("x")
    assert "만드는 중" in vm.error and vm.music_model_id is None
    await vm.generate_music()                          # 두 번 눌러도 새 작업이 생기지 않음
    assert sum(j.kind == JobKind.MUSIC_GENERATE for j in container.job_repo.list_recent()) == 1
    await _until(lambda: vm.job.status.is_terminal)
    assert vm.job.status == JobStatus.SUCCEEDED, vm.error

    assert len(vm.songs) == 1 and vm.songs[0].title == "비 오는 밤"
    assert vm.songs[0].generation.get("seed") in (3, "3")
    # 다 만들어지면 처음 들어온 화면처럼 초기화
    assert not vm.locked and vm.draft_id is None and vm.prompt_text == "" and vm.song_title == ""
    assert vm.seed is None and vm.duration == 60
    vm.dispose()
    assert container.bus.subscriber_count("job.updated") == 0


async def test_create_song_vm_restores_screen_while_running(container, music):
    """생성 중에 다른 화면에 다녀오면 버튼 누르기 직전 상태를 그대로 되살린다."""
    from app.viewmodels.create_song_vm import CreateSongViewModel

    lyric = container.lyrics_book.create("빗소리", "[Verse 1]\n비가 내리는 밤")
    music.polls_until_done = 10**9
    vm = CreateSongViewModel(container)
    vm.select_lyric(lyric.id)
    vm.prompt_text, vm.seed, vm.song_title = "piano ballad", 42, "테스트곡"
    await vm.generate_music()
    await _until(lambda: vm.job and vm.job.status.value == "running")
    vm.dispose()

    again = CreateSongViewModel(container)            # 화면에 다시 들어옴
    assert again.locked and again.draft_id == lyric.id and again.lyrics_text.startswith("[Verse 1]")
    assert again.prompt_text == "piano ballad" and again.seed == 42 and again.song_title == "테스트곡"
    music.polls_until_done = 1
    await _until(lambda: again.job.status.is_terminal)
    assert again.job.status == JobStatus.SUCCEEDED and again.draft_id is None and again.songs
    again.dispose()


async def test_create_song_vm_requires_lyric(container):
    from app.viewmodels.create_song_vm import CreateSongViewModel

    vm = CreateSongViewModel(container)
    await vm.generate_music()
    assert "가사집의 가사를 먼저 고르세요" in vm.error and vm.job is None


def test_lyrics_book_viewmodels(container):
    """가사집: 저장은 덮어쓰기(가사 하나 = 기록 하나), 카테고리 필터, 카테고리 편집·삭제."""
    from app.viewmodels.lyrics_book_vm import LyricCategoriesViewModel, LyricsBookViewModel

    book = LyricsBookViewModel(container)
    book.new()
    book.title, book.text = "첫 가사", "[Verse 1]\n처음"
    book.save()
    first = book.selected_id
    assert first and book.total == 1
    book.text = "[Verse 1]\n고친 가사"
    book.save()
    assert book.selected_id == first and book.total == 1          # v2 를 만들지 않고 덮어씀
    assert container.lyrics_book.get(first).lyrics.endswith("고친 가사")

    cats = LyricCategoriesViewModel(container)
    cats.create("발라드")
    assert cats.category_name == "발라드"
    cats.add(first)
    cats.load()
    assert cats.member_ids == {first}
    book.reload()
    assert book.category_names(first) == ["발라드"]
    book.set_filter(cats.category_id)
    assert [d.id for d in book.items] == [first]

    cats.delete_category()                                          # 카테고리만 지우고 가사는 남김
    assert cats.categories == [] and container.lyrics_book.get(first)
    cats.delete_lyric(first)
    book.reload()
    assert book.total == 0 and book.error == ""


async def test_dashboard_vm_reacts_to_events(container, ballad):
    from app.domain.enums import JobKind
    from app.viewmodels.dashboard_vm import DashboardViewModel

    vm = DashboardViewModel(container)
    job = container.jobs.submit(JobKind.LYRICS_GENERATE, {"category_id": ballad.id, "topic": "t"})
    await container.jobs.wait(job.id, timeout=5)
    assert vm.jobs and vm.jobs[0].id == job.id and vm.jobs[0].status == JobStatus.SUCCEEDED
    await vm.refresh_health()
    assert vm.ace_health.is_mock and vm.qwen_health.is_mock


async def test_create_song_shows_results_finished_while_away(container, music):
    """창을 내렸거나 연결이 끊긴 사이에 끝난 곡은, 화면이 다시 열리면 결과로 보여 준다 (입력은 처음 상태)."""
    from app.viewmodels.create_song_vm import CreateSongViewModel

    lyric = container.lyrics_book.create("빗소리", "[Verse 1]\n비가 내리는 밤")
    vm = CreateSongViewModel(container)
    vm.is_live = lambda: False                       # 브라우저 연결이 끊긴 상태
    vm.select_lyric(lyric.id)
    vm.song_title = "안 본 곡"
    await vm.generate_music()
    await _until(lambda: vm.job.status.is_terminal)
    vm.dispose()

    again = CreateSongViewModel(container)          # 다시 연결되어 화면이 새로 열림
    assert [s.title for s in again.songs] == ["안 본 곡"] and "완성" in again.info
    assert again.draft_id is None and not again.locked and again.prompt_text == ""
    again.dispose()
    third = CreateSongViewModel(container)          # 한 번 보여 준 결과는 다시 띄우지 않음
    assert third.songs == []
    third.dispose()

    vm = CreateSongViewModel(container)              # 화면을 보고 있는 동안 끝나면 → 다음에 들어올 땐 처음 화면
    vm.select_lyric(lyric.id)
    await vm.generate_music()
    await _until(lambda: vm.job.status.is_terminal)
    assert vm.songs and vm.draft_id is None
    vm.dispose()
    assert CreateSongViewModel(container).songs == []


def test_music_training_steps_unlock_in_order_and_lock_while_running(container, tmp_path):
    """음악 LoRA 학습: 단계는 순서대로 열리고, 한 단계가 실행 중이면 모든 단계·입력이 잠긴다."""
    from app.domain.entities import Dataset, Job
    from app.domain.enums import DatasetKind
    from app.viewmodels.music_training_vm import MusicTrainingViewModel

    ds = Dataset(category_id=None, kind=DatasetKind.MUSIC, version=1, status="scanned", source_path=str(tmp_path))
    container.music_datasets.datasets.add(ds)
    vm = MusicTrainingViewModel(container)
    assert vm.dataset_id == ds.id
    ready = {k: vm.step_ready(k) for _, k, _ in vm.STEPS}
    assert ready == {JobKind.MUSIC_SCAN: True, JobKind.MUSIC_LABEL: True,
                     JobKind.MUSIC_PREPROCESS: False, JobKind.MUSIC_TRAIN: False}     # 스캔까지 했으면 라벨링까지만

    vm.job = Job(kind=JobKind.MUSIC_LABEL, status=JobStatus.RUNNING)                  # ③ 라벨링 실행 중
    assert vm.locked and vm.running_step(JobKind.MUSIC_LABEL)
    assert not any(vm.step_ready(k) for _, k, _ in vm.STEPS)
    vm.upload("a.wav", b"RIFF....")
    assert "실행 중" in vm.error and vm.staged == []
    vm.error = ""
    vm.choose_dataset(None)
    assert vm.dataset_id == ds.id and "실행 중" in vm.error
    vm.scan()
    assert container.job_repo.list_recent() == []                                     # 새 단계가 시작되지 않음
    vm.dispose()


async def test_lyrics_training_locks_restores_and_unlocks(container, tmp_path, lyrics):
    """가사 LoRA 학습: 학습 중 잠금, 다시 들어오면 입력값 그대로, 끝나면 풀림."""
    from app.adapters.ports import TrainHandle
    from app.viewmodels.misc_vms import TrainingViewModel

    gate = asyncio.Event()

    class _Handle(TrainHandle):
        def __init__(self, spec):
            self.spec = spec

        async def wait(self):
            await gate.wait()
            self.spec.adapter_path.mkdir(parents=True, exist_ok=True)
            (self.spec.adapter_path / "adapter_config.json").write_text("{}")
            (self.spec.adapter_path / "adapters.safetensors").write_bytes(b"0")
            return 0

        async def cancel(self, grace_s=10.0):
            gate.set()

    async def start_training(spec, on_progress):
        return _Handle(spec)

    lyrics.start_training = start_training
    vm = TrainingViewModel(container)
    for i in range(2):
        vm.add_txt(f"s{i}.txt", f"[Verse 1]\n가사 {i}\n[Chorus]\n후렴 {i}".encode())
    vm.build_from_txt()
    vm.model_title, vm.form["iters"] = "감성 작사", 77
    vm.start_training()
    await _until(lambda: vm.job and vm.job.status == JobStatus.RUNNING)
    assert vm.locked
    vm.add_txt("x.txt", "[Verse]\n또".encode())
    assert vm.txt_files == [] and "학습 중" in vm.error
    vm.dispose()

    again = TrainingViewModel(container)                       # 다른 화면에 다녀옴
    assert again.locked and again.model_title == "감성 작사" and again.form["iters"] == 77
    gate.set()
    await _until(lambda: again.job.status.is_terminal)
    assert again.job.status == JobStatus.SUCCEEDED, again.error
    assert not again.locked
    again.dispose()
