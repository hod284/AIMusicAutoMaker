"""서비스 계층: 카테고리 분리, 가사, 프롬프트 디렉터, 음악 생성, 잡 매니저, 리소스 락."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.core.errors import CapabilityUnsupportedError, PathSecurityError, ValidationError
from app.core.observer import Topics
from app.domain.enums import Engine, JobKind, JobStatus, ValidationStatus
from app.services.lyrics_analysis import analyze, clean_llm_lyrics, count_syllables
from app.services.lyrics_service import LyricsRequest
from app.services.music_service import MusicGenerationSpec
from app.services.prompt_service import (
    MusicPrompt,
    PromptDirector,
    PromptStrategy,
    TemplateStrategy,
    default_guardrails,
)
from app.services.resource_lock import ResourceLock

FINAL_LORA = ("ACE-Step-1.5", "lora_output", "final_lora", "adapter")


# ---------------- 카테고리 / 분리 ----------------
def test_categories_isolated_and_rename_keeps_folder(container, ballad, settings):
    acoustic = container.categories.create("acoustic")
    ws_a = container.categories.workspace_of(ballad.id)
    ws_b = container.categories.workspace_of(acoustic.id)
    assert ws_a.root != ws_b.root and ws_a.root.name == ballad.id
    renamed = container.categories.update(ballad.id, name="록 발라드 (개명)")
    assert renamed.slug == ballad.slug
    assert container.categories.workspace_of(ballad.id).root == ws_a.root
    dup = container.categories.create("2000년대 감성 록 발라드")
    assert dup.slug.endswith("-2")


def test_delete_category_grouping_keeps_models(container, ballad, settings):
    """모델 카테고리 삭제는 묶음만 지운다 — 모델과 파일은 그대로."""
    adapter = settings.music_root.joinpath(*FINAL_LORA)
    m = container.categories.register_model(None, Engine.ACE, "acestep-v15-turbo", adapter)
    container.categories.add_model_to_category(ballad.id, m.id)
    assert [x.id for x in container.categories.category_models(ballad.id, Engine.ACE)] == [m.id]
    container.categories.delete_grouping(ballad.id)
    assert ballad.id not in [c.id for c in container.categories.list()]
    assert container.categories.models.get(m.id) is not None and adapter.is_dir()


def test_workspace_cannot_write_into_protected_ace_assets(container, settings):
    with pytest.raises(PathSecurityError):
        container.guard.ensure_writable(settings.ace_root / "datasets" / "test01" / "x.json")
    with pytest.raises(PathSecurityError):
        container.guard.ensure_writable(settings.ace_root / "lora_output" / "final_lora" / "adapter" / "x")


def test_register_existing_final_lora_by_reference(container, ballad, settings):
    adapter = settings.music_root.joinpath(*FINAL_LORA)
    before = {p: p.read_bytes() for p in adapter.iterdir()}
    model = container.categories.register_model(None, Engine.ACE, "acestep-v15-turbo", adapter)
    assert model.validation_status == ValidationStatus.STRUCTURE_OK
    container.categories.set_active_model(ballad.id, Engine.ACE, model.id)
    assert container.categories.active_model(ballad.id, Engine.ACE).id == model.id
    assert {p: p.read_bytes() for p in adapter.iterdir()} == before  # 원본 무변경
    with pytest.raises(ValidationError):
        container.categories.register_model(ballad.id, Engine.ACE, "x", settings.music_root)  # 구조 불일치


# ---------------- 가사 ----------------
def test_lyrics_analysis_heuristics():
    text = "[Verse 1]\n비가 내리는 밤\n[Chorus]\n돌아와 줘\n[Chorus]\n돌아와 줘"
    r = analyze(text, 3, 6, forbidden=["사랑"])
    assert r.has_chorus and r.chorus_repeated and r.sections == ["Verse 1", "Chorus", "Chorus"]
    assert count_syllables("비가 내리는 밤") == 6
    lyrics, title = clean_llm_lyrics("```\n제목: 빗소리\n[Verse 1]\n가사\n```")
    assert title == "빗소리" and lyrics.startswith("[Verse 1]")


async def test_lyrics_generate_uses_lora_only_when_selected(container, lyrics, settings, tmp_path):
    """가사 모델(Qwen LoRA)은 직접 고른 경우에만 쓴다. 안 고르면 Qwen 기본 모델."""
    drafts = await container.lyrics.generate(LyricsRequest(topic="비 오는 밤"))
    assert lyrics.requests[-1].adapter_path is None and drafts[0].model_id is None
    qdir = tmp_path / "qwen_adapter"
    qdir.mkdir()
    (qdir / "adapter_config.json").write_text("{}")
    (qdir / "adapters.safetensors").write_bytes(b"0")
    m = container.categories.register_model(None, Engine.QWEN, settings.qwen_base_model, qdir)
    (draft,) = await container.lyrics.generate(LyricsRequest(topic="x", model_id=m.id))
    assert lyrics.requests[-1].adapter_path == str(qdir) and draft.model_id == m.id
    await container.lyrics.generate(LyricsRequest(topic="x"))
    assert lyrics.requests[-1].adapter_path is None       # 다음 생성에 남지 않음
    with pytest.raises(ValidationError):
        await container.lyrics.generate(LyricsRequest(topic="x", model_id="없는-모델"))


async def test_lyrics_edit_overwrites_in_place(container):
    """가사 저장은 덮어쓰기 — 가사 하나 = 기록 하나 (v2, v3 를 만들지 않는다)."""
    (draft,) = await container.lyrics.generate(LyricsRequest(topic="비"))
    edited = container.lyrics.save_edit(draft.id, draft.lyrics + "\n[Outro]\n안녕")
    assert edited.id == draft.id and edited.version == draft.version
    assert container.lyrics.get(draft.id).lyrics.endswith("안녕")
    assert len(container.lyrics.history(draft.id)) == 1
    with pytest.raises(ValidationError):
        container.lyrics.save_edit(draft.id, "   ")


# ---------------- 프롬프트 디렉터 ----------------
class _Fixed(PromptStrategy):
    name = "fixed"

    def __init__(self, prompts):
        self.prompts = list(prompts)

    async def produce(self, category, lyrics, wishes):
        return self.prompts.pop(0)


async def test_director_rejects_genre_drift_then_falls_back_to_template(ballad):
    drift = MusicPrompt(prompt="aggressive trap beat, 808", genre="trap", bpm=140)
    director = PromptDirector([_Fixed([drift, drift]), TemplateStrategy()])
    result = await director.direct(ballad, "[Verse]\n가사")
    assert result.source == "template"
    assert "rock ballad" in result.prompt and "male vocals" in result.prompt
    assert 70 <= result.bpm <= 85


async def test_director_qwen_json_passes_and_bpm_clamped(container, ballad, lyrics):
    lyrics.next_text = 'Sure! ```json\n{"prompt": "emotional rock ballad, piano", "genre": "rock ballad", "bpm": 120}\n```'
    result = await container.director.direct(ballad, "[Verse]\n가사")
    assert result.source == "qwen" and result.bpm == 85
    assert lyrics.requests[-1].adapter_path is None  # 음악 디렉터는 항상 Qwen 기본 모델
    assert any("BPM" in n for n in result.notes)


def test_manual_prompt_guardrail(container, ballad):
    with pytest.raises(ValidationError):
        container.director.validate_manual(ballad, "hard techno")
    ok = container.director.validate_manual(ballad, "rock ballad with strings", bpm=80)
    assert ok.source == "manual"


def test_forbidden_term_guard(ballad):
    ballad.style.forbidden_terms = ["autotune"]
    from app.services.prompt_service import GuardrailViolation
    with pytest.raises(GuardrailViolation):
        default_guardrails().handle(MusicPrompt(prompt="rock ballad, heavy autotune"), ballad.style)


# ---------------- 음악 생성 ----------------
async def test_music_generation_base_model_records_metadata(container, music, tmp_path):
    (draft,) = await container.lyrics.generate(LyricsRequest(topic="비"))
    songs = await container.music.generate(MusicGenerationSpec(
        draft_id=draft.id, title="비 오는 밤", prompt="rock ballad, piano", seed=11, use_lora=False))
    song = songs[0]
    assert Path(song.audio_path).is_file() and song.title == "비 오는 밤"
    assert Path(song.audio_path).is_relative_to(container.music.resolve_output_dir(""))   # 기본: library/songs
    assert song.generation["lora"]["requested"] is False and song.generation["is_mock"] is True
    assert song.generation["request"]["seed"] == 11 and song.music_model_id is None

    out = tmp_path / "my_songs"                                                            # 저장 폴더 지정 (있는 폴더만)
    out.mkdir()
    (custom,) = await container.music.generate(MusicGenerationSpec(
        draft_id=draft.id, prompt="rock ballad", use_lora=False, output_dir=str(out)))
    assert Path(custom.audio_path).parent == out.resolve()


async def test_music_generation_with_lora_loads_and_switches(container, ballad, music, settings):
    adapter = settings.music_root.joinpath(*FINAL_LORA)
    m = container.categories.register_model(None, Engine.ACE, "acestep-v15-turbo", adapter)
    container.categories.set_active_model(ballad.id, Engine.ACE, m.id)
    spec = MusicGenerationSpec(category_id=ballad.id, lyrics="[Verse]\n가사", prompt="rock ballad", seed=1)
    on = (await container.music.generate(spec))[0]
    assert on.music_model_id == m.id and on.generation["lora"]["load_status"] == "loaded"
    off = (await container.music.generate(spec.model_copy(update={"use_lora": False})))[0]
    assert off.generation["lora"]["load_status"] == "unloaded"
    assert (await music.lora_status()).loaded is False


async def test_lora_requested_but_engine_cannot_load_is_error(container, ballad, music, settings):
    from app.adapters.ports import CapabilityReport

    async def no_caps(refresh=False):
        return CapabilityReport(engine="ace")

    music.capabilities = no_caps  # LoRA 라우트 미확인 상황
    adapter = settings.music_root.joinpath(*FINAL_LORA)
    m = container.categories.register_model(None, Engine.ACE, "base", adapter)
    container.categories.set_active_model(ballad.id, Engine.ACE, m.id)
    with pytest.raises(CapabilityUnsupportedError):
        await container.music.generate(MusicGenerationSpec(category_id=ballad.id, lyrics="x", prompt="rock ballad"))


# ---------------- 잡 매니저 / 리소스 락 ----------------
async def test_job_lifecycle_events_and_persistence(container, ballad):
    seen = []
    container.bus.subscribe(Topics.JOB_UPDATED, lambda e: seen.append(e.payload["job"].status))
    job = container.jobs.submit(JobKind.LYRICS_GENERATE, {"category_id": ballad.id, "topic": "비"})
    done = await container.jobs.wait(job.id, timeout=5)
    assert done.status == JobStatus.SUCCEEDED and done.progress == 1.0
    assert seen[:3] == [JobStatus.VALIDATING, JobStatus.QUEUED, JobStatus.RUNNING]
    assert container.job_repo.events(job.id)


async def test_job_validation_failure_and_unknown_kind(container):
    job = container.jobs.submit(JobKind.LYRICS_GENERATE, {"category_id": "x", "topic": ""})
    done = await container.jobs.wait(job.id, timeout=5)
    assert done.status == JobStatus.FAILED and done.progress is None


async def test_gpu_jobs_run_sequentially(container, ballad, lyrics):
    lyrics.latency_s = 0.1
    running = []
    peak = []
    container.bus.subscribe(Topics.RESOURCE_LOCK, lambda e: (
        running.append(1) if e.payload["acquired"] else running.pop(), peak.append(len(running))))
    jobs = [container.jobs.submit(JobKind.LYRICS_GENERATE, {"category_id": ballad.id, "topic": f"t{i}"})
            for i in range(3)]
    for j in jobs:
        assert (await container.jobs.wait(j.id, timeout=5)).status == JobStatus.SUCCEEDED
    assert max(peak) == 1


async def test_cancel_job(container, ballad, music):
    music.polls_until_done = 10_000
    job = container.jobs.submit(JobKind.MUSIC_GENERATE, {"spec": {
        "category_id": ballad.id, "lyrics": "x", "prompt": "rock ballad", "use_lora": False}})
    await asyncio.sleep(0.1)
    assert await container.jobs.cancel(job.id)
    done = container.job_repo.get(job.id)
    assert done.status == JobStatus.CANCELLED
    assert ResourceLock().gpu_in_use == 0


async def test_ab_compare_uses_same_seed(container, ballad, settings, music):
    adapter = settings.music_root.joinpath(*FINAL_LORA)
    m = container.categories.register_model(None, Engine.ACE, "base", adapter)
    container.categories.set_active_model(ballad.id, Engine.ACE, m.id)
    job = container.jobs.submit(JobKind.MUSIC_GENERATE, {"ab_compare": True, "spec": {
        "category_id": ballad.id, "lyrics": "[Verse]\n가사", "prompt": "rock ballad"}})
    done = await container.jobs.wait(job.id, timeout=5)
    assert done.status == JobStatus.SUCCEEDED
    seeds = {r.seed for r in music.submitted[-2:]}
    assert len(seeds) == 1 and None not in seeds
    a, b = (container.song_repo.get(done.result["ab"][k][0]) for k in ("a", "b"))
    assert a.music_model_id is None and b.music_model_id == m.id


async def test_full_song_failure_keeps_lyrics(container, ballad, music):
    music.fail = True
    job = container.jobs.submit(JobKind.FULL_SONG, {"category_id": ballad.id, "topic": "비", "use_lora": False})
    done = await container.jobs.wait(job.id, timeout=5)
    assert done.status == JobStatus.FAILED
    assert container.lyrics.list_for_category(ballad.id)  # 가사는 보존


def test_reconcile_marks_orphan_running_jobs_failed(container):
    from app.domain.entities import Job

    orphan = Job(kind=JobKind.MUSIC_GENERATE, status=JobStatus.RUNNING, progress=0.5)
    container.job_repo.add(orphan)
    assert orphan.id in container.jobs.reconcile_on_startup()
    fixed = container.job_repo.get(orphan.id)
    assert fixed.status == JobStatus.FAILED and fixed.progress is None


async def test_music_train_refuses_without_dataset_or_verified_routes(container, music, settings):
    from app.adapters.ports import CapabilityReport
    from app.domain.entities import Dataset
    from app.domain.enums import DatasetKind

    done = await container.jobs.wait(container.jobs.submit(JobKind.MUSIC_TRAIN, {}).id, timeout=5)
    assert done.status == JobStatus.FAILED and "dataset_id" in done.error

    tensors = settings.workspace_dir / "t"
    tensors.mkdir(parents=True)
    (tensors / "a.pt").write_bytes(b"0")
    ds = Dataset(category_id=None, kind=DatasetKind.MUSIC, version=1, status="preprocessed",
                 source_path=str(tensors), tensor_path=str(tensors), rights_status="confirmed")
    container.music_datasets.datasets.add(ds)

    async def no_caps(refresh=False):
        return CapabilityReport(engine="ace")

    music.capabilities = no_caps        # 학습 라우트가 서버에서 확인되지 않음 → 추측해서 호출하지 않는다
    done = await container.jobs.wait(container.jobs.submit(JobKind.MUSIC_TRAIN, {"dataset_id": ds.id}).id, timeout=5)
    assert done.status == JobStatus.FAILED and "확인되지 않은 기능" in done.error
    assert music.submitted == []


# ---------------- 가사 데이터셋 ----------------
def test_lyrics_dataset_builder_versions_and_validation(container):
    rec = lambda i: json.dumps({"messages": [{"role": "user", "content": f"주제 {i}"},  # noqa: E731
                                             {"role": "assistant", "content": f"[Verse]\n가사 {i}"}]}, ensure_ascii=False)
    text = "\n".join(rec(i) for i in range(10)) + "\n" + rec(1)
    ds1, report = container.lyrics_datasets.build(text)
    assert report.valid == 10 and report.duplicates == 1 and report.warnings
    ds2, _ = container.lyrics_datasets.build(text)
    assert ds1.source_path != ds2.source_path and ds2.version == ds1.version + 1      # 덮어쓰지 않고 새 버전
    assert (Path(ds1.source_path) / "train.jsonl").is_file() and (Path(ds1.source_path) / "valid.jsonl").is_file()
    first = json.loads((Path(ds1.source_path) / "train.jsonl").read_text().splitlines()[0])
    assert first["messages"][0]["role"] == "system"                                  # 생성 때와 같은 지시 포함
    with pytest.raises(ValidationError):
        container.lyrics_datasets.build('{"messages": []}')


def test_lyrics_txt_files_to_dataset(container):
    """가사 .txt 파일만 넣어 데이터셋 만들기."""
    from app.services.training_service import txt_files_to_jsonl

    files = [(f"song{i}.txt", f"제목: 노래 {i}\n[Verse 1]\n비가 내려 {i}\n[Chorus]\n너를 불러 {i}")
             for i in range(6)]
    jsonl = txt_files_to_jsonl(files)
    ds, report = container.lyrics_datasets.build(jsonl)
    assert report.valid == 6 and Path(ds.source_path).is_dir()
