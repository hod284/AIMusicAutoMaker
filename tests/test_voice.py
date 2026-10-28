"""목소리 모델: 만들기(학습/참고 녹음만), 카테고리, 보컬 바꾸기(분리→변환→합치기), 로컬 어댑터 명령줄.

로컬 어댑터 테스트는 실제 FFmpeg 로 자르기·합치기를 돌리고, audio-separator / Seed-VC 는
설치본 코드의 argparse 정의를 그대로 옮긴 가짜 스크립트로 '보내는 옵션이 실제로 받아지는지' 확인한다.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from app.adapters.ports import VoiceConvertSpec, VoiceTrainSpec
from app.adapters.voice.local_adapter import LocalVoiceAdapter, parse_seedvc_train_line, wav_seconds
from app.adapters.voice.mock_adapter import write_tone
from app.core.errors import ValidationError
from app.domain.enums import Engine, JobKind, JobStatus
from app.services.music_service import MusicGenerationSpec


def _wav_bytes(tmp_path: Path, seconds: float = 3.0) -> bytes:
    return write_tone(tmp_path / f"_src_{seconds}.wav", seconds).read_bytes()


async def _train(container, tmp_path, name="내 목소리", mode="train", **cfg):
    container.voice.stage_upload("my_song.wav", _wav_bytes(tmp_path))
    job = container.jobs.submit(JobKind.VOICE_TRAIN, {"config": {"model_name": name, "mode": mode,
                                                                 "max_steps": 100, **cfg}})
    return await container.jobs.wait(job.id, timeout=10)


async def _song(container):
    (song,) = await container.music.generate(MusicGenerationSpec(lyrics="[Verse]\n가사", prompt="ballad",
                                                                 use_lora=False, title="원곡"))
    return song


# ---------------- 목소리 모델 만들기 ----------------
async def test_train_voice_model_registers_and_keeps_only_final(container, tmp_path, voice):
    done = await _train(container, tmp_path)
    assert done.status == JobStatus.SUCCEEDED, done.error
    model = container.categories.models.get(done.result["model_id"])
    root = Path(model.adapter_path)
    assert model.engine == Engine.VOICE and model.display_name == "내 목소리"
    assert root == container.voice.default_models_dir() / "내_목소리"            # 폴더 이름은 안전한 이름으로
    assert (root / "reference.wav").is_file() and (root / "ft_model.pth").is_file() and list(root.glob("*.yml"))
    assert (root / "source" / "my_song.wav").is_file() and list((root / "clips").rglob("*.wav"))
    assert not list(root.glob("DiT_epoch_*.pth"))                   # 중간 체크포인트는 정리
    assert container.voice.staged_files() == []                     # 올린 파일은 모델 폴더로 옮겨짐
    assert voice.calls == ["split", "train"]
    with pytest.raises(ValidationError):                            # 같은 이름은 덮어쓰지 않음
        container.voice.resolve_export_dir("내 목소리")


async def test_reference_only_and_separation_option(container, tmp_path, voice):
    done = await _train(container, tmp_path, name="참고만", mode="reference", separate=True)
    assert done.status == JobStatus.SUCCEEDED, done.error
    root = Path(container.categories.models.get(done.result["model_id"]).adapter_path)
    assert (root / "reference.wav").is_file() and not (root / "ft_model.pth").exists()
    assert voice.calls == ["separate", "split"]                     # 학습 없이, 반주 제거 후 조각만


async def test_train_failure_restores_uploads_and_removes_folder(container, tmp_path, voice):
    from app.adapters.voice.mock_adapter import _MockHandle

    async def failing(spec, on_progress):
        async def _run():
            return 1
        return _MockHandle(asyncio.create_task(_run()))

    voice.start_training = failing
    done = await _train(container, tmp_path, name="실패")
    assert done.status == JobStatus.FAILED and "종료 코드 1" in done.error
    assert [p.name for p in container.voice.staged_files()] == ["my_song.wav"]   # 다시 시도할 수 있게
    assert not (container.voice.default_models_dir() / "실패").exists()


async def test_voice_train_validation(container, tmp_path):
    done = await container.jobs.wait(container.jobs.submit(JobKind.VOICE_TRAIN, {"config": {"model_name": "x"}}).id,
                                     timeout=5)
    assert done.status == JobStatus.FAILED and "녹음 파일" in done.error                 # 올린 파일 없음
    with pytest.raises(ValidationError):
        container.voice.stage_upload("a.txt", b"hello")
    with pytest.raises(ValidationError):
        container.voice.stage_upload("a.wav", b"not a wav")


async def test_voice_models_use_categories(container, tmp_path):
    from app.viewmodels.models_vm import VoiceModelsViewModel

    done = await _train(container, tmp_path, name="허스키")
    vm = VoiceModelsViewModel(container)
    assert [m.display_name for m in vm.models] == ["허스키"]
    vm.create_category("남자 보컬")
    vm.add(done.result["model_id"])
    assert vm.member_ids == {done.result["model_id"]} and vm.active_id == done.result["model_id"]
    assert container.voice.voice_models(vm.category_id)[0].id == done.result["model_id"]
    root = Path(container.categories.models.get(done.result["model_id"]).adapter_path)
    assert vm.delete_targets(done.result["model_id"]) == [str(root)]
    vm.delete_model(done.result["model_id"])
    assert vm.models == [] and not root.exists()


# ---------------- 보컬 바꾸기 ----------------
async def test_convert_song_creates_new_song_and_keeps_original(container, tmp_path, voice):
    model_id = (await _train(container, tmp_path)).result["model_id"]
    song = await _song(container)
    before = Path(song.audio_path).read_bytes()
    job = container.jobs.submit(JobKind.VOICE_CONVERT, {"song_id": song.id, "model_id": model_id,
                                                         "semi_tone_shift": -12, "diffusion_steps": 40})
    done = await container.jobs.wait(job.id, timeout=10)
    assert done.status == JobStatus.SUCCEEDED, done.error
    new = container.song_repo.get(done.result["song_ids"][0])
    assert new.id != song.id and new.title == "원곡 · 내 목소리" and Path(new.audio_path).is_file()
    assert Path(new.audio_path).suffix == ".mp3" and Path(new.audio_path).parent == container.voice.output_dir("")
    assert new.generation["voice"]["model_id"] == model_id and new.generation["source_song_id"] == song.id
    stems = Path(new.generation["voice"]["stems_dir"])
    assert (stems / "vocals_converted.wav").is_file() and (stems / "instrumental.wav").is_file()
    assert Path(song.audio_path).read_bytes() == before                          # 원곡 그대로
    spec = voice.last_convert
    assert spec.checkpoint and spec.checkpoint.name == "ft_model.pth" and spec.config.suffix == ".yml"
    assert spec.semi_tone_shift == -12 and spec.diffusion_steps == 40 and spec.f0_condition
    assert not container.voice.work_dir(job.id).exists()                          # 작업 폴더 정리


async def test_convert_with_reference_only_model_has_no_checkpoint(container, tmp_path, voice):
    model_id = (await _train(container, tmp_path, name="참고", mode="reference")).result["model_id"]
    song = await _song(container)
    done = await container.jobs.wait(container.jobs.submit(
        JobKind.VOICE_CONVERT, {"song_id": song.id, "model_id": model_id}).id, timeout=10)
    assert done.status == JobStatus.SUCCEEDED, done.error
    assert voice.last_convert.checkpoint is None and voice.last_convert.config is None


async def test_convert_external_file_with_base_model_into_chosen_folder(container, tmp_path, voice):
    """밖에서 가져온 곡 + 기본 모델(참고 녹음) → 제목으로 지정 폴더에 저장, 같은 제목이면 _2."""
    src = container.voice.store_input("남의곡.wav", _wav_bytes(tmp_path, 4.0))
    ref = container.voice.store_input("내목소리.wav", _wav_bytes(tmp_path, 2.0))
    out = tmp_path / "내결과"
    out.mkdir()
    params = {"source": "file", "source_file": src.name, "voice": "reference", "reference_file": ref.name,
              "reference_separate": True, "title": "새 노래", "output_dir": str(out), "audio_format": "wav"}
    first = await container.jobs.wait(container.jobs.submit(JobKind.VOICE_CONVERT, params).id, timeout=10)
    assert first.status == JobStatus.SUCCEEDED, first.error
    second = await container.jobs.wait(container.jobs.submit(JobKind.VOICE_CONVERT, params).id, timeout=10)
    assert second.status == JobStatus.SUCCEEDED, second.error
    assert Path(first.result["path"]) == out / "새_노래.wav"            # 제목 = 파일 이름
    assert Path(second.result["path"]) == out / "새_노래_2.wav"         # 덮어쓰지 않음
    assert (out / "새_노래_stems" / "vocals_converted.wav").is_file()
    new = container.song_repo.get(first.result["song_ids"][0])
    assert new.title == "새 노래" and new.generation["source"]["type"] == "file"
    assert new.generation["voice"]["base_model"] is True and new.generation["voice"]["model_id"] is None
    assert voice.last_convert.checkpoint is None and voice.last_convert.reference.name == "vocals.wav"   # 반주 제거된 참고
    assert src.is_file() and ref.is_file()                                                        # 올린 원본 유지
    assert voice.calls.count("separate") == 4                                                     # (참고 + 원곡) × 2


async def test_convert_requires_inputs(container):
    for params, msg in [({"source": "file", "voice": "model", "model_id": "x"}, "곡 파일을 올리세요"),
                        ({"source": "file", "source_file": "없음.wav", "voice": "model"}, "다시 올리세요"),
                        ({"source": "song", "voice": "reference"}, "곡을 고르세요")]:
        done = await container.jobs.wait(container.jobs.submit(JobKind.VOICE_CONVERT, params).id, timeout=5)
        assert done.status == JobStatus.FAILED and msg in done.error, done.error


async def test_convert_failure_leaves_no_partial_files(container, tmp_path, voice):
    model_id = (await _train(container, tmp_path)).result["model_id"]
    song = await _song(container)
    folder_before = sorted(p.name for p in Path(song.audio_path).parent.iterdir())
    voice.fail_convert = True
    done = await container.jobs.wait(container.jobs.submit(
        JobKind.VOICE_CONVERT, {"song_id": song.id, "model_id": model_id}).id, timeout=10)
    assert done.status == JobStatus.FAILED
    assert sorted(p.name for p in Path(song.audio_path).parent.iterdir()) == folder_before
    assert len(container.song_repo.list()) == 1


async def test_convert_validation(container):
    done = await container.jobs.wait(container.jobs.submit(
        JobKind.VOICE_CONVERT, {"song_id": "x", "model_id": "y"}).id, timeout=5)
    assert done.status == JobStatus.FAILED and "곡을 찾을 수 없습니다" in done.error


def test_migration_allows_voice_engine_and_keeps_old_models(tmp_path):
    """006 까지 있던 DB 에 007 을 적용해도 기존 음악/가사 모델이 그대로 남는다."""
    import sqlite3

    from app.db import database as dbmod

    old = tmp_path / "old_migrations"
    old.mkdir()
    for f in sorted(dbmod.MIGRATIONS_DIR.glob("*.sql")):
        if f.name < "007":
            shutil.copy(f, old / f.name)
    path = tmp_path / "s.db"
    orig = dbmod.MIGRATIONS_DIR
    try:
        dbmod.MIGRATIONS_DIR = old
        db = object.__new__(dbmod.Database)
        db.__init__(path)
        db.migrate()
        c = sqlite3.connect(path)
        c.execute("INSERT INTO models (id, engine, base_model, adapter_path, version, created_at, display_name) "
                  "VALUES ('m1', 'ace', 'b', '/x', 1, '2026-01-01', '발라드')")
        c.commit()
        c.close()
        dbmod.MIGRATIONS_DIR = orig
        assert db.migrate() == ["007_voice_models.sql"]
    finally:
        dbmod.MIGRATIONS_DIR = orig
    c = sqlite3.connect(path)
    assert c.execute("SELECT display_name FROM models WHERE id='m1'").fetchone()[0] == "발라드"
    c.execute("INSERT INTO models (id, engine, base_model, adapter_path, version, created_at) "
              "VALUES ('v1', 'voice', 'b', '/y', 1, '2026-01-01')")
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO models (id, engine, base_model, adapter_path, version, created_at) "
                  "VALUES ('z', 'nope', 'b', '/y', 1, '2026-01-01')")


# ---------------- 로컬 어댑터 (실제 명령줄) ----------------
_FAKE_SEPARATOR = '''#!{py}
import argparse, json, os, shutil, sys
p = argparse.ArgumentParser()
p.add_argument("audio_files", nargs="*")
p.add_argument("-m", "--model_filename", default="model_bs_roformer_ep_317_sdr_12.9755.ckpt")
p.add_argument("--output_format", default="FLAC")
p.add_argument("--output_dir", default=None)
p.add_argument("--model_file_dir", default="/tmp/audio-separator-models/")
p.add_argument("--custom_output_names", type=json.loads, default=None)
a = p.parse_args()
names = {{k.lower(): v for k, v in a.custom_output_names.items()}}
for stem in ("vocals", "instrumental"):
    shutil.copy(a.audio_files[0], os.path.join(a.output_dir, names[stem] + "." + a.output_format.lower()))
print("separated", a.model_file_dir)
'''

# Seed-VC (Plachtaa/seed-vc @51383ef) inference.py / train.py 의 argparse 정의를 그대로 옮김
_FAKE_INFERENCE = '''
import argparse, os, shutil
def str2bool(v):
    return str(v).lower() in ("yes", "true", "t", "y", "1")
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="./examples/source/source_s1.wav")
parser.add_argument("--target", type=str, default="./examples/reference/s1p1.wav")
parser.add_argument("--output", type=str, default="./reconstructed")
parser.add_argument("--diffusion-steps", type=int, default=30)
parser.add_argument("--length-adjust", type=float, default=1.0)
parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
parser.add_argument("--f0-condition", type=str2bool, default=False)
parser.add_argument("--auto-f0-adjust", type=str2bool, default=False)
parser.add_argument("--semi-tone-shift", type=int, default=0)
parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file", default=None)
parser.add_argument("--config", type=str, help="Path to the config file", default=None)
parser.add_argument("--fp16", type=str2bool, default=True)
args = parser.parse_args()
assert os.path.isfile("configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"), "cwd must be seed-vc"
assert args.f0_condition is True and args.semi_tone_shift == -12
os.makedirs(args.output, exist_ok=True)
src = os.path.basename(args.source).split(".")[0]
tgt = os.path.basename(args.target).split(".")[0]
shutil.copy(args.source, os.path.join(args.output, f"vc_{src}_{tgt}_{args.length_adjust}_{args.diffusion_steps}_{args.inference_cfg_rate}.wav"))
print("RTF: 0.1", args.checkpoint, args.config)
'''

_FAKE_TRAIN = '''
import argparse, os, shutil, re
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml')
parser.add_argument('--pretrained-ckpt', type=str, default=None)
parser.add_argument('--dataset-dir', type=str, default='/path/to/dataset')
parser.add_argument('--run-name', type=str, default='my_run')
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--max-steps', type=int, default=1000)
parser.add_argument('--max-epochs', type=int, default=1000)
parser.add_argument('--save-every', type=int, default=500)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
args = parser.parse_args()
text = open(args.config).read()
log_dir = re.search(r'^log_dir: "(.*)"$', text, re.M).group(1)
out = os.path.join(log_dir, args.run_name)
os.makedirs(out, exist_ok=True)
shutil.copyfile(args.config, os.path.join(out, os.path.basename(args.config)))
for step in range(0, args.max_steps, 10):
    print(f"epoch 0, step {step}, loss: 0.5", flush=True)
open(os.path.join(out, "ft_model.pth"), "wb").write(b"x")
print(f"Final model saved at {os.path.join(out, 'ft_model.pth')}")
'''


@pytest.fixture
def local(tmp_path) -> LocalVoiceAdapter:
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg 없음")
    root = tmp_path / "voice-ai"
    sep = root / ".venv" / "bin" / "audio-separator"
    sep.parent.mkdir(parents=True)
    sep.write_text(_FAKE_SEPARATOR.format(py=sys.executable))
    sep.chmod(0o755)
    svc = root / "seed-vc"
    (svc / "configs" / "presets").mkdir(parents=True)
    (svc / "configs" / "presets" / "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml").write_text(
        'log_dir: "./runs"\nsave_freq: 1\nlog_interval: 10\n')
    (svc / "inference.py").write_text(textwrap.dedent(_FAKE_INFERENCE))
    (svc / "train.py").write_text(textwrap.dedent(_FAKE_TRAIN))
    return LocalVoiceAdapter(sep, root / "models", svc, Path(sys.executable), ffmpeg="ffmpeg")


async def test_local_adapter_split_and_mix_with_real_ffmpeg(local, tmp_path):
    src = write_tone(tmp_path / "long.wav", 34.0, sr=22050)
    clips = await local.split_clips(src, tmp_path / "clips", 15.0)
    assert len(clips) == 3 and all(1.0 <= wav_seconds(c) <= 15.5 for c in clips)     # 15 + 15 + 4초
    with pytest.raises(ValidationError):
        await local.split_clips(src, tmp_path / "clips")                              # 덮어쓰지 않음
    out = await local.mix(write_tone(tmp_path / "v.wav", 3.0), write_tone(tmp_path / "i.wav", 4.0),
                          tmp_path / "mixed.mp3")
    assert out.stat().st_size > 1000
    with pytest.raises(ValidationError):
        await local.mix(tmp_path / "v.wav", tmp_path / "i.wav", tmp_path / "mixed.mp3")


async def test_local_adapter_cli_arguments_match_installed_tools(local, tmp_path):
    logs: list[str] = []
    vocals, inst = await local.separate(write_tone(tmp_path / "song.wav", 2.0), tmp_path / "sep", logs.append)
    assert vocals.name == "vocals.wav" and inst.name == "instrumental.wav"
    assert any(str(tmp_path / "voice-ai" / "models") in line for line in logs)

    model_root = tmp_path / "models" / "내목소리"
    clips = model_root / "clips"
    clips.mkdir(parents=True)
    write_tone(clips / "a.wav")
    steps: list[int] = []
    handle = await local.start_training(
        VoiceTrainSpec(dataset_dir=clips, log_dir=model_root.parent, run_name=model_root.name, max_steps=30),
        lambda tp: steps.append(tp.step) if tp.step is not None else None)
    assert await handle.wait() == 0
    assert steps == [0, 10, 20] and (model_root / "ft_model.pth").is_file()
    cfg = model_root / "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
    assert cfg.is_file() and str(model_root.parent) in cfg.read_text()
    preset = local.seedvc_root / "configs" / "presets" / cfg.name
    assert preset.read_text().startswith('log_dir: "./runs"')                        # 원본 프리셋은 그대로

    out = await local.convert(VoiceConvertSpec(source=vocals, reference=write_tone(tmp_path / "ref.wav"),
                                               output_dir=tmp_path / "conv", checkpoint=model_root / "ft_model.pth",
                                               config=cfg, semi_tone_shift=-12), logs.append)
    assert out.name.startswith("vc_vocals_ref_") and out.is_file()


def test_parse_seedvc_train_line():
    tp = parse_seedvc_train_line("epoch 3, step 120, loss: 0.4321")
    assert tp.step == 120 and tp.train_loss == pytest.approx(0.4321)
    assert parse_seedvc_train_line("Loading model...") is None


async def test_local_adapter_health_reports_missing(tmp_path):
    a = LocalVoiceAdapter(tmp_path / "no-sep", tmp_path / "m", tmp_path / "no-seedvc", tmp_path / "no-py")
    h = await a.health()
    assert not h.ok and "audio-separator" in h.message


def test_last_error_picks_exception_line():
    from app.adapters.voice.local_adapter import last_error

    tail = ["It is strongly recommended to pass the `sampling_rate` argument", "Traceback (most recent call last):",
            '  File "inference.py", line 329, in main', "TypeError: Cannot convert a MPS Tensor to float64 dtype."]
    assert last_error(tail).startswith("TypeError: Cannot convert")
    assert last_error(["loading", "done"]) == "done"
