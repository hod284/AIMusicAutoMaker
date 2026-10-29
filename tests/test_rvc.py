from pathlib import Path
import sys

import pytest

from app.adapters.ports import VoiceConvertSpec
from app.adapters.voice.local_adapter import LocalVoiceAdapter
from app.adapters.voice.mock_adapter import write_tone
from app.core.errors import EngineResponseError, ValidationError
from app.domain.enums import JobKind, JobStatus
from app.viewmodels.voice_vm import VoiceConvertViewModel


async def test_rvc_uploaded_model_conversion(container, tmp_path, voice):
    vm = VoiceConvertViewModel(container)
    vm.upload_source('song.wav', write_tone(tmp_path / 'song.wav').read_bytes())
    vm.upload_rvc('singer.pth', b'model')
    vm.upload_rvc('singer.index', b'index')
    vm.semi_tone_shift = 12
    vm.start()
    done = await container.jobs.wait(vm.job.id, timeout=10)
    assert done.status == JobStatus.SUCCEEDED, done.error
    spec = voice.last_convert
    assert spec.backend == 'rvc' and spec.reference is None
    assert spec.checkpoint.name.endswith('singer.pth')
    assert spec.index.name.endswith('singer.index') and spec.semi_tone_shift == 12
    song = container.song_repo.get(done.result['song_ids'][0])
    assert song.generation['voice']['backend'] == 'rvc'
    assert song.generation['voice']['base_model'] is False
    assert vm.rvc_model is None and vm.rvc_index is None
    assert not spec.checkpoint.exists() and not spec.index.exists()
    assert VoiceConvertViewModel(container).rvc_model is None


def test_rvc_upload_replaces_one_file_per_slot(container):
    vm = VoiceConvertViewModel(container)
    vm.upload_rvc('one.pth', b'one')
    previous = container.voice.rvc_path(vm.rvc_model, '.pth')
    vm.upload_rvc('one.index', b'index')
    index = vm.rvc_index
    vm.upload_rvc('two.pth', b'two')
    assert not previous.exists()
    assert vm.rvc_index == index
    assert len(list(container.voice.rvc_dir().glob('*.pth'))) == 1
    vm.upload_rvc('two.index', b'new index')
    assert not (container.voice.rvc_dir() / index).exists()
    assert len(list(container.voice.rvc_dir().glob('*.index'))) == 1


def test_rvc_upload_validation(container):
    for name, data in [('model.zip', b'zip'), ('empty.pth', b'')]:
        with pytest.raises(ValidationError):
            container.voice.store_rvc(name, data)
    a = container.voice.store_rvc('../../voice.pth', b'a')
    b = container.voice.store_rvc('voice.pth', b'b')
    assert a.parent == container.voice.rvc_dir() and a != b
    assert a.read_bytes() == b'a'
    with pytest.raises(ValidationError):
        container.voice.rvc_path(a.name, '.index')


async def test_rvc_missing_model_fails_before_separation(container, tmp_path, voice):
    src = container.voice.store_input('song.wav', write_tone(tmp_path / 'song.wav').read_bytes())
    job = container.jobs.submit(JobKind.VOICE_CONVERT, dict(source='file', source_file=src.name, voice='rvc'))
    done = await container.jobs.wait(job.id, timeout=10)
    assert done.status == JobStatus.FAILED and '.pth' in done.error
    assert not voice.calls


@pytest.mark.parametrize('with_index', [True, False])
async def test_applio_worker_contract(tmp_path, with_index):
    # A strict fake of the documented Applio inference entry point. The worker
    # really executes in a subprocess; verifies argument names, cwd and output.
    root = tmp_path / 'Applio'
    root.mkdir()
    (root / 'core.py').write_text('''
from pathlib import Path
import shutil

def run_infer_script(pitch, index_rate, volume_envelope, protect, f0_method,
                     input_path, output_path, pth_path, index_path,
                     f0_autotune, f0_autotune_strength, proposed_pitch,
                     proposed_pitch_threshold, export_format, embedder_model,
                     embedder_model_custom=None):
    assert Path.cwd().name == 'Applio'
    assert Path(pth_path).read_bytes() == b'model'
    assert pitch == -12 and f0_method == 'rmvpe' and export_format == 'WAV'
    assert index_rate == (0.3 if index_path else 0.0)
    if index_path:
        assert Path(index_path).read_bytes() == b'index'
    shutil.copy2(input_path, output_path)
    return 'ok', output_path
''')
    model = tmp_path / 'singer.pth'
    model.write_bytes(b'model')
    index = tmp_path / 'singer.index'
    index.write_bytes(b'index')
    adapter = LocalVoiceAdapter(tmp_path/'sep', tmp_path/'models', tmp_path/'seed', tmp_path/'py',
                                applio_root=root, applio_python=Path(sys.executable))
    result = await adapter.convert(VoiceConvertSpec(
        source=write_tone(tmp_path/'source.wav'), output_dir=tmp_path/'out', backend='rvc',
        checkpoint=model, index=index if with_index else None, semi_tone_shift=-12), lambda _: None)
    assert result.name == 'rvc_vocals.wav' and result.stat().st_size > 44
    (root / 'core.py').write_text('def run_infer_script(**kwargs):\n    raise RuntimeError("bad model")\n')
    with pytest.raises(EngineResponseError):
        await adapter.convert(VoiceConvertSpec(source=tmp_path/'source.wav', output_dir=tmp_path/'bad',
                                               backend='rvc', checkpoint=model), lambda _: None)


def test_missing_applio_reports_setup_error(tmp_path):
    adapter = LocalVoiceAdapter(tmp_path/'sep', tmp_path/'models', tmp_path/'seed', tmp_path/'py',
                                applio_root=tmp_path/'missing', applio_python=tmp_path/'missing/python')
    with pytest.raises(Exception, match='Applio'):
        adapter.validate_rvc()
