"""Run Applio inside its own Python environment; never load model weights in Studio."""
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
import sys
import tempfile

from rvc_audio import trim_edges, restore_edges


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("root", "source", "output", "model"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.75)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("APPLIO_CONFIG_DIR", str(root / ".studio-config"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".studio-config" / "matplotlib"))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(root / ".studio-config" / "numba"))
    os.environ.setdefault("HF_HOME", str(root / ".studio-config" / "huggingface"))
    from core import run_infer_script

    with tempfile.TemporaryDirectory(prefix="rvc-active-", dir=Path(args.output).parent) as scratch:
        source, timing = trim_edges(Path(args.source), scratch)
        if timing is not None:
            print(f"RVC: exclude silent edges from inference; restore offset {timing[0]:.3f}s", flush=True)
        options = dict(
            pitch=args.pitch, index_rate=args.index_rate if args.index else 0.0,
            volume_envelope=1.0, protect=0.33, f0_method="rmvpe",
            input_path=str(source), output_path=args.output, pth_path=args.model,
            index_path=args.index, f0_autotune=False, f0_autotune_strength=1.0,
            proposed_pitch=False, proposed_pitch_threshold=155.0, export_format="WAV",
            embedder_model="contentvec", embedder_model_custom=None,
            filter_radius=3, resample_sr=0, split_audio=False, clean_audio=False,
            clean_strength=0.5, upscale_audio=False,
        )
        # Applio releases add/remove optional inference parameters.
        signature = inspect.signature(run_infer_script)
        kwargs = {k: v for k, v in options.items() if k in signature.parameters}
        signature.bind(**kwargs)
        print(run_infer_script(**kwargs), flush=True)
        restore_edges(Path(args.output), timing)
    dest = Path(args.output)
    if not dest.is_file() or dest.stat().st_size <= 44:
        raise RuntimeError("Applio did not produce a WAV file")


if __name__ == "__main__":
    main()
