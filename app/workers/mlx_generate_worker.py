"""MLX-LM 추론 워커 — **lyrics-ai/.venv 의 Python 으로 실행**된다.

studio 패키지를 import 하지 않는 독립 스크립트다 (두 가상환경 분리 유지).
stdin 으로 JSON 요청을 받고 stdout 마지막 줄에 JSON 결과를 출력한다.

요청 예:
{"model": "mlx-community/Qwen3-4B-4bit", "adapter_path": null, "system": "...",
 "user": "...", "max_tokens": 1024, "temperature": 0.8, "top_p": 0.95, "seed": 42,
 "enable_thinking": false}

주의: mlx_lm Python API(load/generate/make_sampler) 시그니처는 버전에 따라 다를 수 있다.
설치본에서 ``tests/manual/test_mlx_worker.sh`` 로 먼저 확인할 것.
"""

from __future__ import annotations

import json
import re
import sys
import time

RESULT_PREFIX = "@@RESULT@@"
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_thinking(text: str) -> tuple[str, str]:
    thinking = "\n".join(m.strip() for m in THINK_RE.findall(text))
    body = THINK_RE.sub("", text)
    # 닫히지 않은 <think> (max_tokens 도달) 처리
    if "<think>" in body:
        head, _, tail = body.partition("<think>")
        thinking = (thinking + "\n" + tail).strip()
        body = head
    return body.strip(), thinking.strip()


def main() -> int:
    req = json.loads(sys.stdin.read())
    started = time.time()

    import mlx.core as mx  # noqa: PLC0415 - lyrics-ai venv 에만 존재
    from mlx_lm import generate, load  # noqa: PLC0415

    if req.get("seed") is not None:
        mx.random.seed(int(req["seed"]))

    load_kwargs = {}
    if req.get("adapter_path"):
        load_kwargs["adapter_path"] = req["adapter_path"]
    model, tokenizer = load(req["model"], **load_kwargs)

    messages = []
    if req.get("system"):
        messages.append({"role": "system", "content": req["system"]})
    messages.append({"role": "user", "content": req["user"]})

    try:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=bool(req.get("enable_thinking", False)),
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    gen_kwargs = {"max_tokens": int(req.get("max_tokens", 1024)), "verbose": False}
    try:
        from mlx_lm.sample_utils import make_sampler  # noqa: PLC0415

        gen_kwargs["sampler"] = make_sampler(temp=float(req.get("temperature", 0.8)),
                                             top_p=float(req.get("top_p", 0.95)))
    except ImportError:
        gen_kwargs["temp"] = float(req.get("temperature", 0.8))

    text = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
    body, thinking = _split_thinking(text)
    result = {
        "text": body,
        "thinking": thinking,
        "model": req["model"],
        "adapter_path": req.get("adapter_path"),
        "elapsed_s": round(time.time() - started, 2),
    }
    sys.stdout.write("\n" + RESULT_PREFIX + json.dumps(result, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
