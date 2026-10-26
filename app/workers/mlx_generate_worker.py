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
# 가사 언어별로 섞이면 안 되는 문자 범위 (영어·숫자·[Verse] 표시는 별도 규칙)
_OTHERS = [(0x0370, 0x03FF), (0x0400, 0x04FF), (0x0590, 0x05FF), (0x0600, 0x06FF), (0x0900, 0x0DFF),
           (0x0E00, 0x0E7F)]                                   # 그리스·키릴·히브리·아랍·인도계·태국
_HAN = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FFFF)]
_KANA = [(0x3040, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F)]
_HANGUL = [(0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF)]
BANNED_RANGES = {
    "ko": _OTHERS + _HAN + _KANA,             # 한국어: 한자·가나·기타 문자 차단
    "ko_en": _OTHERS + _HAN + _KANA,          # 한국어+영어: 한국어와 같되 영어는 어디서나 허용
    "en": _OTHERS + _HAN + _KANA + _HANGUL,   # 영어: 한글까지 차단
    "ja": _OTHERS + _HANGUL,                  # 일본어: 한자·가나 허용, 한글 차단
}


def _ranges_re(ranges: list[tuple[int, int]]) -> re.Pattern[str]:
    return re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in ranges) + "]")


BANNED_SCRIPTS = {lang: _ranges_re(r) for lang, r in BANNED_RANGES.items()}
FOREIGN_RE = BANNED_SCRIPTS["ko"]
_LATIN_RE = re.compile("[A-Za-z]")
LATIN_ONLY_IN_MARKERS = {"ko", "ja"}   # 한국어·일본어 가사: 영어는 [Verse 1] 같은 구조 표시 안에서만 허용


def _latin_outside_marker(piece: str) -> bool:
    """토큰 조각에 영문이 있고, 그 영문 앞에 '[' 가 없으면 True ('[Verse' 같은 토큰은 허용)."""
    m = _LATIN_RE.search(piece)
    return bool(m) and "[" not in piece[: m.start()]


def _byte_decoder() -> dict[str, int]:
    """GPT-2/Qwen byte-level BPE 의 '토큰 글자 → 원래 바이트' 표."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _utf8_len(lead: int) -> int:
    if lead < 0x80:
        return 1
    if 0xC0 <= lead < 0xE0:
        return 2
    if 0xE0 <= lead < 0xF0:
        return 3
    if 0xF0 <= lead < 0xF8:
        return 4
    return 0   # 이어지는 바이트(0x80~0xBF) 또는 잘못된 바이트


def _split_incomplete(data: bytes) -> tuple[bytes, bytes]:
    """(완성된 부분, 끝에 남은 미완성 UTF-8 글자 조각)."""
    for back in range(1, min(4, len(data)) + 1):
        need = _utf8_len(data[-back])
        if need == 0:
            continue
        return (data, b"") if need <= back else (data[:-back], data[-back:])
    return data, b""


class LanguageLock:
    """mlx_lm logits processor: 고른 가사 언어에 맞지 않는 글자를 생성 단계에서 막는다.

    Qwen 토큰은 '바이트' 단위라 드문 한자(窓 등)는 한 글자가 여러 토큰으로 쪼개진다.
    그래서 토큰을 글자가 아니라 원래 UTF-8 바이트로 보고,
      - 그 토큰으로 완성되는 글자가 금지 문자이거나
      - 그 토큰이 시작하는 미완성 글자가 어떻게 끝나도 금지 문자뿐이면 막는다.
    앞 토큰이 글자를 반쯤 써 놓은 상태(남은 조각)면 그 조각에 이어 붙여 다시 판단한다 (조각별로 캐시).
    한국어·일본어는 영문을 '[ ]' 구조 표시 안에서만 허용한다.
    """

    def __init__(self, tokenizer, lang: str) -> None:  # type: ignore[no-untyped-def]
        import mlx.core as mx  # noqa: PLC0415

        self.mx = mx
        self.tok = tokenizer
        self.ranges = sorted(BANNED_RANGES[lang])
        self.char_re = BANNED_SCRIPTS[lang]
        hf = getattr(tokenizer, "_tokenizer", tokenizer)
        dec = _byte_decoder()
        size = len(hf.get_vocab()) if hasattr(hf, "get_vocab") else tokenizer.vocab_size
        self.raw: list[bytes | None] = []
        for tid in range(size):
            piece = hf.convert_ids_to_tokens(tid)
            try:
                self.raw.append(bytes(dec[c] for c in piece) if piece else b"")
            except (KeyError, TypeError):
                self.raw.append(None)   # <|im_end|> 같은 특수 토큰
        self.cont_ids = [t for t, b in enumerate(self.raw) if b and 0x80 <= b[0] < 0xC0]
        base: list[int] = []
        latin: list[int] = []
        for tid, b in enumerate(self.raw):
            if b is None:
                continue
            if self._bytes_banned(b):
                base.append(tid)
            elif lang in LATIN_ONLY_IN_MARKERS and _latin_outside_marker(b.decode("utf-8", "ignore")):
                latin.append(tid)
        self.base = set(base)
        # 글자 경계에서 '이어지는 바이트' 로 시작하는 토큰은 깨진 글자(�)만 만든다 → 함께 막는다
        boundary = sorted(self.base | set(self.cont_ids))
        self.base_arr = mx.array(boundary) if boundary else None
        self.latin = mx.array(latin) if latin else None
        self.base_noncont = [t for t in base if t not in set(self.cont_ids)]
        self.cache: dict[bytes, object] = {}
        self.counts = (len(base), len(latin))

    # --- 판단 ---
    def _range_banned(self, lo: int, hi: int) -> bool:
        """[lo, hi] 코드포인트가 전부 금지 범위 안이면 True."""
        cur = lo
        for a, b in self.ranges:
            if b < cur:
                continue
            if a > cur:
                return False
            cur = b + 1
            if cur > hi:
                return True
        return False

    def _bytes_banned(self, data: bytes) -> bool:
        done, tail = _split_incomplete(data)
        if self.char_re.search(done.decode("utf-8", "ignore")):
            return True
        if tail:
            need = _utf8_len(tail[0])
            value = tail[0] & (0xFF >> (need + 1))
            for c in tail[1:]:
                value = (value << 6) | (c & 0x3F)
            rest = 6 * (need - len(tail))
            return self._range_banned(value << rest, ((value + 1) << rest) - 1)
        return False

    def _mask_for(self, pending: bytes):  # type: ignore[no-untyped-def]
        if not pending:
            return self.base_arr
        if pending not in self.cache:
            ids = list(self.base_noncont)
            ids += [t for t in self.cont_ids if self._bytes_banned(pending + self.raw[t])]  # type: ignore[operator]
            self.cache[pending] = self.mx.array(ids) if ids else None
        return self.cache[pending]

    def _recent(self, tokens):  # type: ignore[no-untyped-def]
        return [int(t) for t in tokens[-24:].tolist()]

    def __call__(self, tokens, logits):  # type: ignore[no-untyped-def]
        recent = self._recent(tokens)
        tail_bytes = b"".join(self.raw[t] or b"" for t in recent[-4:])
        pending = _split_incomplete(tail_bytes)[1]
        mask = self._mask_for(pending)
        if mask is not None:
            logits = logits.at[:, mask].add(-1e9)
        if self.latin is not None and not pending:
            text = b"".join(self.raw[t] or b"" for t in recent).decode("utf-8", "ignore")
            if text.rfind("[") <= text.rfind("]"):
                logits = logits.at[:, self.latin].add(-1e9)
        return logits


# --- 구간별 생성: [Verse 1] 마다 몇 줄씩만 쓰게 해 가사 형식을 보장한다 ---
def section_lines(name: str) -> int:
    """구간 이름별 줄 수."""
    n = name.lower()
    if n.startswith(("pre", "bridge", "intro", "outro", "hook")):
        return 2 if n.startswith(("intro", "outro")) else 3
    return 4   # Verse, Chorus, 그 밖


def section_key(name: str) -> str:
    """같은 가사를 반복해야 하는 구간 이름 (Chorus, Pre-Chorus 는 번호가 달라도 같은 가사)."""
    n = name.lower().strip()
    for base in ("pre-chorus", "pre chorus", "prechorus", "chorus", "hook"):
        if n.startswith(base):
            return base.replace(" ", "-")
    return n


def cut_section(text: str, max_lines: int) -> tuple[str, bool]:
    """모델이 쓴 글에서 한 구간만 잘라낸다. (구간 가사, 끝났는지)"""
    for stop in ("[", "\n\n"):
        idx = text.find(stop)
        if idx != -1 and text[:idx].strip():
            text = text[:idx]
            return "\n".join(ln.strip() for ln in text.strip().splitlines() if ln.strip()), True
    lines = [ln.strip() for ln in text.split("\n")]
    done_lines = [ln for ln in lines[:-1] if ln]           # 줄바꿈까지 끝난 줄
    if len(done_lines) >= max_lines:
        return "\n".join(done_lines[:max_lines]), True
    return "\n".join(ln for ln in lines if ln), False


def generate_sections(stream_generate, model, tokenizer, prompt: str, sections: list[str],  # type: ignore[no-untyped-def]
                      title_label: str, gen_kwargs: dict) -> str:
    """제목 → [구간] 순서로 직접 머리말을 써 주고, 모델은 각 구간의 가사 몇 줄만 채운다.

    - 같은 이름의 구간(Chorus 반복)은 처음 쓴 가사를 그대로 복사 → 후렴 일치, 무한 반복 없음
    - 줄 수·빈 줄·'[' 가 나오면 그 구간을 끝낸다
    """
    def run(head: str, max_tokens: int, stop_fn) -> str:  # type: ignore[no-untyped-def]
        out = ""
        for resp in stream_generate(model, tokenizer, prompt=prompt + head, max_tokens=max_tokens, **gen_kwargs):
            out += resp.text
            if stop_fn(out):
                break
        return out

    title_head = f"{title_label}:"
    title = run(title_head, 24, lambda t: "\n" in t.lstrip()).strip().split("\n")[0].strip()
    body = f"{title_head} {title}\n"
    written: dict[str, str] = {}
    for name in sections:
        key = section_key(name)
        if key in written:
            part = written[key]
        else:
            n = section_lines(name)
            head = f"{body}\n[{name}]\n"
            raw = run(head, 32 * n, lambda t, n=n: cut_section(t, n)[1])
            part = cut_section(raw, n)[0] or "..."
            written[key] = part
        body += f"\n[{name}]\n{part}\n"
    return body.strip()


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

    lang = req.get("lock_language")
    if lang in BANNED_SCRIPTS:
        lock = LanguageLock(tokenizer, lang)
        gen_kwargs["logits_processors"] = [lock]
        sys.stderr.write(f"lock_language={lang}: 다른 문자 토큰 {lock.counts[0]}개 항상 차단, "
                         f"구조 표시 밖 영문 토큰 {lock.counts[1]}개 차단\n")

    if req.get("sections"):
        from mlx_lm import stream_generate  # noqa: PLC0415

        # 같은 줄 무한 반복 방지 (후렴은 복사하므로 반복 벌점이 후렴을 망치지 않는다)
        try:
            from mlx_lm.sample_utils import make_logits_processors  # noqa: PLC0415

            gen_kwargs["logits_processors"] = gen_kwargs.get("logits_processors", []) + make_logits_processors(
                repetition_penalty=1.1, repetition_context_size=64)
        except ImportError:
            pass
        kwargs = {k: v for k, v in gen_kwargs.items() if k not in ("max_tokens", "verbose")}
        text = generate_sections(stream_generate, model, tokenizer, prompt, list(req["sections"]),
                                 req.get("title_label") or "제목", kwargs)
    else:
        try:
            text = generate(model, tokenizer, prompt=prompt, **gen_kwargs)
        except TypeError:   # 아주 오래된 mlx_lm: logits_processors 인자 없음
            gen_kwargs.pop("logits_processors", None)
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
