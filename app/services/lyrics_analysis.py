"""가사 구조/음절 휴리스틱 검사 (LYR-004)."""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, Field

SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
HANGUL_RE = re.compile(r"[가-힣]")
KNOWN_SECTIONS = ("verse", "pre-chorus", "chorus", "bridge", "outro", "intro", "hook", "interlude")


class LineCheck(BaseModel):
    section: str
    text: str
    syllables: int
    in_range: bool


class LyricsReport(BaseModel):
    sections: list[str] = Field(default_factory=list)
    lines: list[LineCheck] = Field(default_factory=list)
    has_chorus: bool = False
    chorus_repeated: bool = False
    forbidden_found: list[str] = Field(default_factory=list)
    out_of_range_count: int = 0
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.forbidden_found and bool(self.sections)


def count_syllables(line: str) -> int:
    """한글은 글자=음절. 영문은 모음 그룹 수로 근사."""
    hangul = len(HANGUL_RE.findall(line))
    latin_words = re.findall(r"[A-Za-z]+", line)
    latin = sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in latin_words)
    return hangul + latin


def normalize_section(name: str) -> str:
    base = re.sub(r"\s*\d+$", "", name.strip().lower())
    return base


def analyze(lyrics: str, syllable_min: int | None = None, syllable_max: int | None = None,
            forbidden: list[str] | None = None) -> LyricsReport:
    report = LyricsReport()
    current = ""
    chorus_bodies: list[str] = []
    chorus_buf: list[str] = []

    def flush_chorus() -> None:
        if chorus_buf:
            chorus_bodies.append("\n".join(chorus_buf))
            chorus_buf.clear()

    for raw in lyrics.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := SECTION_RE.match(line):
            flush_chorus()
            current = m.group(1).strip()
            report.sections.append(current)
            continue
        syl = count_syllables(line)
        in_range = (syllable_min is None or syl >= syllable_min) and (syllable_max is None or syl <= syllable_max)
        report.lines.append(LineCheck(section=current, text=line, syllables=syl, in_range=in_range))
        if not in_range:
            report.out_of_range_count += 1
        if normalize_section(current) == "chorus":
            chorus_buf.append(line)
    flush_chorus()

    normalized = [normalize_section(s) for s in report.sections]
    report.has_chorus = "chorus" in normalized
    report.chorus_repeated = len(chorus_bodies) >= 2 and Counter(chorus_bodies).most_common(1)[0][1] >= 2

    lower = lyrics.lower()
    report.forbidden_found = [w for w in (forbidden or []) if w and w.lower() in lower]

    if not report.sections:
        report.warnings.append("섹션 마커([Verse], [Chorus] 등)가 없습니다.")
    unknown = [s for s, n in zip(report.sections, normalized) if n not in KNOWN_SECTIONS]
    if unknown:
        report.warnings.append(f"알 수 없는 섹션 이름: {', '.join(unknown)}")
    if report.has_chorus and len(chorus_bodies) >= 2 and not report.chorus_repeated:
        report.warnings.append("후렴이 반복되지만 가사가 서로 다릅니다.")
    if report.out_of_range_count:
        report.warnings.append(f"음절 범위를 벗어난 줄 {report.out_of_range_count}개 (휴리스틱)")
    if report.forbidden_found:
        report.warnings.append(f"금지 표현 포함: {', '.join(report.forbidden_found)}")
    return report


_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")


def clean_llm_lyrics(text: str) -> tuple[str, str]:
    """LLM 출력에서 코드펜스/머리말 제거. (가사, 추출된 제목)"""
    text = _FENCE_RE.sub("", text).strip()
    title = ""
    lines = text.splitlines()
    first_section = next((i for i, ln in enumerate(lines) if SECTION_RE.match(ln.strip())), None)
    if first_section is not None:
        for ln in lines[:first_section]:
            m = re.match(r"^\s*(?:제목|title|タイトル)\s*[:：]\s*(.+)$", ln.strip(), re.IGNORECASE)
            if m:
                title = m.group(1).strip().strip('"「」')
        lines = lines[first_section:]
    return "\n".join(lines).strip(), title
