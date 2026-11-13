"""가사 .txt 인코딩 읽기 (decode_text) 테스트."""
import pytest

from app.core.errors import ValidationError
from app.services.training_service import decode_text

LYRICS = "제목: 봄날\n[Verse 1]\n보고 싶다 이렇게 말하니까\n더 보고 싶다\n[Chorus]\nLa la la\n"
HANGUL_ONLY = "보고싶다\n더보고싶다\n눈을감아도\n"


@pytest.mark.parametrize("enc", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "cp949"])
def test_decode_supported_encodings(enc):
    assert decode_text(LYRICS.encode(enc)) == LYRICS


@pytest.mark.parametrize("enc", ["utf-16-le", "utf-16-be"])
def test_utf16_without_bom_hangul_only(enc):
    assert decode_text(HANGUL_ONLY.encode(enc)) == HANGUL_ONLY


def test_bom_is_removed():
    assert not decode_text(LYRICS.encode("utf-16")).startswith("﻿")


def test_unreadable_bytes_raise():
    with pytest.raises(ValidationError):
        decode_text(b"\x00\xff\x00\xfe\x81")
