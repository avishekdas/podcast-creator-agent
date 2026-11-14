"""Tests for engines/dia/settings.py (stdlib only; no MLX needed). Run: ./test.sh"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines" / "dia"))

from settings import DEFAULT_REF, load_settings  # noqa: E402


@pytest.fixture
def ref(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"RIFF")
    text = tmp_path / "ref.txt"
    text.write_text("[S1] Hello there. [S2] Hi.\n")
    return wav, text


def test_defaults_are_v2():
    s = load_settings({})
    assert s.chunk_words == 20
    assert s.tempo == 0.75
    assert s.seed == 42
    assert s.ref_audio == str(DEFAULT_REF)
    assert s.ref_text == "[S1] Welcome back to Signal and Noise. Today's topic sounds small but gets hard " \
                         "fast: how do you design the like button for a video platform? [S2] A like " \
                         "button? It's one number going up."


def test_default_reference_files_exist():
    assert DEFAULT_REF.is_file() and DEFAULT_REF.with_suffix(".txt").is_file()


def test_env_overrides_defaults():
    s = load_settings({"DIA_CHUNK_WORDS": "45", "DIA_TEMPO": "1.0", "DIA_SEED": "7", "DIA_KEEP_RAW": "1"})
    assert (s.chunk_words, s.tempo, s.seed, s.keep_raw) == (45, 1.0, 7, True)


@pytest.mark.parametrize("tempo", ["0.4", "2.5", "fast"])
def test_bad_tempo_rejected(tempo):
    with pytest.raises(SystemExit):
        load_settings({"DIA_TEMPO": tempo})


def test_bad_chunk_words_rejected():
    with pytest.raises(SystemExit):
        load_settings({"DIA_CHUNK_WORDS": "0"})


def test_reference_can_be_disabled():
    s = load_settings({"DIA_REF_AUDIO": "none"})
    assert s.ref_audio is None and s.ref_text is None


def test_custom_reference_with_text_file(ref):
    wav, text = ref
    s = load_settings({"DIA_REF_AUDIO": str(wav), "DIA_REF_TEXT": str(text)})
    assert s.ref_audio == str(wav) and s.ref_text == "[S1] Hello there. [S2] Hi."


def test_custom_reference_with_inline_text(ref):
    wav, _ = ref
    s = load_settings({"DIA_REF_AUDIO": str(wav), "DIA_REF_TEXT": "[S1] A.  [S2] B."})
    assert s.ref_text == "[S1] A. [S2] B."


def test_reference_without_text_rejected(ref):
    with pytest.raises(SystemExit):
        load_settings({"DIA_REF_AUDIO": str(ref[0])})


def test_missing_reference_file_rejected():
    with pytest.raises(SystemExit):
        load_settings({"DIA_REF_AUDIO": "/no/such.wav", "DIA_REF_TEXT": "[S1] a [S2] b"})


def test_reference_text_needs_both_speakers(ref):
    with pytest.raises(SystemExit):
        load_settings({"DIA_REF_AUDIO": str(ref[0]), "DIA_REF_TEXT": "[S1] only one"})
