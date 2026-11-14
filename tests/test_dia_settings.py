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


def test_defaults_are_v3_with_retry():
    s = load_settings({})
    assert s.ref_mode == "alternate"
    assert s.trim_silence is True
    assert s.max_retries == 3 and s.verify is False
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
    s = load_settings({"DIA_REF_MODE": "single", "DIA_REF_AUDIO": "none"})
    assert s.ref_audio is None and s.ref_text is None


def test_custom_reference_with_text_file(ref):
    wav, text = ref
    s = load_settings({"DIA_REF_MODE": "single", "DIA_REF_AUDIO": str(wav), "DIA_REF_TEXT": str(text)})
    assert s.ref_audio == str(wav) and s.ref_text == "[S1] Hello there. [S2] Hi."


def test_custom_reference_with_inline_text(ref):
    wav, _ = ref
    s = load_settings({"DIA_REF_MODE": "single", "DIA_REF_AUDIO": str(wav), "DIA_REF_TEXT": "[S1] A.  [S2] B."})
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


# --- v3: alternate references (DIA_REF_MODE=alternate) ---------------------------------------

from settings import DEFAULT_REF_S1, last_speaker, pick_reference  # noqa: E402


@pytest.mark.parametrize("text,speaker", [("[S1] a [S2] b", "[S2]"), ("[S1] a [S2] b [S1] c", "[S1]"),
                                          ("[S2] x", "[S2]")])
def test_last_speaker(text, speaker):
    assert last_speaker(text) == speaker


def test_single_mode_restores_v2():
    s = load_settings({"DIA_REF_MODE": "single"})
    assert s.ref_mode == "single" and len(s.references) == 1 and s.trim_silence is False


def test_fixed_is_an_alias_for_single():
    assert load_settings({"DIA_REF_MODE": "fixed"}).ref_mode == "single"


def test_retry_and_verify_overrides():
    s = load_settings({"DIA_MAX_RETRIES": "0", "DIA_VERIFY": "1"})
    assert s.max_retries == 0 and s.verify is True


@pytest.mark.parametrize("value", ["-1", "11", "many"])
def test_bad_max_retries_rejected(value):
    with pytest.raises(SystemExit):
        load_settings({"DIA_MAX_RETRIES": value})


def test_default_loads_both_references():
    s = load_settings({})
    assert {last_speaker(text) for _, text in s.references} == {"[S1]", "[S2]"}
    assert str(DEFAULT_REF_S1) in [audio for audio, _ in s.references]


def test_default_s1_reference_extends_the_s2_reference():
    s2_text = DEFAULT_REF.with_suffix(".txt").read_text().strip()
    s1_text = DEFAULT_REF_S1.with_suffix(".txt").read_text().strip()
    assert s1_text.startswith(s2_text + " [S1] ")


def test_pick_reference_keeps_tags_alternating():
    refs = (("a.wav", "[S1] x [S2] y"), ("b.wav", "[S1] x [S2] y [S1] z"))
    assert pick_reference(refs, "[S1]") == refs[0]  # chunk opens with S1 -> ref must end with S2
    assert pick_reference(refs, "[S2]") == refs[1]


def test_pick_reference_with_single_reference_returns_it():
    refs = (("a.wav", "[S1] x [S2] y"),)
    assert pick_reference(refs, "[S2]") == refs[0]


def test_pick_reference_without_references_returns_none():
    assert pick_reference((), "[S1]") == (None, None)


def test_alternate_mode_rejects_two_references_ending_alike(ref, tmp_path):
    wav, text = ref
    with pytest.raises(SystemExit):
        load_settings({"DIA_REF_MODE": "alternate", "DIA_REF_AUDIO": str(wav), "DIA_REF_TEXT": str(text),
                       "DIA_REF2_AUDIO": str(wav), "DIA_REF2_TEXT": str(text)})


def test_alternate_mode_with_reference_disabled_rejected():
    with pytest.raises(SystemExit):
        load_settings({"DIA_REF_MODE": "alternate", "DIA_REF_AUDIO": "none"})


def test_unknown_ref_mode_rejected():
    with pytest.raises(SystemExit):
        load_settings({"DIA_REF_MODE": "random"})
