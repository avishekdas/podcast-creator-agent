"""Dia run settings from environment variables. Standard library only, so it is testable without MLX.

Defaults are the "v3 + retry" setup (README, "Default: v3 + retry"): 20-word chunks, tempo 0.75,
DIA_REF_MODE=alternate with silence trimming, and up to DIA_MAX_RETRIES=3 regenerations of a chunk
that comes back silent or cut short. Every value can be overridden with its environment variable.

DIA_REF_MODE=alternate: two references, refs/dia-ref.* (ends with an [S2] turn) and
refs/dia-ref-s1.* (ends with an [S1] turn; override with DIA_REF2_AUDIO/DIA_REF2_TEXT). Each chunk
gets the one whose last speaker differs from the chunk's first speaker, so speaker tags alternate.
DIA_REF_MODE=single ("fixed" is an alias) gives v2 back: one reference, no trimming. For an exact
v2 rerun also set DIA_MAX_RETRIES=0. DIA_VERIFY=1 also Whisper-checks every chunk and retries
garbled ones (slower; off by default).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF = ROOT / "refs" / "dia-ref.wav"  # ends with an [S2] turn
DEFAULT_REF_S1 = ROOT / "refs" / "dia-ref-s1.wav"  # same clip plus one [S1] turn
REF_MODES = ("single", "alternate")
REF_MODE_ALIASES = {"fixed": "single"}
MAX_RETRIES_LIMIT = 10
DEFAULTS = {
    "DIA_MODEL": "mlx-community/Dia-1.6B-fp16",
    "DIA_CHUNK_WORDS": "20",
    "DIA_TEMPO": "0.75",
    "DIA_SEED": "42",
    "DIA_KEEP_RAW": "0",
    "DIA_REF_MODE": "alternate",
    "DIA_MAX_RETRIES": "3",
    "DIA_VERIFY": "0",
}
TEMPO_RANGE = (0.5, 2.0)


@dataclass(frozen=True)
class Settings:
    model: str
    chunk_words: int
    tempo: float
    seed: int
    keep_raw: bool
    ref_audio: str | None
    ref_text: str | None
    ref_mode: str = "alternate"
    references: tuple[tuple[str, str], ...] = ()
    trim_silence: bool = True
    max_retries: int = 3
    verify: bool = False


def _number(env: Mapping[str, str], name: str, kind: type):
    raw = env.get(name, DEFAULTS[name])
    try:
        return kind(raw)
    except ValueError:
        raise SystemExit(f"{name} must be a {kind.__name__}, got {raw!r}") from None


def read_reference(audio: str, text: str | None) -> tuple[str, str]:
    """Validate a reference clip and return (audio_path, normalized tagged text)."""
    if not Path(audio).is_file():
        raise SystemExit(f"reference audio not found: {audio}")
    if not text:
        raise SystemExit(f"{audio} needs its exact [S1]/[S2] text (DIA_REF_TEXT)")
    if Path(text).is_file():
        text = Path(text).read_text(encoding="utf-8")
    text = " ".join(text.split())
    if "[S1]" not in text or "[S2]" not in text:
        raise SystemExit("reference text must contain both [S1] and [S2] turns")
    return audio, text


def _reference(env: Mapping[str, str]) -> tuple[str | None, str | None]:
    audio = env.get("DIA_REF_AUDIO")
    if audio == "none":
        return None, None
    if audio is None:
        return read_reference(str(DEFAULT_REF),
                              env.get("DIA_REF_TEXT") or str(DEFAULT_REF.with_suffix(".txt")))
    return read_reference(audio, env.get("DIA_REF_TEXT"))


def last_speaker(text: str) -> str:
    """The last [S1]/[S2] tag in a tagged text."""
    return "[S1]" if text.rfind("[S1]") > text.rfind("[S2]") else "[S2]"


def pick_reference(references, first_tag: str) -> tuple[str | None, str | None]:
    """Pick the reference whose last speaker differs from the chunk's first speaker, so the
    prompt Dia sees keeps alternating ([S1] ... [S2] | [S1] ...). Falls back to the first one."""
    if not references:
        return None, None
    for audio, text in references:
        if last_speaker(text) != first_tag:
            return audio, text
    return references[0]


def _references(env: Mapping[str, str], mode: str, first: tuple[str | None, str | None]):
    if first[0] is None:
        if mode == "alternate":
            raise SystemExit("DIA_REF_MODE=alternate needs a reference (DIA_REF_AUDIO is 'none')")
        return ()
    if mode == "single":
        return (first,)
    second_audio = env.get("DIA_REF2_AUDIO")
    if second_audio is None:
        second = read_reference(str(DEFAULT_REF_S1), env.get("DIA_REF2_TEXT")
                                or str(DEFAULT_REF_S1.with_suffix(".txt")))
    else:
        second = read_reference(second_audio, env.get("DIA_REF2_TEXT"))
    if last_speaker(first[1]) == last_speaker(second[1]):
        raise SystemExit("DIA_REF_MODE=alternate needs one reference ending with [S1] and one ending "
                         "with [S2]; both end with " + last_speaker(first[1]))
    return (first, second)


def _max_retries(env: Mapping[str, str]) -> int:
    value = _number(env, "DIA_MAX_RETRIES", int)
    if not 0 <= value <= MAX_RETRIES_LIMIT:
        raise SystemExit(f"DIA_MAX_RETRIES must be between 0 and {MAX_RETRIES_LIMIT}, got {value}")
    return value


def load_settings(env: Mapping[str, str]) -> Settings:
    tempo = _number(env, "DIA_TEMPO", float)
    if not TEMPO_RANGE[0] <= tempo <= TEMPO_RANGE[1]:
        raise SystemExit(f"DIA_TEMPO must be between {TEMPO_RANGE[0]} and {TEMPO_RANGE[1]}, got {tempo}")
    chunk_words = _number(env, "DIA_CHUNK_WORDS", int)
    if chunk_words < 1:
        raise SystemExit(f"DIA_CHUNK_WORDS must be at least 1, got {chunk_words}")
    ref_audio, ref_text = _reference(env)
    mode = env.get("DIA_REF_MODE", DEFAULTS["DIA_REF_MODE"])
    mode = REF_MODE_ALIASES.get(mode, mode)
    if mode not in REF_MODES:
        raise SystemExit(f"DIA_REF_MODE must be one of {REF_MODES}, got {mode!r}")
    return Settings(
        model=env.get("DIA_MODEL", DEFAULTS["DIA_MODEL"]),
        chunk_words=chunk_words,
        tempo=tempo,
        seed=_number(env, "DIA_SEED", int),
        keep_raw=env.get("DIA_KEEP_RAW", DEFAULTS["DIA_KEEP_RAW"]) == "1",
        ref_audio=ref_audio,
        ref_text=ref_text,
        ref_mode=mode,
        references=_references(env, mode, (ref_audio, ref_text)),
        trim_silence=mode == "alternate",
        max_retries=_max_retries(env),
        verify=env.get("DIA_VERIFY", DEFAULTS["DIA_VERIFY"]) == "1",
    )
