"""Dia run settings from environment variables. Standard library only, so it is testable without MLX.

Defaults are the "v2" setup the user picked by ear (README, "Drift fix (v2)"): 20-word chunks,
tempo 0.75, and the fixed two-speaker voice reference in refs/dia-ref.wav + refs/dia-ref.txt.
Every value can be overridden with its environment variable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REF = ROOT / "refs" / "dia-ref.wav"
DEFAULTS = {
    "DIA_MODEL": "mlx-community/Dia-1.6B-fp16",
    "DIA_CHUNK_WORDS": "20",
    "DIA_TEMPO": "0.75",
    "DIA_SEED": "42",
    "DIA_KEEP_RAW": "0",
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


def load_settings(env: Mapping[str, str]) -> Settings:
    tempo = _number(env, "DIA_TEMPO", float)
    if not TEMPO_RANGE[0] <= tempo <= TEMPO_RANGE[1]:
        raise SystemExit(f"DIA_TEMPO must be between {TEMPO_RANGE[0]} and {TEMPO_RANGE[1]}, got {tempo}")
    chunk_words = _number(env, "DIA_CHUNK_WORDS", int)
    if chunk_words < 1:
        raise SystemExit(f"DIA_CHUNK_WORDS must be at least 1, got {chunk_words}")
    ref_audio, ref_text = _reference(env)
    return Settings(
        model=env.get("DIA_MODEL", DEFAULTS["DIA_MODEL"]),
        chunk_words=chunk_words,
        tempo=tempo,
        seed=_number(env, "DIA_SEED", int),
        keep_raw=env.get("DIA_KEEP_RAW", DEFAULTS["DIA_KEEP_RAW"]) == "1",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
