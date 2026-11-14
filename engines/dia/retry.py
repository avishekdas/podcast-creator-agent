"""Detect silent or truncated Dia chunks and regenerate them with a new seed (numpy only).

Dia sometimes returns long silence, or stops after a word or two, instead of reading the turn.
Measured on the v2/v3 video-likes runs: raw Dia speech runs at about 0.25 s a word (240 words a
minute). Every chunk Whisper confirmed as silent or cut short had less than 0.3 of that expected
speech time (turns 30, 36: 0.00; turn 38: 0.22), and the lowest normal chunk had 0.38. So a chunk
is bad below MIN_SPEECH_RATIO. Garbled-but-full-length chunks need the optional verifier.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

SECONDS_PER_WORD = 0.25
MIN_SPEECH_RATIO = 0.3
WINDOW_SECONDS = 0.02
SPEECH_RMS = 0.01
MIN_VERIFY_SCORE = 0.6


@dataclass(frozen=True)
class Attempt:
    audio: np.ndarray
    seed: int
    ratio: float
    score: float | None
    stage: str = "primary"


@dataclass(frozen=True)
class RetryResult:
    audio: np.ndarray
    seed: int
    attempts: int
    ok: bool
    ratio: float
    score: float | None
    stage: str = "primary"


Generator = Callable[[int], np.ndarray]


def speech_seconds(audio: np.ndarray, rate: int) -> float:
    """Seconds of audible sound: 20 ms windows at or above SPEECH_RMS."""
    window = max(1, int(rate * WINDOW_SECONDS))
    frames = len(audio) // window
    if frames == 0:
        return 0.0
    blocks = audio[: frames * window].reshape(frames, window).astype(np.float64)
    return float((np.sqrt(np.mean(blocks ** 2, axis=1)) >= SPEECH_RMS).sum() * window / rate)


def speech_ratio(audio: np.ndarray, rate: int, words: int) -> float:
    """Audible seconds divided by the seconds Dia normally needs for `words` words."""
    return speech_seconds(audio, rate) / max(1, words) / SECONDS_PER_WORD


def is_bad(audio: np.ndarray, rate: int, words: int) -> bool:
    return speech_ratio(audio, rate, words) < MIN_SPEECH_RATIO


def _passes(attempt: Attempt) -> bool:
    if attempt.ratio < MIN_SPEECH_RATIO:
        return False
    return attempt.score is None or attempt.score >= MIN_VERIFY_SCORE


def _best(attempts: list[Attempt]) -> Attempt:
    if attempts[0].score is not None:
        return max(attempts, key=lambda a: (a.score, a.ratio))
    return max(attempts, key=lambda a: a.ratio)


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def generate_with_retry(make: Generator, words: int, seed: int, rate: int, max_retries: int,
                        verify: Callable[[np.ndarray], float] | None = None,
                        label: str = "chunk", log: Callable[[str], None] = _stderr,
                        fallbacks: Sequence[tuple[str, Generator]] = ()) -> RetryResult:
    """Call make(seed); while the result is bad, retry with seed + attempt, up to max_retries.

    If every seed fails, try each (name, generator) in `fallbacks` once, in order (the pipeline
    passes the other reference clip, then no reference). Returns the first attempt that passes,
    or else the best attempt across all stages (most speech, or highest verifier score when a
    verifier is given). The generators' arrays are never modified.
    """
    schedule = ([("primary", make, seed + i) for i in range(max_retries + 1)]
                + [(name, generator, seed) for name, generator in fallbacks])
    expected = words * SECONDS_PER_WORD
    attempts: list[Attempt] = []
    for number, (stage, generator, stage_seed) in enumerate(schedule, 1):
        audio = generator(stage_seed)
        ratio = speech_ratio(audio, rate, words)
        score = verify(audio) if verify is not None and ratio >= MIN_SPEECH_RATIO else (
            0.0 if verify is not None else None)
        current = Attempt(audio, stage_seed, ratio, score, stage)
        attempts.append(current)
        detail = (f"{label} attempt {number}/{len(schedule)} ({stage}, seed {stage_seed}): "
                  f"{speech_seconds(audio, rate):.1f}s speech, expected {expected:.1f}s"
                  + ("" if score is None else f", verify score {score:.2f}"))
        if _passes(current):
            if number > 1:
                log(f"  retry {detail}: ok, {stage} wins")
            return RetryResult(audio, stage_seed, number, True, ratio, score, stage)
        if number < len(schedule):
            log(f"  retry {detail}: bad, retrying")
    best = _best(attempts)
    if len(schedule) > 1:
        log(f"  retry {label} failed after {len(attempts)} attempts; keeping {best.stage} seed "
            f"{best.seed} ({best.ratio:.2f} of expected speech)")
    return RetryResult(best.audio, best.seed, len(attempts), False, best.ratio, best.score, best.stage)
