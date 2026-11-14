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
from typing import Callable

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


@dataclass(frozen=True)
class RetryResult:
    audio: np.ndarray
    seed: int
    attempts: int
    ok: bool
    ratio: float
    score: float | None


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


def generate_with_retry(make: Callable[[int], np.ndarray], words: int, seed: int, rate: int,
                        max_retries: int, verify: Callable[[np.ndarray], float] | None = None,
                        label: str = "chunk", log: Callable[[str], None] = _stderr) -> RetryResult:
    """Call make(seed); while the result is bad, retry with seed + attempt, up to max_retries.

    Returns the first attempt that passes, or else the best one (most speech, or highest
    verifier score when a verifier is given). The generator's arrays are never modified.
    """
    expected = words * SECONDS_PER_WORD
    attempts: list[Attempt] = []
    for attempt in range(max_retries + 1):
        audio = make(seed + attempt)
        ratio = speech_ratio(audio, rate, words)
        score = verify(audio) if verify is not None and ratio >= MIN_SPEECH_RATIO else (
            0.0 if verify is not None else None)
        current = Attempt(audio, seed + attempt, ratio, score)
        attempts.append(current)
        passed = _passes(current)
        detail = (f"{label} attempt {attempt + 1}/{max_retries + 1} (seed {seed + attempt}): "
                  f"{speech_seconds(audio, rate):.1f}s speech, expected {expected:.1f}s"
                  + ("" if score is None else f", verify score {score:.2f}"))
        if passed:
            if attempt > 0:
                log(f"  retry {detail}: ok")
            return RetryResult(audio, current.seed, len(attempts), True, ratio, score)
        if attempt < max_retries:
            log(f"  retry {detail}: bad, retrying")
    best = _best(attempts)
    if max_retries > 0:
        log(f"  retry {label} failed after {len(attempts)} attempts; keeping seed {best.seed} "
            f"({best.ratio:.2f} of expected speech)")
    return RetryResult(best.audio, best.seed, len(attempts), False, best.ratio, best.score)
