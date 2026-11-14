"""Audio helpers for the Dia pipeline (numpy only, so they are testable without MLX)."""
from __future__ import annotations

import numpy as np

WINDOW_SECONDS = 0.02
SILENCE_RMS = 0.01


def trim_silence(audio: np.ndarray, rate: int, margin: float = 0.05,
                 threshold: float = SILENCE_RMS) -> np.ndarray:
    """Return a copy of `audio` without leading/trailing stretches quieter than `threshold` RMS.

    Silence inside the speech is kept. `margin` seconds of the trimmed edge are kept on each side
    so word onsets are not clipped. Input that is silent throughout is returned unchanged.
    """
    window = max(1, int(rate * WINDOW_SECONDS))
    frames = len(audio) // window
    if frames == 0:
        return audio.copy()
    blocks = audio[: frames * window].reshape(frames, window).astype(np.float64)
    loud = np.flatnonzero(np.sqrt(np.mean(blocks ** 2, axis=1)) >= threshold)
    if loud.size == 0:
        return audio.copy()
    keep = int(rate * margin)
    start = max(0, loud[0] * window - keep)
    end = min(len(audio), (loud[-1] + 1) * window + keep)
    return audio[start:end].copy()
