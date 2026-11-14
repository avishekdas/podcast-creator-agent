"""Tests for tools/make_dia_ref.py cutting helpers (synthetic audio; no models). Run: ./test.sh"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from make_dia_ref import GAP_SECONDS, cut_turn, gap_starts  # noqa: E402

RATE = 1000


def fake_run(lengths):
    """Turns of constant non-zero audio, each followed by GAP_SECONDS of exact silence."""
    parts = []
    for n, length in enumerate(lengths, 1):
        parts += [np.full(length, n / 10, dtype=np.float32), np.zeros(int(RATE * GAP_SECONDS), np.float32)]
    return np.concatenate(parts)


def test_gap_starts_finds_each_turn_end():
    assert gap_starts(fake_run([300, 500, 200]), RATE) == [300, 1050, 1500]


def test_gap_starts_ignores_short_silences():
    audio = fake_run([300, 300])
    audio[100:150] = 0  # 50 ms of silence inside a turn
    assert gap_starts(audio, RATE) == [300, 850]


def test_cut_turn_returns_only_that_turns_audio():
    audio = fake_run([300, 500, 200])
    for n, length in enumerate([300, 500, 200], 1):
        clip = cut_turn(audio, RATE, n)
        assert len(clip) == length and np.all(clip == np.float32(n / 10))
