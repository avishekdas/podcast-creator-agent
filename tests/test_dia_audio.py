"""Tests for engines/dia/audio_utils.py (numpy only; no MLX). Run: ./test.sh"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines" / "dia"))

from audio_utils import trim_silence  # noqa: E402

RATE = 1000


def speech(n):
    return np.full(n, 0.2, dtype=np.float32)


def test_trims_long_leading_silence_keeping_a_margin():
    out = trim_silence(np.concatenate([np.zeros(11 * RATE, np.float32), speech(500)]), RATE, margin=0.05)
    assert len(out) == 550 and np.all(out[50:] == np.float32(0.2))


def test_trims_trailing_silence():
    out = trim_silence(np.concatenate([speech(500), np.zeros(3 * RATE, np.float32)]), RATE, margin=0.05)
    assert len(out) == 550


def test_keeps_silence_inside_speech():
    audio = np.concatenate([speech(300), np.zeros(400, np.float32), speech(300)])
    assert len(trim_silence(audio, RATE, margin=0.0)) == 1000


def test_low_noise_counts_as_silence():
    audio = np.concatenate([np.full(2000, 0.001, np.float32), speech(500)])
    assert len(trim_silence(audio, RATE, margin=0.0)) == 500


def test_all_silent_input_returned_unchanged():
    audio = np.zeros(800, np.float32)
    assert len(trim_silence(audio, RATE)) == 800


def test_does_not_mutate_input():
    audio = np.concatenate([np.zeros(1000, np.float32), speech(200)])
    copy = audio.copy()
    trim_silence(audio, RATE)
    assert np.array_equal(audio, copy)
