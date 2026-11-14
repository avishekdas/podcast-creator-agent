"""Tests for engines/dia/retry.py with a fake generator (numpy only; no MLX). Run: ./test.sh"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines" / "dia"))

from retry import MIN_SPEECH_RATIO, SECONDS_PER_WORD, generate_with_retry, is_bad, speech_ratio  # noqa: E402

RATE = 1000


def speech(seconds, silence_before=0.0):
    return np.concatenate([np.zeros(int(silence_before * RATE), np.float32),
                           np.full(int(seconds * RATE), 0.2, np.float32)])


def silent(seconds):
    return np.zeros(int(seconds * RATE), np.float32)


def test_thresholds_match_measured_dia_pace():
    assert SECONDS_PER_WORD == 0.25  # Dia raw speech, about 240 words a minute
    assert MIN_SPEECH_RATIO == 0.3


def test_speech_ratio_counts_only_audible_time():
    assert speech_ratio(speech(2.0, silence_before=10.0), RATE, words=8) == pytest.approx(1.0, abs=0.02)


def test_all_silent_chunk_is_bad():
    assert is_bad(silent(16.5), RATE, words=8)


def test_truncated_chunk_is_bad():
    assert is_bad(speech(1.1), RATE, words=20)  # v3 turn 38: 20 words in 1.1 s


@pytest.mark.parametrize("seconds,words", [(0.6, 6), (0.7, 7), (2.1, 22), (5.0, 20)])
def test_normal_chunks_pass(seconds, words):
    assert not is_bad(speech(seconds), RATE, words=words)  # lowest normal v3 chunks pass


class FakeGenerator:
    """Returns the queued outputs in order and records the seeds it was called with."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.seeds = []

    def __call__(self, seed):
        self.seeds.append(seed)
        return self.outputs[len(self.seeds) - 1]


def test_good_first_attempt_is_not_retried():
    make = FakeGenerator([speech(2.0)])
    result = generate_with_retry(make, words=8, seed=42, rate=RATE, max_retries=3)
    assert make.seeds == [42] and result.attempts == 1 and result.ok and result.seed == 42


def test_bad_chunk_is_retried_with_seed_plus_attempt():
    make = FakeGenerator([silent(16), silent(16), speech(2.0)])
    result = generate_with_retry(make, words=8, seed=42, rate=RATE, max_retries=3)
    assert make.seeds == [42, 43, 44]
    assert result.ok and result.attempts == 3 and result.seed == 44
    assert np.array_equal(result.audio, speech(2.0))


def test_gives_up_after_max_retries_and_keeps_best_attempt():
    outputs = [silent(16), speech(0.2), speech(0.4), speech(0.1)]
    make = FakeGenerator(outputs)
    result = generate_with_retry(make, words=20, seed=7, rate=RATE, max_retries=3)
    assert make.seeds == [7, 8, 9, 10]
    assert not result.ok and result.attempts == 4
    assert np.array_equal(result.audio, outputs[2])  # most speech of the four


def test_zero_retries_returns_first_attempt_even_if_bad():
    make = FakeGenerator([silent(5)])
    result = generate_with_retry(make, words=8, seed=1, rate=RATE, max_retries=0)
    assert make.seeds == [1] and not result.ok and result.attempts == 1


def test_each_retry_is_logged_with_chunk_attempt_and_durations():
    lines = []
    make = FakeGenerator([silent(16), speech(2.0)])
    generate_with_retry(make, words=8, seed=42, rate=RATE, max_retries=3, label="chunk 30", log=lines.append)
    assert len(lines) == 2
    assert "chunk 30" in lines[0] and "attempt 1/4" in lines[0]
    assert "0.0s speech" in lines[0] and "expected 2.0s" in lines[0] and "retrying" in lines[0]
    assert "attempt 2/4" in lines[1] and "ok" in lines[1]


def test_good_first_attempt_logs_nothing():
    lines = []
    generate_with_retry(FakeGenerator([speech(2.0)]), words=8, seed=0, rate=RATE, max_retries=3,
                        log=lines.append)
    assert lines == []


def test_final_failure_is_logged():
    lines = []
    generate_with_retry(FakeGenerator([silent(5)] * 2), words=8, seed=0, rate=RATE, max_retries=1,
                        label="chunk 36", log=lines.append)
    assert "chunk 36" in lines[-1] and "failed" in lines[-1]


def test_verifier_failure_triggers_retry():
    make = FakeGenerator([speech(2.0), speech(2.0)])
    scores = iter([0.1, 0.9])
    result = generate_with_retry(make, words=8, seed=42, rate=RATE, max_retries=3,
                                 verify=lambda audio: next(scores))
    assert make.seeds == [42, 43] and result.ok and result.score == 0.9


def test_with_verifier_best_attempt_is_highest_score():
    outputs = [speech(2.0), speech(2.1), speech(1.9)]
    make = FakeGenerator(outputs)
    scores = iter([0.2, 0.4, 0.3])
    result = generate_with_retry(make, words=8, seed=0, rate=RATE, max_retries=2,
                                 verify=lambda audio: next(scores))
    assert not result.ok and np.array_equal(result.audio, outputs[1]) and result.score == 0.4


def test_generator_output_is_not_mutated():
    out = speech(2.0)
    copy = out.copy()
    generate_with_retry(FakeGenerator([out]), words=8, seed=0, rate=RATE, max_retries=0)
    assert np.array_equal(out, copy)
