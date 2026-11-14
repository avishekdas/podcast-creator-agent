"""Tests for the reference-clip fallback in engines/dia/retry.py and settings.py. Run: ./test.sh

After the seed retries are used up, a still-bad chunk is tried with other generators ("stages"),
by default the other reference clip and then no reference, one attempt each.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engines" / "dia"))

from retry import generate_with_retry  # noqa: E402
from settings import load_settings  # noqa: E402

RATE = 1000


def speech(seconds):
    return np.full(int(seconds * RATE), 0.2, np.float32)


def silent(seconds):
    return np.zeros(int(seconds * RATE), np.float32)


class Stage:
    """Fake generator for one stage; records (stage name, seed) into a shared call list."""

    def __init__(self, name, outputs, calls):
        self.name, self.outputs, self.calls = name, list(outputs), calls

    def __call__(self, seed):
        self.calls.append((self.name, seed))
        return self.outputs.pop(0)


def stages(calls, primary, other, none):
    return (Stage("primary", primary, calls),
            [("other", Stage("other", other, calls)), ("none", Stage("none", none, calls))])


def test_fallbacks_run_in_order_after_seed_retries():
    calls = []
    make, fallbacks = stages(calls, [silent(9)] * 4, [silent(9)], [speech(2)])
    result = generate_with_retry(make, words=8, seed=42, rate=RATE, max_retries=3, fallbacks=fallbacks)
    assert calls == [("primary", 42), ("primary", 43), ("primary", 44), ("primary", 45),
                     ("other", 42), ("none", 42)]
    assert result.ok and result.stage == "none" and result.attempts == 6


def test_stops_at_first_fallback_that_passes():
    calls = []
    make, fallbacks = stages(calls, [silent(9)] * 2, [speech(2)], [speech(2)])
    result = generate_with_retry(make, words=8, seed=0, rate=RATE, max_retries=1, fallbacks=fallbacks)
    assert [name for name, _ in calls] == ["primary", "primary", "other"]
    assert result.ok and result.stage == "other"


def test_good_primary_never_runs_fallbacks():
    calls = []
    make, fallbacks = stages(calls, [speech(2)], [speech(2)], [speech(2)])
    result = generate_with_retry(make, words=8, seed=0, rate=RATE, max_retries=3, fallbacks=fallbacks)
    assert calls == [("primary", 0)] and result.stage == "primary"


def test_best_attempt_is_kept_across_all_stages():
    calls = []
    other_out = speech(0.5)  # most speech of all attempts, still under the 0.3 bar for 8 words
    make, fallbacks = stages(calls, [silent(9), speech(0.1)], [other_out], [speech(0.2)])
    result = generate_with_retry(make, words=8, seed=0, rate=RATE, max_retries=1, fallbacks=fallbacks)
    assert not result.ok and result.stage == "other"
    assert np.array_equal(result.audio, other_out)


def test_no_fallbacks_keeps_old_behaviour():
    calls = []
    make, _ = stages(calls, [silent(9)] * 2, [], [])
    result = generate_with_retry(make, words=8, seed=0, rate=RATE, max_retries=1)
    assert len(calls) == 2 and not result.ok and result.stage == "primary"


def test_log_names_the_winning_stage():
    lines, calls = [], []
    make, fallbacks = stages(calls, [silent(9)], [speech(2)], [speech(2)])
    generate_with_retry(make, words=8, seed=0, rate=RATE, max_retries=0, fallbacks=fallbacks,
                        label="chunk 30", log=lines.append)
    assert any("chunk 30" in line and "other" in line and "ok" in line for line in lines)


def test_log_names_the_kept_stage_when_everything_fails():
    lines, calls = [], []
    make, fallbacks = stages(calls, [silent(9)], [speech(0.3)], [silent(9)])
    generate_with_retry(make, words=8, seed=0, rate=RATE, max_retries=0, fallbacks=fallbacks,
                        label="chunk 30", log=lines.append)
    assert "failed" in lines[-1] and "other" in lines[-1]


def test_fallback_setting_defaults_to_other_then_none():
    assert load_settings({}).fallback == ("other", "none")


@pytest.mark.parametrize("value,expected", [("", ()), ("none", ("none",)), ("other", ("other",)),
                                            (" other , none ", ("other", "none"))])
def test_fallback_setting_parsing(value, expected):
    assert load_settings({"DIA_FALLBACK": value}).fallback == expected


def test_unknown_fallback_stage_rejected():
    with pytest.raises(SystemExit):
        load_settings({"DIA_FALLBACK": "other,random"})
