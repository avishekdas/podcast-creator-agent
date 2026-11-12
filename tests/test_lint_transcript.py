"""Tests for scripts/lint_transcript.py. Run: ./test.sh"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from lint_transcript import lint, lint_file, load_phonetics  # noqa: E402
from transcript import Turn  # noqa: E402

PHONETICS = {"Redis": "Reddis"}


def rules(problems):
    return [p.rule for p in problems]


def turn(text, speaker="HOST_A"):
    return Turn(speaker, text)


def test_clean_turns_pass():
    turns = [turn("Welcome back. Today we talk about caching."),
             turn("Why does caching matter so much?", "HOST_B")]
    assert lint(turns, phonetics=PHONETICS) == []


@pytest.mark.parametrize("text", ["It gets 1.2M likes.", "About 50K users.", "A 3B parameter model."])
def test_abbreviated_units_flagged(text):
    assert "unit" in rules(lint([turn(text)], phonetics=PHONETICS))


def test_written_out_units_pass():
    assert lint([turn("It gets 1.2 million likes from 50 thousand users.")], phonetics=PHONETICS) == []


@pytest.mark.parametrize("text", ["See https://example.com for more.", "Go to www.example.com now."])
def test_urls_flagged(text):
    assert "url" in rules(lint([turn(text)], phonetics=PHONETICS))


@pytest.mark.parametrize("text", ["It drops 50% of traffic.", "Costs $5 a month.", "Reads & writes.",
                                  "Use the /videos path.", "Set a = b."])
def test_symbols_flagged(text):
    assert "symbol" in rules(lint([turn(text)], phonetics=PHONETICS))


def test_ordinary_punctuation_passes():
    text = "Well, it's fast: about 50 milliseconds. Right? Yes! (Mostly.) The so-called \"edge\"."
    assert lint([turn(text)], phonetics=PHONETICS) == []


@pytest.mark.parametrize("text", ["It has 1,204,311 likes.", "Exactly 4567 requests.", "A ratio of 0.98765."])
def test_long_exact_numbers_flagged(text):
    assert "exact-number" in rules(lint([turn(text)], phonetics=PHONETICS))


@pytest.mark.parametrize("text", ["About 20,000 writes.", "Around 1.25 million.", "Back in 2019.",
                                  "Under 50 milliseconds.", "Maybe 120 servers."])
def test_round_numbers_and_years_pass(text):
    assert lint([turn(text)], phonetics=PHONETICS) == []


def test_phonetic_name_flagged_with_suggestion():
    problems = lint([turn("Store it in Redis.")], phonetics=PHONETICS)
    assert rules(problems) == ["phonetic"]
    assert "Reddis" in problems[0].message


def test_phonetic_spelling_passes():
    assert lint([turn("Store it in Reddis.")], phonetics=PHONETICS) == []


def test_long_turn_flagged():
    assert "turn-length" in rules(lint([turn("word " * 41)], phonetics=PHONETICS))


def test_forty_word_turn_passes():
    assert lint([turn("word " * 40)], phonetics=PHONETICS) == []


def test_word_count_outside_target_flagged():
    turns = [turn("word " * 30)] * 10  # 300 words; 3 minutes targets 450 +-15%
    assert "length" in rules(lint(turns, minutes=3, phonetics=PHONETICS))


def test_word_count_inside_target_passes():
    turns = [turn("word " * 30)] * 15  # 450 words
    assert lint(turns, minutes=3, phonetics=PHONETICS) == []


def test_problems_carry_turn_number():
    problems = lint([turn("Fine."), turn("It has 5K users.")], phonetics=PHONETICS)
    assert problems[0].turn == 2


def test_load_phonetics_reads_rules_table():
    table = load_phonetics(ROOT / "scripts" / "speech_rules.md")
    assert table["Redis"] == "Reddis"


def test_lint_file_reports_parse_error(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("HOST_C: who am I?\n")
    assert rules(lint_file(bad)) == ["parse"]


def test_lint_file_ignores_comment_lines(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("# https://redis.io/docs 1.2M 50%\nHOST_A: Hello there.\nHOST_B: Hi.\n")
    assert lint_file(good) == []


def test_existing_video_likes_script_has_known_problems():
    found = set(rules(lint_file(ROOT / "transcripts" / "video-likes.txt")))
    assert {"phonetic", "unit", "exact-number", "turn-length"} <= found
