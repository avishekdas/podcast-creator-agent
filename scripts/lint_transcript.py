"""Lint a HOST_A/HOST_B transcript against scripts/speech_rules.md.

Usage: python scripts/lint_transcript.py <transcript.txt> [--minutes N]
Prints one line per problem and exits 1 if any; exits 0 when clean. Standard library only.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transcript import Turn, parse  # noqa: E402

RULES_FILE = Path(__file__).resolve().parent / "speech_rules.md"
WORDS_PER_MINUTE = 150
LENGTH_TOLERANCE = 0.15
MAX_TURN_WORDS = 40
MAX_SIGNIFICANT_DIGITS = 3

_UNIT = re.compile(r"\d(?:\.\d+)?[KMB]\b")
_URL = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|org|net|io|dev|ai)\b", re.IGNORECASE)
_SYMBOL = re.compile(r"[%$&/=+*#@<>|\\~^_\[\]{}]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PHONETIC_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|")


@dataclass(frozen=True)
class Problem:
    turn: int  # 1-based turn number; 0 for whole-file problems
    rule: str
    message: str


def load_phonetics(path: Path = RULES_FILE) -> dict[str, str]:
    """Read the 'Written | Spoken' table from the rules file."""
    rows = (_PHONETIC_ROW.match(line) for line in path.read_text(encoding="utf-8").splitlines())
    return {m.group(1): m.group(2) for m in rows if m}


def _significant_digits(number: str) -> int:
    digits = number.replace(",", "").replace(".", "").lstrip("0").rstrip("0")
    return len(digits)


def _is_year(number: str) -> bool:
    return number.isdigit() and 1900 <= int(number) <= 2099


def _turn_problems(index: int, text: str, phonetics: dict[str, str]) -> list[Problem]:
    found: list[Problem] = []
    for match in _UNIT.finditer(text):
        found.append(Problem(index, "unit", f"write the unit in full, not {match.group()!r}"))
    for match in _URL.finditer(text):
        found.append(Problem(index, "url", f"no URLs in spoken lines ({match.group()!r})"))
    for match in _SYMBOL.finditer(text):
        found.append(Problem(index, "symbol", f"spell out symbol {match.group()!r}"))
    for match in _NUMBER.finditer(text):
        number = match.group().rstrip(",")
        if not _is_year(number) and _significant_digits(number) > MAX_SIGNIFICANT_DIGITS:
            found.append(Problem(index, "exact-number", f"round {number!r} (3 significant digits max)"))
    for written, spoken in phonetics.items():
        if re.search(rf"\b{re.escape(written)}\b", text):
            found.append(Problem(index, "phonetic", f"write {written!r} as {spoken!r}"))
    words = len(text.split())
    if words > MAX_TURN_WORDS:
        found.append(Problem(index, "turn-length", f"{words} words (max {MAX_TURN_WORDS})"))
    return found


def lint(turns: list[Turn], minutes: float | None = None,
         phonetics: dict[str, str] | None = None) -> list[Problem]:
    table = load_phonetics() if phonetics is None else phonetics
    problems = [p for i, t in enumerate(turns, 1) for p in _turn_problems(i, t.text, table)]
    if minutes:
        total = sum(len(t.text.split()) for t in turns)
        target = minutes * WORDS_PER_MINUTE
        if abs(total - target) > target * LENGTH_TOLERANCE:
            problems.append(Problem(0, "length", f"{total} words; {minutes:g} min targets "
                                                 f"{target:.0f} (+-{LENGTH_TOLERANCE:.0%})"))
    return problems


def lint_file(path: Path, minutes: float | None = None,
              rules_path: Path = RULES_FILE) -> list[Problem]:
    try:
        turns = parse(path)
    except (ValueError, OSError) as error:
        return [Problem(0, "parse", str(error))]
    return lint(turns, minutes, load_phonetics(rules_path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--minutes", type=float)
    args = parser.parse_args()
    problems = lint_file(args.transcript, args.minutes)
    for p in problems:
        where = f"turn {p.turn}" if p.turn else "file"
        print(f"{where}: {p.rule}: {p.message}")
    if not problems:
        turns = parse(args.transcript)
        print(f"clean: {len(turns)} turns, {sum(len(t.text.split()) for t in turns)} words")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
