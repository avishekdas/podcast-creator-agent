"""Parse a two-speaker transcript: one turn per line, 'HOST_A:' or 'HOST_B:' prefix."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SPEAKERS = ("HOST_A", "HOST_B")
_LINE = re.compile(r"^\s*(HOST_A|HOST_B)\s*:\s*(.+?)\s*$")


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


def parse(path: str | Path) -> list[Turn]:
    turns: list[Turn] = []
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        match = _LINE.match(raw)
        if not match:
            raise ValueError(f"line {number}: expected 'HOST_A:' or 'HOST_B:' prefix, got {raw[:40]!r}")
        turns.append(Turn(match.group(1), match.group(2)))
    if not turns:
        raise ValueError(f"{path}: no HOST_A/HOST_B lines found")
    return turns
