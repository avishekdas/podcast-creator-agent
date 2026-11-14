"""Cut a fixed two-speaker Dia reference clip from an earlier raw Dia output.

Usage: python tools/make_dia_ref.py <raw.wav> <transcript.txt> <out_prefix> [turns=2]
       python tools/make_dia_ref.py --append <base_prefix> <raw.wav> <transcript.txt> <turn> <out_prefix>
The Dia pipeline joins chunks with exactly GAP_SECONDS of digital silence, so chunk boundaries
are the runs of exact zeros. With DIA_CHUNK_WORDS=20 each chunk is one turn. Writes
<out_prefix>.wav and <out_prefix>.txt (the [S1]/[S2]-tagged text of those turns).
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from transcript import parse  # noqa: E402

TAGS = {"HOST_A": "[S1]", "HOST_B": "[S2]"}
MIN_GAP_SECONDS = 0.2
GAP_SECONDS = 0.25  # silence the Dia pipeline puts after every chunk


def gap_starts(audio: np.ndarray, rate: int) -> list[int]:
    edges = np.diff(np.concatenate([[0], (audio == 0).astype(np.int8), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [int(s) for s, e in zip(starts, ends) if (e - s) / rate >= MIN_GAP_SECONDS]


def cut_turn(audio: np.ndarray, rate: int, number: int) -> np.ndarray:
    """Audio of turn `number` (1-based) of a raw Dia run made with one turn per chunk."""
    gaps = gap_starts(audio, rate)
    if not 1 <= number <= len(gaps):
        raise SystemExit(f"turn {number} out of range (found {len(gaps)} turns)")
    start = 0 if number == 1 else gaps[number - 2] + int(rate * GAP_SECONDS)
    return audio[start:gaps[number - 1]]


def append_turn(base_prefix: Path, raw: str, transcript: str, number: int, out_prefix: Path) -> None:
    """Write out_prefix.wav/.txt = base reference + GAP_SECONDS silence + turn `number` of raw."""
    base, rate = sf.read(base_prefix.with_suffix(".wav"), dtype="float32")
    audio, raw_rate = sf.read(raw, dtype="float32")
    if raw_rate != rate:
        raise SystemExit(f"sample rates differ: {rate} vs {raw_rate}")
    turn = parse(transcript)[number - 1]
    clip = np.concatenate([base, np.zeros(int(rate * GAP_SECONDS), np.float32), cut_turn(audio, rate, number)])
    sf.write(out_prefix.with_suffix(".wav"), clip, rate)
    text = f"{base_prefix.with_suffix('.txt').read_text().strip()} {TAGS[turn.speaker]} {turn.text}"
    out_prefix.with_suffix(".txt").write_text(text + "\n")
    print(f"{out_prefix}.wav: {len(clip) / rate:.2f}s\n{text}")


def main() -> None:
    if sys.argv[1] == "--append":
        # --append <base_prefix> <raw.wav> <transcript.txt> <turn_number> <out_prefix>
        base, raw, transcript, number, out = sys.argv[2:7]
        append_turn(Path(base), raw, transcript, int(number), Path(out))
        return
    raw, transcript, prefix = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    turns = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    audio, rate = sf.read(raw, dtype="float32")
    gaps = gap_starts(audio, rate)
    if len(gaps) < turns:
        raise SystemExit(f"found only {len(gaps)} chunk gaps, need {turns}")
    chosen = parse(transcript)[:turns]
    if len({t.speaker for t in chosen}) < 2:
        raise SystemExit("reference must contain both speakers")
    clip = audio[: gaps[turns - 1]]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    sf.write(prefix.with_suffix(".wav"), clip, rate)
    text = " ".join(f"{TAGS[t.speaker]} {t.text}" for t in chosen)
    prefix.with_suffix(".txt").write_text(text + "\n")
    print(f"{prefix}.wav: {len(clip) / rate:.2f}s\n{text}")


if __name__ == "__main__":
    main()
