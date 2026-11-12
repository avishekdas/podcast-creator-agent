"""Kokoro-82M baseline: one preset voice per speaker, line by line, joined with ffmpeg."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from transcript import parse  # noqa: E402

SAMPLE_RATE = 24_000
VOICES = {"HOST_A": os.environ.get("KOKORO_VOICE_A", "af_heart"),
          "HOST_B": os.environ.get("KOKORO_VOICE_B", "am_michael")}
GAP_SECONDS = 0.35


def synthesize(transcript: Path, out_mp3: Path, device: str) -> None:
    import numpy as np
    import soundfile as sf
    from kokoro import KPipeline

    turns = parse(transcript)
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device)
    gap = np.zeros(int(SAMPLE_RATE * GAP_SECONDS), dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for index, turn in enumerate(turns):
            chunks = [audio.numpy() if hasattr(audio, "numpy") else audio
                      for _, _, audio in pipeline(turn.text, voice=VOICES[turn.speaker])]
            if not chunks:
                raise RuntimeError(f"turn {index}: Kokoro returned no audio")
            wav = Path(tmp) / f"{index:03d}.wav"
            sf.write(wav, np.concatenate([*chunks, gap]), SAMPLE_RATE)
            parts.append(wav)
            print(f"  [{index + 1}/{len(turns)}] {turn.speaker} ({VOICES[turn.speaker]})", flush=True)

        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-codec:a", "libmp3lame", "-q:a", "2", str(out_mp3)],
                       check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("out_mp3", type=Path)
    parser.add_argument("--device", default=os.environ.get("KOKORO_DEVICE", "cpu"),
                        choices=["cpu", "mps"])
    args = parser.parse_args()
    started = time.perf_counter()
    synthesize(args.transcript, args.out_mp3, args.device)
    print(f"kokoro: wrote {args.out_mp3} in {time.perf_counter() - started:.1f}s on {args.device}")


if __name__ == "__main__":
    main()
