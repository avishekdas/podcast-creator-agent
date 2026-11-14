"""Dia 1.6B (Nari Labs) via mlx-audio on Apple Silicon: native two-speaker dialogue.

Dia writes a whole exchange in one pass using [S1]/[S2] tags, but one call holds only ~30 s
of audio. So turns are grouped into short chunks, and every chunk gets the same fixed
two-speaker reference clip (audio + its exact [S1]/[S2] text) so both hosts keep their voices.
The reference is never replaced by generated output, which is what let voices drift before.

Reference: DIA_REF_AUDIO (WAV) + DIA_REF_TEXT (the exact tagged text, or a path to a .txt).
Default: refs/dia-ref.wav + refs/dia-ref.txt. DIA_REF_AUDIO=none disables it.
Defaults (v2, picked by ear): DIA_CHUNK_WORDS=20, DIA_TEMPO=0.75. All settings: settings.py.
"""
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
from transcript import Turn, parse  # noqa: E402

from settings import load_settings  # noqa: E402

TAGS = {"HOST_A": "[S1]", "HOST_B": "[S2]"}
GAP_SECONDS = 0.25
SETTINGS = load_settings(os.environ)  # v2 defaults; see settings.py


def chunk(turns: list[Turn]) -> list[list[Turn]]:
    """Group consecutive turns into chunks of at most SETTINGS.chunk_words words."""
    chunks: list[list[Turn]] = []
    current: list[Turn] = []
    words = 0
    for turn in turns:
        size = len(turn.text.split())
        if current and words + size > SETTINGS.chunk_words:
            chunks.append(current)
            current, words = [], 0
        current = [*current, turn]
        words += size
    return [*chunks, current] if current else chunks


def render(turns: list[Turn]) -> str:
    return " ".join(f"{TAGS[t.speaker]} {t.text}" for t in turns)


def synthesize(transcript: Path, out_mp3: Path) -> None:
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.utils import load

    model = load(SETTINGS.model)
    rate = model.config.model.sample_rate
    gap = np.zeros(int(rate * GAP_SECONDS), dtype=np.float32)
    groups = chunk(parse(transcript))
    ref_path, ref_text = SETTINGS.ref_audio, SETTINGS.ref_text
    ref_audio = None
    if ref_path:
        from mlx_audio.utils import load_audio
        ref_audio = load_audio(ref_path, sample_rate=rate)  # loaded once, same for every chunk
        print(f"  reference: {ref_path} ({ref_audio.shape[0] / rate:.1f}s)", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        parts: list[Path] = []
        for index, group in enumerate(groups):
            mx.random.seed(SETTINGS.seed)
            text = render(group)
            audio, _ = model._generate(text, ref_audio=ref_audio, ref_text=ref_text)
            samples = np.asarray(audio, dtype=np.float32)
            if samples.size == 0:
                raise RuntimeError(f"chunk {index}: Dia returned no audio")
            wav = Path(tmp) / f"{index:03d}.wav"
            sf.write(wav, np.concatenate([samples, gap]), rate)
            parts.append(wav)
            print(f"  [{index + 1}/{len(groups)}] {len(group)} turns, {samples.size / rate:.1f}s",
                  flush=True)
            mx.clear_cache()

        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        out_mp3.parent.mkdir(parents=True, exist_ok=True)
        if SETTINGS.keep_raw:  # untouched audio, so other tempos can be made without regenerating
            raw = out_mp3.with_suffix(".raw.wav")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                            "-i", str(listing), str(raw)], check=True)
            print(f"  raw audio kept at {raw}")
        tempo_filter = [] if SETTINGS.tempo == 1.0 else ["-af", f"atempo={SETTINGS.tempo}"]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), *tempo_filter,
                        "-codec:a", "libmp3lame", "-q:a", "2", str(out_mp3)],
                       check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument("out_mp3", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    synthesize(args.transcript, args.out_mp3)
    print(f"dia: wrote {args.out_mp3} in {time.perf_counter() - started:.1f}s ({SETTINGS.model}, chunks of {SETTINGS.chunk_words} words, tempo {SETTINGS.tempo})")


if __name__ == "__main__":
    main()
