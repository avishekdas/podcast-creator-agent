"""Dia 1.6B (Nari Labs) via mlx-audio on Apple Silicon: native two-speaker dialogue.

Dia writes a whole exchange in one pass using [S1]/[S2] tags, but one call holds only ~30 s
of audio. So turns are grouped into short chunks (one turn each by default), and every chunk gets
a fixed reference clip (audio + its exact [S1]/[S2] text) so both hosts keep their voices. The
reference is never replaced by generated output, which is what let voices drift in v1.

Defaults ("v3 + retry", see settings.py): two references chosen per chunk so speaker tags
alternate, leading/trailing silence trimmed, and a chunk that comes back silent or cut short is
regenerated with seed + attempt (DIA_MAX_RETRIES, default 3; see retry.py), then once with the
other reference clip and once with none (DIA_FALLBACK=other,none). DIA_VERIFY=1 also
Whisper-checks each chunk. DIA_REF_MODE=single DIA_MAX_RETRIES=0 reproduces v2.
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

from audio_utils import trim_silence  # noqa: E402
from retry import generate_with_retry  # noqa: E402
from settings import load_settings, pick_reference  # noqa: E402

TAGS = {"HOST_A": "[S1]", "HOST_B": "[S2]"}
GAP_SECONDS = 0.25
SETTINGS = load_settings(os.environ)  # v3 + retry defaults; see settings.py
VERIFY_MODEL = "mlx-community/whisper-small.en-asr-fp16"


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


def fallback_stages(ref_path, maker):
    """(name, generator) pairs to try once each after the seed retries, per DIA_FALLBACK.

    "other": the reference clip this chunk did not get (alternate mode only).
    "none": no reference at all.
    """
    others = [(p, t) for p, t in SETTINGS.references if p != ref_path]
    stages = []
    for name in SETTINGS.fallback:
        if name == "other" and others:
            stages.append(("other", maker(*others[0])))
        elif name == "none" and ref_path is not None:
            stages.append(("none", maker(None, None)))
    return stages


def make_verifier(tmp: Path, rate: int):
    """DIA_VERIFY=1: Whisper-transcribe a chunk and score it against its text (0 to 1)."""
    import difflib
    import re

    import soundfile as sf
    from mlx_audio.stt.utils import load as load_stt

    stt = load_stt(VERIFY_MODEL)

    def words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower().replace("-", " "))

    def verify(audio, spoken: str) -> float:
        path = tmp / "verify.wav"
        sf.write(path, audio, rate)
        heard = stt.generate(str(path)).text
        return difflib.SequenceMatcher(a=words(spoken), b=words(heard), autojunk=False).ratio()

    return verify


def synthesize(transcript: Path, out_mp3: Path) -> None:
    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from mlx_audio.tts.utils import load

    model = load(SETTINGS.model)
    rate = model.config.model.sample_rate
    gap = np.zeros(int(rate * GAP_SECONDS), dtype=np.float32)
    groups = chunk(parse(transcript))
    from mlx_audio.utils import load_audio
    loaded = {}  # each reference clip is loaded once and never replaced by generated output
    for path, _ in SETTINGS.references:
        loaded[path] = load_audio(path, sample_rate=rate)
        print(f"  reference: {path} ({loaded[path].shape[0] / rate:.1f}s, mode {SETTINGS.ref_mode})",
              flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        parts: list[Path] = []
        retried: list[int] = []
        failed: list[int] = []
        fallback_wins: list[str] = []
        verifier = make_verifier(Path(tmp), rate) if SETTINGS.verify else None
        for index, group in enumerate(groups):
            text = render(group)
            ref_path, ref_text = pick_reference(SETTINGS.references, TAGS[group[0].speaker])

            def maker(path, tagged, text=text, index=index):
                """Generator for this chunk with one reference clip (or none when path is None)."""
                def make(seed: int) -> np.ndarray:
                    mx.random.seed(seed)
                    audio, _ = model._generate(text, ref_audio=loaded.get(path), ref_text=tagged)
                    samples = np.asarray(audio, dtype=np.float32)
                    if samples.size == 0:
                        raise RuntimeError(f"chunk {index + 1}: Dia returned no audio")
                    mx.clear_cache()
                    if not SETTINGS.trim_silence:
                        return samples
                    trimmed = trim_silence(samples, rate)
                    if samples.size - trimmed.size > rate:
                        print(f"    trimmed {(samples.size - trimmed.size) / rate:.1f}s of silence",
                              flush=True)
                    return trimmed
                return make

            spoken = " ".join(t.text for t in group)
            result = generate_with_retry(
                maker(ref_path, ref_text), words=len(spoken.split()), seed=SETTINGS.seed, rate=rate,
                max_retries=SETTINGS.max_retries, label=f"chunk {index + 1}",
                verify=(lambda audio, spoken=spoken: verifier(audio, spoken)) if verifier else None,
                fallbacks=fallback_stages(ref_path, maker))
            if result.stage != "primary":
                fallback_wins.append(f"{index + 1}:{result.stage}")
            if result.attempts > 1:
                retried.append(index + 1)
            if not result.ok:
                failed.append(index + 1)
            wav = Path(tmp) / f"{index:03d}.wav"
            sf.write(wav, np.concatenate([result.audio, gap]), rate)
            parts.append(wav)
            print(f"  [{index + 1}/{len(groups)}] {len(group)} turns, {result.audio.size / rate:.1f}s"
                  + (f", {result.attempts} attempts" if result.attempts > 1 else ""), flush=True)
        print(f"  retries: {len(retried)} chunks retried {retried or ''}; "
              f"{len(failed)} still bad after retries {failed or ''}; "
              f"kept from a fallback stage: {fallback_wins or 'none'}", flush=True)

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
