"""Word-level diff between a transcript script and a Whisper transcription of its audio.

Usage: python tools/compare.py <transcript.txt> <audio.mp3> [...]
Prints each run of words that differ (script -> heard), so misread acronyms and numbers stand out.
"""
import difflib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from transcript import parse  # noqa: E402
from mlx_audio.stt.utils import load  # noqa: E402


def words(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*%?|[a-z0-9]+", text)


def main() -> None:
    script = words(" ".join(t.text for t in parse(sys.argv[1])))
    model = load("mlx-community/whisper-small.en-asr-fp16")
    for audio in sys.argv[2:]:
        heard_text = model.generate(audio).text.strip()
        Path(audio).with_suffix(".whisper.txt").write_text(heard_text + "\n")
        heard = words(heard_text)
        matcher = difflib.SequenceMatcher(a=script, b=heard, autojunk=False)
        diffs = [(" ".join(script[i1:i2]) or "-", " ".join(heard[j1:j2]) or "-")
                 for op, i1, i2, j1, j2 in matcher.get_opcodes() if op != "equal"]
        print(f"== {audio}: {len(script)} script words, {len(heard)} heard, "
              f"match ratio {matcher.ratio():.3f}, {len(diffs)} differences")
        for said, got in diffs:
            print(f"  {said!r} -> {got!r}")


if __name__ == "__main__":
    main()
