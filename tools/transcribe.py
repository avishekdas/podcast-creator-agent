"""Check an output by ear-proxy: transcribe an MP3 with Whisper small.en (MLX) and print the text."""
import sys

from mlx_audio.stt.utils import load

model = load("mlx-community/whisper-small.en-asr-fp16")
for path in sys.argv[1:]:
    print(f"== {path}\n{model.generate(path).text.strip()}\n")
