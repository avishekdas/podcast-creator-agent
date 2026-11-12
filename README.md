# Podcast TTS bake-off

Turn a two-speaker transcript into one MP3 per TTS engine, so the engines can be compared by ear.
Everything runs locally. No paid API keys.

## Use

```bash
./setup.sh kokoro dia                      # one time: creates engines/<name>/.venv (needs uv, ffmpeg)
./run.sh transcripts/sample.txt            # -> out/sample.kokoro.mp3
./run.sh transcripts/sample.txt kokoro dia # -> also out/sample.dia.mp3
```

Transcript format: one turn per line, `HOST_A: ...` or `HOST_B: ...`. Blank lines and `#` comments
are ignored; any other line is an error. Sample: `transcripts/sample.txt` (156 words, ~60 s).

Each engine has its own virtual environment under `engines/<name>/.venv`. `run.sh` points
`HF_HOME` at `models/hf`, so every model download stays in this folder.

Knobs (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `KOKORO_VOICE_A` / `KOKORO_VOICE_B` | `af_heart` / `am_michael` | Kokoro preset voices |
| `KOKORO_DEVICE` | `cpu` | `cpu` or `mps` (CPU was faster on M1, see below) |
| `DIA_MODEL` | `mlx-community/Dia-1.6B-fp16` | MLX port of Dia |
| `DIA_CHUNK_WORDS` | `45` | words per Dia call (one call holds ~30 s of audio) |
| `DIA_SEED` | `42` | sampling seed |
| `DIA_TEMPO` | `1.0` | slow (<1) or speed up the Dia output with ffmpeg `atempo`, 0.5–2.0 |

## Results

Test machine: Apple M1, 16 GB RAM, macOS. Same transcript for every engine.

| Engine | Result | Audio | Wall time | License | Hardware |
|---|---|---|---|---|---|
| Kokoro-82M | Works. `out/sample.kokoro.mp3` | 63 s | 24 s CPU, 32 s MPS | Apache-2.0 (code and weights) | Any Mac CPU |
| MOSS-TTSD v1.0 | Not run | | | Apache-2.0 | NVIDIA GPU, see below |
| SoulX-Podcast-1.7B | Not run | | | Apache-2.0 | NVIDIA GPU, see below |
| Dia 1.6B (MLX) | Works. `out/sample.dia.mp3` | 32 s | 393 s (6.5 min) | Apache-2.0 (weights), MIT (mlx-audio) | Apple Silicon via MLX, ~6 GB RAM |
| OpenVoice V2 | Not run: no reference clip supplied | | | MIT | CPU or MPS |

Wall times exclude the first-run model download.

### Kokoro (baseline)

- One preset voice per speaker, generated line by line, 0.35 s pause between turns, joined with ffmpeg.
- A Whisper transcription of the output matches the script almost word for word.
- CPU beat MPS on the M1 (24 s vs 32 s for 63 s of audio). CPU is the default.
- Turn-taking is mechanical: every pause is the same length and there is no overlap or laughter.
- Disk: model ~313 MB, venv ~875 MB (PyTorch, spaCy).

### Dia 1.6B (native dialogue, via `mlx-audio`)

- Runs on the Mac through Apple's MLX, using `mlx-audio` 0.6.5 and the `mlx-community/Dia-1.6B-fp16`
  weights. The upstream `nari-labs/dia` PyTorch code targets CUDA. An MPS pull request was opened
  upstream but was not merged at the time of writing.
- One Dia call holds about 30 s of audio, so turns are grouped into chunks of up to 45 words. The
  first chunk's audio is reused as an audio prompt for later chunks so the two voices stay the same.
- Whisper transcription of the output is nearly complete (it dropped the word "models" once).
- **Speech is fast**: 156 words in 32 s, about 300 words a minute (Kokoro takes 63 s). Dia is
  known to rush.
- **Shorter chunks barely help.** With `DIA_CHUNK_WORDS=20` (one turn per call) the same script
  came out at 36 s (about 260 words a minute) in 315 s of wall time. The first chunk, a single
  26-word turn with no audio prompt, takes 5.4 s either way, so the rushing comes from Dia's
  sampling, not from chunk length. Both versions are kept: `out/sample.dia.chunk45.mp3` and
  `out/sample.dia.chunk20.mp3`. Untried: lower temperature or CFG scale, or an audio prompt of
  slower speech.
- **Slowing the finished audio works.** ffmpeg's `atempo` filter stretches time and keeps pitch.
  Applied to the 20-word-chunk version:

  | Tempo | Length | Words a minute | File |
  |---|---|---|---|
  | 1.0 | 36 s | ~260 | `out/sample.dia.chunk20.mp3` |
  | 0.85 | 43 s | ~220 | `out/sample.dia.chunk20.tempo0.85.mp3` |
  | 0.75 | 48 s | ~195 | `out/sample.dia.chunk20.tempo0.75.mp3` |
  | 0.65 | 56 s | ~170 | `out/sample.dia.chunk20.tempo0.65.mp3` |

  Whisper still transcribes 0.75 and 0.65 correctly; how natural they sound has to be judged by ear.
  These files were re-encoded from the MP3. To apply it inside the pipeline (from the raw WAVs),
  set `DIA_TEMPO`, e.g. `DIA_TEMPO=0.75 DIA_CHUNK_WORDS=20 ./run.sh transcripts/sample.txt dia`.
- Whether the two voices stay consistent across chunks has **not** been checked by ear.
- Slow on M1: about 12x slower than real time.
- Disk: Dia + DAC codec ~3.8 GB, venv ~450 MB.

### MOSS-TTSD (OpenMOSS): stopped, needs a GPU

- License Apache-2.0 (Hugging Face model card).
- v1.0 is an 8B model: 16.7 GB of bf16 weights. It does not fit in 16 GB of unified memory.
- The official install is CUDA-only (conda, Python 3.10, `flash-attn`).
- Unofficial MLX and GGUF ports exist (`actnowai/MOSS-TTSD-v1.0-MLX-8bit`, 10.9 GB). They come from
  small third-party publishers and were not tried.
- Would need: one NVIDIA GPU with 24 GB or more (for example an L4, A10G, or RTX 4090), or a
  cloud instance such as AWS `g5.xlarge` or `g6.xlarge`. The upstream README reports under 7 GB VRAM
  for older, smaller versions; the v0.7 model (4.1 GB) would fit a 16 GB card.

### SoulX-Podcast-1.7B: stopped, needs a GPU

- License Apache-2.0. Weights 5.8 GB.
- Official install targets Linux with CUDA (PyTorch cu121 wheels). Only an unofficial Gitee fork
  claims CPU/MPS support; not tried.
- Would need: one NVIDIA GPU with 16 GB or more (T4, L4, A10G), for example AWS `g4dn.xlarge` or
  `g6.xlarge`.

### OpenVoice V2: not run

Optional. It changes the timbre of existing speech to match a short reference recording. Supply a
10–30 s clean voice clip per host and it can be applied on top of the Kokoro output.

## Long-form test

Script: `transcripts/video-likes.txt`, "system design blueprint for handling high-throughput video
likes". 49 turns, 1,199 words, sources in the comment lines at the top. It deliberately includes
acronyms (API, TTL, QPS, P99, CDC), numbers and product names to stress the engines.

| Output | Audio | Words a minute | Generation time |
|---|---|---|---|
| `out/video-likes.kokoro.mp3` | 490 s (8.2 min) | 147 | 114 s |
| `out/video-likes.dia.mp3` (tempo 0.75) | 394 s (6.6 min) | 182 | 2,453 s (41 min) |
| `out/video-likes.dia.tempo0.85.mp3` | 348 s (5.8 min) | 207 | from the raw audio |
| `out/video-likes.dia.raw.wav` (untouched) | 296 s | 243 | |

Dia ran with `DIA_TEMPO=0.75 DIA_CHUNK_WORDS=20 DIA_KEEP_RAW=1`. Other tempos can be made from the
raw WAV: `ffmpeg -i out/video-likes.dia.raw.wav -af atempo=0.8 out/x.mp3`.

Whisper check (`tools/compare.py`, script vs. transcription, word level):

| | Kokoro | Dia (0.75) |
|---|---|---|
| Word match | 97.2% | 94.6% |
| API, TTL, QPS, P99, CDC | all correct | all correct |
| Kafka, Kinesis, DynamoDB, Cassandra | correct | correct, except "Cassandra" once after "Redis" |
| "1.2 million likes per minute", "50 milliseconds" | correct | correct |
| Redis | wrong every time ("Riddies") | wrong every time ("read as", "redder") |
| "1.2M" | number dropped, heard as "M" | garbled |
| "1,204,311 likes or 1,204,350" | correct | garbled ("1,204,304,304,304") |
| Other | none | "idempotent" garbled once; one phrase spoken twice; a few dropped or swapped words ("video" became "row") |

Most of the remaining differences are not errors: spelled-out numbers come back as digits
("twenty thousand" as "20,000") and homophones ("write" and "right"). Whisper may also mishear
some stretched Dia audio, so treat the Dia list as an upper bound.

Script fixes that would help both engines: spell Redis phonetically (for example "Reddis"), write
"1.2 million" instead of "1.2M", and avoid long exact numbers.

Voice drift: `out/video-likes.dia.first30.mp3` and `out/video-likes.dia.last30.mp3` are the first
and last 30 s of the Dia output. Whether the voices stay the same over six minutes **has to be
judged by ear**; no tool here checks it.

## Research to script

A Claude Code skill in this folder turns a topic into a checked transcript, and optionally audio:

```
/podcast-script <topic> [--minutes N] [--engine kokoro|dia]
```

Open Claude Code in this folder to use it (`.claude/skills/podcast-script/SKILL.md`). It uses
Claude's own web search and fetch, so it needs no API key. Steps: research 3–6 reputable sources,
outline, write `transcripts/<slug>.txt` (sources in `#` lines, turns under 40 words, about 150
words a minute), lint until clean, then show the script. Audio is generated only with `--engine`;
Dia runs in the background with the fixed voice reference and refuses to start if another Dia job
is running.

- `scripts/speech_rules.md`: the speech rules (from the Whisper findings above) plus a phonetic
  table (`Redis` is written `Reddis`). Edit it to add names; the lint reads the table.
- `scripts/lint_transcript.py <file> [--minutes N]`: flags abbreviated units (`1.2M`), URLs,
  symbols, numbers with more than 3 significant digits (years are fine), names from the phonetic
  table, turns over 40 words, and length outside ±15% of the target. Standard library only.
- `./test.sh`: 32 pytest tests for the lint (pytest is fetched by uv; no engine venv needed).

First run: "how CDNs cache video segments", 3 minutes, Kokoro.

| | |
|---|---|
| Transcript | `transcripts/cdn-video-segment-caching.txt`, 25 turns, 464 words, lint clean on first pass |
| Sources | RFC 8216 (HLS), Fastly streaming guidelines, CloudFront Origin Shield docs |
| Audio | `out/cdn-video-segment-caching.kokoro.mp3`, 194 s (3.2 min), 65 s to generate* |
| Whisper match | 99.5%; the three differences are not errors ("forty" as "40", "shields" as "shield", one "and") |

\* Kokoro shared the machine with a Dia run, so it was slower than its usual ~35 s for this length.
A Netflix Open Connect source was dropped because its pages could not be read (one refused the
request, one was a PDF with no extractable text); the skill does not cite unread pages.

## Layout

```
run.sh, setup.sh          entry points
scripts/transcript.py     shared transcript parser
scripts/lint_transcript.py, speech_rules.md   transcript lint and speech rules
.claude/skills/podcast-script/   /podcast-script skill (research to script)
tests/, test.sh           lint tests
refs/                     Dia voice reference clip
engines/kokoro/           generate.py + .venv
engines/dia/              generate.py + .venv
tools/transcribe.py       Whisper check of an output MP3 (uses the Dia venv)
models/hf/                all Hugging Face downloads (~4.1 GB now, incl. Whisper small ~0.5 GB)
out/                      generated MP3s
```

## Disk note

The machine had 25 GB free at the start and 16 GB at the end. About 5.5 GB is this folder; the
rest is the shared `uv` package cache (`~/.cache/uv`, 44 GB in total, mostly from before).
`uv cache prune` frees space there.
