---
name: podcast-script
description: Research a topic on the web and write a two-host podcast transcript (HOST_A/HOST_B) that the local TTS engines read well, then optionally generate audio with Kokoro or Dia. Use for "/podcast-script <topic>", "write a podcast episode about X", or "turn this topic into a podcast script".
argument-hint: <topic> [--minutes N] [--engine kokoro|dia]
---

# Podcast script: research to transcript to audio

Arguments: `$ARGUMENTS`

Parse them as: topic = everything except the flags; `--minutes N` (default 6); `--engine kokoro|dia`
(default: none, so stop after the script for review). All paths below are relative to the
`podcast-tts-bakeoff` folder that holds this skill. Uses Claude's own WebSearch and WebFetch: no
API keys. Never add paid API keys.

Before starting, read `scripts/speech_rules.md`. It is the source of truth for how lines must be
written and may have changed since this skill was written.

## 1. Research

- Run 3 to 6 web searches on the topic, sent together where possible.
- Keep 3 to 6 reputable sources: official documentation, engineering blogs from the company that
  built the system, standards (RFCs), or papers. Skip SEO listicles and AI-generated summaries.
- Open (WebFetch) the pages you rely on; do not cite a page only from its search snippet.
- Note each key fact with the source it came from.
- **Never invent statistics.** A number with no source must be marked illustrative in the
  dialogue itself ("to pick an illustrative number, say..."). Add a `#` line saying so.

## 2. Outline

Hook, then 4 to 6 sections, then a one-paragraph recap and sign-off. Target length is
`minutes x 150` words (±15%). Plan roughly how many words each section gets.

## 3. Write `transcripts/<slug>.txt`

- `<slug>`: lowercase topic, words joined by hyphens, at most about 5 words
  (for example `cdn-video-segment-caching`). Do not overwrite an existing file; add `-2`.
- Header: `#` comment lines with the topic, the illustrative-numbers note if any, and one
  `# <url>  (what it supports)` line per source. URLs appear only in `#` lines.
- Body: one turn per line, `HOST_A: ...` or `HOST_B: ...`.
  - HOST_A is the explainer. HOST_B is the curious co-host who asks questions, restates things
    in plain words, and pushes back with real objections; HOST_A answers them.
  - Alternate speakers. Keep every turn under 40 words, so one Dia chunk holds one turn.
  - Must be technically correct and consistent with the sources.
- Follow every rule in `scripts/speech_rules.md` (phonetic names, units in full, round numbers,
  no symbols or URLs in spoken lines, expand rare abbreviations on first use).

## 4. Validate

```bash
python3 scripts/lint_transcript.py transcripts/<slug>.txt --minutes <N>
```

Fix every reported problem and re-run until it prints `clean`. The lint cannot check
abbreviation expansion or facts: re-read the script once for both.

Then show the user the transcript path, word count, source list, and the full script text.

## 5. Generate (only when `--engine` was given)

- **kokoro** (about 2 minutes for 6 minutes of audio), run inline:
  ```bash
  ./run.sh transcripts/<slug>.txt kokoro
  ```
  Output: `out/<slug>.kokoro.mp3`.
- **dia** (about 6 to 7 minutes of compute per minute of audio on an M1). First check no other Dia
  job is running: `pgrep -f engines/dia/generate.py`. If one is, stop and tell the user rather than
  starting a second. Otherwise run it in the background:
  ```bash
  DIA_TEMPO=0.75 DIA_CHUNK_WORDS=20 DIA_KEEP_RAW=1 ./run.sh transcripts/<slug>.txt dia
  ```
  It uses the fixed two-speaker voice reference in `refs/dia-ref.wav` automatically. Output:
  `out/<slug>.dia.mp3` plus `out/<slug>.dia.raw.wav`.

After generating, check pronunciation with Whisper and report the match percentage and any
misread names or numbers:

```bash
HF_HOME=$PWD/models/hf engines/dia/.venv/bin/python tools/compare.py transcripts/<slug>.txt out/<slug>.<engine>.mp3
```

Spelled-out numbers coming back as digits ("twenty" as "20") and homophones ("write"/"right") are
not errors. If a name is misread, add it to the phonetic table in `scripts/speech_rules.md`.

## Report

Transcript path, word count, sources, lint result, and if generated: MP3 path, audio length,
generation time, and the Whisper match percentage. Commit nothing.
