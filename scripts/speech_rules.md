# Speech rules for podcast transcripts

Rules for writing lines that TTS engines (Kokoro, Dia) read correctly. They come from Whisper
checks of earlier runs (README, "Long-form test"). `scripts/lint_transcript.py` enforces the ones
that can be checked mechanically, and reads the phonetic table below, so edit this file to add names.

## Rules

1. **Spell tricky names phonetically.** Both engines said "Redis" wrong every time. Use the
   spoken form from the table below in dialogue (the `#` source lines can keep the real name).
2. **Write units in full.** "1.2 million", not "1.2M"; "50 thousand", not "50K". Kokoro dropped
   the number in "1.2M"; Dia garbled it.
3. **Round numbers.** No long exact figures: "about 1.2 million", not "1,204,311". At most three
   significant digits. Years (1900–2099) are fine.
4. **No symbols or URLs in spoken lines.** Write "percent", "dollars", "slash", "and", "plus".
   URLs belong only in `#` comment lines.
5. **Expand rare abbreviations on first use.** "CDC, change data capture". Common ones (API, CPU,
   GPU, URL, HTTP) need no expansion. Not linted; the writer checks this.
6. **Short turns.** At most 40 words per turn, so each Dia chunk holds one turn.
7. **Don't end a HOST_B turn on a key term.** Dia often drops the last word of a HOST_B turn
   (README, "Drift fix (v2)"). Put names and numbers mid-sentence: "Why call Reddis directly?",
   not "Why call Reddis?". The lint flags a HOST_B turn whose last word is a number, a number
   followed by a unit word ("50 thousand"), or a capitalized name or acronym.
8. **Length.** Aim for 150 words per minute of target audio, within ±15%.

## Phonetic spellings

The lint flags the left column inside spoken lines and suggests the right one.

| Written | Spoken |
|---|---|
| `Redis` | `Reddis` |
| `nginx` | `engine x` |
| `PostgreSQL` | `Postgres` |
| `kubectl` | `cube control` |
| `SQLite` | `S Q Lite` |
