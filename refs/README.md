`dia-ref.wav` + `dia-ref.txt`: fixed two-speaker voice reference for Dia (default for `engines/dia`).
Cut from the opening of `out/video-likes.dia.raw.wav` (v1) with `tools/make_dia_ref.py`, then trimmed
to 0–8.76 s because the end of the second turn came out garbled ("What am I missing?" was not
spoken). The text was checked word for word against a Whisper transcription of the clip.
