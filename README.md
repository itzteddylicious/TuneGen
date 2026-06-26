# TuneGen

TuneGen is an AI-assisted musical co-creator that helps musicians find their next idea.

You play a sequence of notes. TuneGen listens, understands the musical context, and suggests where the music could go next — not as a single answer, but as multiple possibilities with different intentions.

It is not a music generator. It does not compose for you.
It is a thinking tool — something to push back against, react to, and build with.

---

## How it works

Give TuneGen a sequence of notes and it responds with suggestions for what could come next. Each suggestion represents a different musical intention — staying close to the harmony, departing from it, or introducing contrast and surprise.

You decide which direction to follow. TuneGen continues from there.

---

## Project structure

```
TuneGen/
  core/       — ML-based system (current development)
  legacy/     — Rule-based prototype (reference implementation)
```

### core
The main system. Built on an LSTM model trained on the GiantMIDI-Piano dataset — 10,841 classical piano performances by 2,786 composers. The model learns harmonic and melodic patterns from real human performances rather than predefined rules.

### legacy
The original rule-based prototype. Operates within the C Major scale and uses deterministic musical rules to generate three continuations — SAFE, CREATIVE, and UNEXPECTED. Kept as a reference implementation and proof of concept.

---

## Dataset

This project uses the GiantMIDI-Piano dataset by ByteDance, released under the Apache 2.0 license.

- GitHub: https://github.com/bytedance/GiantMIDI-Piano
- 10,855 MIDI files, 1,237 hours of classical piano music

Download the MIDI files and place them in:
```
core/giantmidi/
```

---

## Status

The project is in active development. The core ML model is functional and producing musically meaningful suggestions. Further training, evaluation, and refinement are ongoing.

---

## License

Proprietary. See [LICENSE](LICENSE) for details.