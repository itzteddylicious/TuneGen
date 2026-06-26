"""
preprocess_step1.py — MIDI Pitch Extraction
=============================================
Step 1 of the TuneGen ML preprocessing pipeline.

What this script does
---------------------
- Walks through all .mid files in the dataset folder
- Parses each file with pretty_midi
- Extracts pitch sequences (MIDI note numbers 0-127)
- Reports stats so you can verify everything looks correct
- Saves extracted sequences to a .npz file for the next step

Run
---
    python preprocess_step1.py

Output
------
    data/pitch_sequences.npz   — numpy array of pitch sequences
    data/step1_stats.txt       — summary report

Dependencies
------------
    pip install pretty_midi numpy
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pretty_midi

warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_DIR = Path("../giantmidi")
OUTPUT_DIR   = Path("../data")
OUTPUT_NPZ   = OUTPUT_DIR / "pitch_sequences.npz"
OUTPUT_STATS = OUTPUT_DIR / "step1_stats.txt"

MIN_NOTES = 10


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_pitches(midi_path: Path) -> np.ndarray | None:
    """
    Parse a MIDI file and return a numpy array of pitch values
    sorted by note onset time.

    Returns None if the file is unreadable or has too few notes.
    """
    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))

        notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                notes.append((note.start, note.pitch))

        notes.sort(key=lambda x: x[0])
        pitches = [pitch for _, pitch in notes]

        if len(pitches) < MIN_NOTES:
            return None

        return np.array(pitches, dtype=np.int16)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    midi_files   = sorted(DATASET_DIR.glob("*.mid"))
    total_files  = len(midi_files)

    if total_files == 0:
        print(f"[Error] No .mid files found in '{DATASET_DIR}'.")
        print("Make sure the dataset folder is inside your core/ directory.")
        return

    print(f"\n  Found {total_files} MIDI files. Starting extraction...\n")

    sequences     = []
    failed        = 0
    skipped_short = 0
    total_notes   = 0
    start_time    = time.time()

    for i, midi_path in enumerate(midi_files, 1):
        pitches = extract_pitches(midi_path)

        if pitches is None:
            if midi_path.stat().st_size < 1000:
                skipped_short += 1
            else:
                failed += 1
            continue

        sequences.append(pitches)
        total_notes += len(pitches)

        if i % 500 == 0 or i == total_files:
            elapsed = time.time() - start_time
            print(f"  [{i}/{total_files}] "
                  f"{len(sequences)} sequences extracted "
                  f"({elapsed:.1f}s elapsed)")

    # Save as numpy object array (sequences vary in length)
    sequences_array = np.array(sequences, dtype=object)
    np.savez(OUTPUT_NPZ, sequences=sequences_array)

    elapsed_total = time.time() - start_time
    lengths       = [len(s) for s in sequences]
    avg_notes     = total_notes // len(sequences) if sequences else 0
    min_len       = min(lengths) if lengths else 0
    max_len       = max(lengths) if lengths else 0

    report = f"""
Step 1 — Pitch Extraction Report
==================================
  Total MIDI files found   : {total_files}
  Successfully extracted   : {len(sequences)}
  Failed (unreadable)      : {failed}
  Skipped (too short)      : {skipped_short}

  Total notes extracted    : {total_notes:,}
  Average notes per file   : {avg_notes:,}
  Shortest sequence        : {min_len:,} notes
  Longest sequence         : {max_len:,} notes

  Time taken               : {elapsed_total:.1f}s
  Output saved to          : {OUTPUT_NPZ}
==================================
"""

    print(report)

    with open(OUTPUT_STATS, "w") as f:
        f.write(report)

    print(f"  Stats saved to {OUTPUT_STATS}\n")


if __name__ == "__main__":
    main()