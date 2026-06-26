"""
preprocess_step2.py — Sliding Window Sequence Builder
======================================================
Step 2 of the TuneGen ML preprocessing pipeline.

What this script does
---------------------
- Loads pitch sequences from step 1 (pitch_sequences.npz)
- Applies a sliding window of 32 notes across each sequence
- Each window produces one training pair:
    input  : 32 consecutive pitches
    output : the single note that follows
- Saves inputs and outputs as numpy arrays in a .npz file

Run
---
    python preprocess_step2.py

Output
------
    data/input_output_pairs.npz  — inputs and outputs as numpy arrays
    data/step2_stats.txt         — summary report

Dependencies
------------
    pip install numpy
"""

import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

INPUT_NPZ    = Path("../data/pitch_sequences.npz")
OUTPUT_NPZ   = Path("../data/input_output_pairs.npz")
OUTPUT_STATS = Path("../data/step2_stats.txt")

WINDOW_SIZE  = 32


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------

def build_pairs(sequences: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a sliding window across every sequence.

    Returns
    -------
    inputs  : numpy array of shape (N, window_size)
    outputs : numpy array of shape (N,)
    """
    inputs  = []
    outputs = []

    for seq in sequences:
        if len(seq) < window_size + 1:
            continue

        for i in range(len(seq) - window_size):
            inputs.append(seq[i : i + window_size])
            outputs.append(seq[i + window_size])

    return np.array(inputs, dtype=np.int16), np.array(outputs, dtype=np.int16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_NPZ.exists():
        print(f"[Error] '{INPUT_NPZ}' not found.")
        print("Run preprocess_step1.py first.")
        return

    print(f"\n  Loading pitch sequences from {INPUT_NPZ}...")
    data      = np.load(INPUT_NPZ, allow_pickle=True)
    sequences = data["sequences"]
    print(f"  Loaded {len(sequences):,} sequences.\n")

    print(f"  Building input/output pairs (window size = {WINDOW_SIZE})...")
    start_time = time.time()

    inputs, outputs = build_pairs(sequences, WINDOW_SIZE)

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s.\n")

    print(f"  Saving to {OUTPUT_NPZ}...")
    np.savez(OUTPUT_NPZ, inputs=inputs, outputs=outputs)

    unique_pitches = len(np.unique(outputs))

    report = f"""
Step 2 — Sliding Window Report
==================================
  Sequences loaded         : {len(sequences):,}
  Window size              : {WINDOW_SIZE} notes
  Total training pairs     : {len(inputs):,}
  Unique output pitches    : {unique_pitches}
  Pitch range              : {outputs.min()} – {outputs.max()}

  inputs shape             : {inputs.shape}
  outputs shape            : {outputs.shape}

  Time taken               : {elapsed:.1f}s
  Output saved to          : {OUTPUT_NPZ}
==================================
"""

    print(report)

    with open(OUTPUT_STATS, "w") as f:
        f.write(report)

    print(f"  Stats saved to {OUTPUT_STATS}\n")


if __name__ == "__main__":
    main()