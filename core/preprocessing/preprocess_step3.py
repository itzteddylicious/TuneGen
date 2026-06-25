"""
preprocess_step3.py — Train / Validation / Test Split
======================================================
Step 3 of the TuneGen ML preprocessing pipeline.

What this script does
---------------------
- Loads input/output pairs from step 2 (input_output_pairs.npz)
- Shuffles the data to remove any file/composer ordering bias
- Splits into 80% train, 10% validation, 10% test
- Saves three separate .npz files ready for model training

Run
---
    python preprocess_step3.py

Output
------
    data/train.npz   — 80% of pairs
    data/val.npz     — 10% of pairs
    data/test.npz    — 10% of pairs
    data/step3_stats.txt — summary report

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

INPUT_NPZ    = Path("../data/input_output_pairs.npz")
OUTPUT_DIR   = Path("../data")
OUTPUT_STATS = OUTPUT_DIR / "step3_stats.txt"

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10

RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_NPZ.exists():
        print(f"[Error] '{INPUT_NPZ}' not found.")
        print("Run preprocess_step2.py first.")
        return

    print(f"\n  Loading input/output pairs from {INPUT_NPZ}...")
    data    = np.load(INPUT_NPZ)
    inputs  = data["inputs"]
    outputs = data["outputs"]
    total   = len(inputs)
    print(f"  Loaded {total:,} pairs.\n")

    # ------------------------------------------------------------------
    # Shuffle
    # ------------------------------------------------------------------
    print(f"  Shuffling with seed {RANDOM_SEED}...")
    rng     = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(total)
    inputs  = inputs[indices]
    outputs = outputs[indices]

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    train_end = int(total * TRAIN_RATIO)
    val_end   = int(total * (TRAIN_RATIO + VAL_RATIO))

    train_inputs,  train_outputs  = inputs[:train_end],  outputs[:train_end]
    val_inputs,    val_outputs    = inputs[train_end:val_end], outputs[train_end:val_end]
    test_inputs,   test_outputs   = inputs[val_end:],    outputs[val_end:]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    start_time = time.time()

    np.savez(OUTPUT_DIR / "train.npz", inputs=train_inputs,  outputs=train_outputs)
    np.savez(OUTPUT_DIR / "val.npz",   inputs=val_inputs,    outputs=val_outputs)
    np.savez(OUTPUT_DIR / "test.npz",  inputs=test_inputs,   outputs=test_outputs)

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    report = f"""
Step 3 — Train / Validation / Test Split Report
==================================
  Total pairs              : {total:,}
  Random seed              : {RANDOM_SEED}

  Train                    : {len(train_inputs):,} pairs  ({TRAIN_RATIO*100:.0f}%)
  Validation               : {len(val_inputs):,} pairs  ({VAL_RATIO*100:.0f}%)
  Test                     : {len(test_inputs):,} pairs  ({TEST_RATIO*100:.0f}%)

  Time taken               : {elapsed:.1f}s
  Output saved to          : {OUTPUT_DIR}
==================================
"""

    print(report)

    with open(OUTPUT_STATS, "w") as f:
        f.write(report)

    print(f"  Stats saved to {OUTPUT_STATS}\n")


if __name__ == "__main__":
    main()