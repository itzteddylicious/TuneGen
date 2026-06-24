"""
main.py — TuneGen Entry Point
==============================
Wires the engine and player together.

Run modes
---------
  python main.py                            # plays all demo sequences with audio
  python main.py --no-audio                 # prints analysis only, no audio
  python main.py --seq C4 E4 G4            # custom sequence with audio
  python main.py --seq C4 E4 G4 --no-audio # custom sequence, no audio

Note timing per mode (edit DURATION_PROFILES in player.py to adjust)
  INPUT      — 0.50s  neutral reference
  SAFE       — 1.00s  long, legato
  CREATIVE   — 0.60s  medium, flowing
  UNEXPECTED — 0.25s  short, staccato

Folder layout expected
----------------------
  tunegen/
    engine.py
    player.py
    main.py        ← you are here
    sounds/
      C4.wav
      D4.wav
      E4.wav
      F4.wav
      G4.wav
      A4.wav
      B4.wav
      C5.wav
"""

import argparse
import sys

from engine import generate, TuneGenResult
from player import Player


# ---------------------------------------------------------------------------
# Display helper (no audio dependency)
# ---------------------------------------------------------------------------

def display(result: TuneGenResult) -> None:
    bar = "─" * 52
    print(f"\n{'═' * 52}")
    print(f"  TuneGen — Rule-Based Music Generator")
    print(f"{'═' * 52}")
    print(f"  Input Sequence : {' → '.join(result.input_sequence)}")
    print(f"  Chord Detected : {result.chord_detected}")
    print(f"  Chord Tones    : {', '.join(result.chord_tones)}")
    print(f"  Contour        : {result.contour}")
    print(f"  Avg Interval   : {result.avg_interval} scale steps")
    print(f"{bar}")
    print(f"  SAFE        : {' → '.join(result.safe)}")
    print(f"  CREATIVE    : {' → '.join(result.creative)}")
    print(f"  UNEXPECTED  : {' → '.join(result.unexpected)}")
    print(f"{'═' * 52}")


# ---------------------------------------------------------------------------
# Demo sequences
# ---------------------------------------------------------------------------

DEMO_SEQUENCES: list[tuple[str, list[str]]] = [
    ("C Major triad (ascending)",   ["C4", "E4", "G4"]),
    ("F Major chord tones",         ["F4", "A4", "C5"]),
    ("G Major chord tones",         ["G4", "B4", "D4"]),
    ("Descending C Major fragment", ["G4", "E4", "C4"]),
    ("Mixed / passing tones",       ["D4", "F4", "A4"]),
    ("Static root",                 ["C4", "C4", "C4"]),
    ("Arch contour",                ["C4", "G4", "E4"]),
    ("Valley contour",              ["G4", "C4", "E4"]),
]


# ---------------------------------------------------------------------------
# Run with audio
# ---------------------------------------------------------------------------

def run_with_audio(sequences: list[tuple[str, list[str]]]) -> None:
    """Generate and play each sequence through the Player."""
    try:
        with Player(sounds_dir="sounds") as player:
            for label, seq in sequences:
                print(f"\n>>> {label}")
                result = generate(seq)
                display(result)
                player.play_result(result)
    except FileNotFoundError as exc:
        print(f"\n[Audio Error] {exc}")
        print("Tip: run with --no-audio to skip playback.\n")
        sys.exit(1)
    except ImportError as exc:
        print(f"\n[Import Error] {exc}\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Run without audio
# ---------------------------------------------------------------------------

def run_no_audio(sequences: list[tuple[str, list[str]]]) -> None:
    """Generate and display each sequence — no audio required."""
    for label, seq in sequences:
        print(f"\n>>> {label}")
        result = generate(seq)
        display(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TuneGen — Rule-Based Music Generator"
    )
    parser.add_argument(
        "--seq",
        nargs=3,
        metavar=("NOTE1", "NOTE2", "NOTE3"),
        help="Custom 3-note input sequence, e.g. --seq C4 E4 G4",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Print analysis only; skip audio playback",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sequences = (
        [("Custom sequence", args.seq)]
        if args.seq
        else DEMO_SEQUENCES
    )

    if args.no_audio:
        run_no_audio(sequences)
    else:
        run_with_audio(sequences)


if __name__ == "__main__":
    main()