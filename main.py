"""
main.py — TuneGen Entry Point
==============================
Wires the engine and player together.

Run modes
---------
  python main.py                                   # demo sequences, no chain
  python main.py --no-audio                        # demo, no audio
  python main.py --seq C4 E4 G4                   # custom sequence, no chain
  python main.py --seq C4 E4 G4 --chain           # chain mode with audio
  python main.py --seq C4 E4 G4 --chain --no-audio # chain mode, no audio
  python main.py --seq C4 E4 G4 --chain --max-chains 4

Chain mode
----------
  After each generation you are asked which continuation to chain into:
    [S]afe  [C]reative  [U]nexpected  [Q]uit
  The chosen 3-note continuation becomes the next input.
  The loop stops when you type Q or the --max-chains limit is reached.

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
      C4.wav  D4.wav  E4.wav  F4.wav
      G4.wav  A4.wav  B4.wav  C5.wav
"""

import argparse
import sys

from engine import generate, TuneGenResult
from player import Player


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display(result: TuneGenResult, chain_step: int = 0) -> None:
    bar = "─" * 52
    step_label = f"  Step {chain_step}" if chain_step > 0 else "  TuneGen — Rule-Based Music Generator"
    print(f"\n{'═' * 52}")
    print(step_label)
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
# Chain history tracker
# ---------------------------------------------------------------------------

def display_chain_history(history: list[tuple[str, list[str]]]) -> None:
    """Print the full melody built so far as a single line."""
    if not history:
        return
    print("\n  ── Melody so far ──────────────────────────────────")
    line_parts = []
    for mode, notes in history:
        line_parts.append(f"[{mode[:1]}] {' → '.join(notes)}")
    print("  " + "  |  ".join(line_parts))
    print("  ────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Mode prompt
# ---------------------------------------------------------------------------

VALID_CHOICES = {"s": "safe", "c": "creative", "u": "unexpected", "q": "quit"}

def ask_chain_choice() -> str:
    """
    Prompt the user to pick a continuation mode.
    Returns one of: 'safe', 'creative', 'unexpected', 'quit'.
    """
    while True:
        raw = input("\n  Chain into → [S]afe  [C]reative  [U]nexpected  [Q]uit : ").strip().lower()
        if raw in VALID_CHOICES:
            return VALID_CHOICES[raw]
        print("  Please enter S, C, U, or Q.")


def get_continuation(result: TuneGenResult, mode: str) -> list[str]:
    return {
        "safe":       result.safe,
        "creative":   result.creative,
        "unexpected": result.unexpected,
    }[mode]


# ---------------------------------------------------------------------------
# Chain loop
# ---------------------------------------------------------------------------

def run_chain(
    seed: list[str],
    max_chains: int,
    audio: bool,
    player: "Player | None" = None,
) -> None:
    """
    Core chain loop — shared by audio and no-audio modes.

    Parameters
    ----------
    seed       : Starting 3-note sequence.
    max_chains : Hard stop after this many iterations.
    audio      : Whether to play each result.
    player     : Player instance (required when audio=True).
    """
    current_seq = seed
    history: list[tuple[str, list[str]]] = []   # (mode_chosen, notes)
    step = 0

    print(f"\n  Chain mode started — max {max_chains} steps. Type Q to stop early.\n")

    while step < max_chains:
        step += 1
        result = generate(current_seq)
        display(result, chain_step=step)

        if audio and player:
            player.play_result(result)

        display_chain_history(history)

        if step == max_chains:
            print(f"\n  Reached max chains ({max_chains}). Stopping.")
            break

        choice = ask_chain_choice()
        if choice == "quit":
            print("\n  Stopped by user.")
            break

        chosen_notes = get_continuation(result, choice)
        history.append((choice.upper(), chosen_notes))
        current_seq = chosen_notes

    # Final summary
    print(f"\n  ── Final melody ({len(history)} chain(s) completed) ─────────")
    all_notes = list(seed)
    for _, notes in history:
        all_notes.extend(notes)
    print(f"  {' → '.join(all_notes)}")
    print(f"  ────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Demo mode (no chain)
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


def run_demo(audio: bool, player: "Player | None") -> None:
    for label, seq in DEMO_SEQUENCES:
        print(f"\n>>> {label}")
        result = generate(seq)
        display(result)
        if audio and player:
            player.play_result(result)


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
        help="Seed sequence for chain mode, e.g. --seq C4 E4 G4",
    )
    parser.add_argument(
        "--chain",
        action="store_true",
        help="Enable interactive chain mode (requires --seq)",
    )
    parser.add_argument(
        "--max-chains",
        type=int,
        default=8,
        metavar="N",
        help="Hard stop after N chain steps (default: 8)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Print analysis only; skip audio playback",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate chain requires --seq
    if args.chain and not args.seq:
        print("[Error] --chain requires a seed sequence. Use --seq NOTE1 NOTE2 NOTE3.")
        sys.exit(1)

    use_audio = not args.no_audio

    try:
        player = Player(sounds_dir="sounds") if use_audio else None

        if args.chain:
            run_chain(
                seed       = args.seq,
                max_chains = args.max_chains,
                audio      = use_audio,
                player     = player,
            )
        elif args.seq:
            result = generate(args.seq)
            display(result)
            if use_audio and player:
                player.play_result(result)
        else:
            run_demo(audio=use_audio, player=player)

        if player:
            player.close()

    except FileNotFoundError as exc:
        print(f"\n[Audio Error] {exc}")
        print("Tip: run with --no-audio to skip playback.\n")
        sys.exit(1)
    except ImportError as exc:
        print(f"\n[Import Error] {exc}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        sys.exit(0)


if __name__ == "__main__":
    main()