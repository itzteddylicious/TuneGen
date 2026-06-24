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
  Every chain session is automatically saved to sessions/<timestamp>.json.

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
    main.py        <- you are here
    sessions/      <- auto-created, one JSON per chain session
    sounds/
      C4.wav  D4.wav  E4.wav  F4.wav
      G4.wav  A4.wav  B4.wav  C5.wav
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from midiutil import MIDIFile

from engine import generate, TuneGenResult
from player import DURATION_PROFILES
from player import Player


# ---------------------------------------------------------------------------
# MIDI constants
# ---------------------------------------------------------------------------

EXPORTS_DIR = Path("exports")

# MIDI pitch for each scale note
NOTE_TO_MIDI: dict[str, int] = {
    "C4": 60, "D4": 62, "E4": 64, "F4": 65,
    "G4": 67, "A4": 69, "B4": 71, "C5": 72,
}

# Tempo used to convert seconds -> beats  (120 BPM = 0.5 s per beat)
MIDI_TEMPO       = 120
SECONDS_PER_BEAT = 60 / MIDI_TEMPO   # 0.5 s


def _seconds_to_beats(seconds: float) -> float:
    return seconds / SECONDS_PER_BEAT


# ---------------------------------------------------------------------------
# MIDI exporter
# ---------------------------------------------------------------------------

def export_midi(
    seed:      list[str],
    steps:     list[dict],
    timestamp: str,
) -> Path:
    """
    Write the chain session to a MIDI file in exports/<timestamp>.mid.

    Duration per note group mirrors the player duration profiles so the
    MIDI sounds like what you heard during the session:
      seed notes   -> INPUT  profile (0.5 s -> 1.0 beat)
      each chain   -> the mode chosen for that group

    Parameters
    ----------
    seed      : The original 3-note seed sequence.
    steps     : session_steps list from run_chain (same data as JSON log).
    timestamp : Shared with the session JSON so the two files match up.

    Returns
    -------
    Path to the saved .mid file.
    """
    EXPORTS_DIR.mkdir(exist_ok=True)
    filepath = EXPORTS_DIR / f"{timestamp}.mid"

    midi = MIDIFile(1)           # single track
    midi.addTempo(0, 0, MIDI_TEMPO)

    track    = 0
    channel  = 0
    volume   = 100
    position = 0.0   # current beat position

    def write_notes(notes: list[str], mode: str) -> None:
        nonlocal position
        hold = DURATION_PROFILES.get(mode, DURATION_PROFILES["INPUT"])
        duration_beats = _seconds_to_beats(hold)
        for note in notes:
            pitch = NOTE_TO_MIDI.get(note, 60)
            midi.addNote(track, channel, pitch, position, duration_beats, volume)
            position += duration_beats

    # Write seed with INPUT timing
    write_notes(seed, "INPUT")

    # Write each chosen continuation with its mode timing
    for step in steps:
        chosen_mode  = step.get("chosen_mode")
        chosen_notes = step.get("chosen_notes")
        if chosen_notes and chosen_mode not in (None, "quit"):
            write_notes(chosen_notes, chosen_mode.upper())

    with open(filepath, "wb") as f:
        midi.writeFile(f)

    return filepath


# ---------------------------------------------------------------------------
# Session logger
# ---------------------------------------------------------------------------

SESSIONS_DIR = Path("sessions")


def save_session(
    seed:         list[str],
    steps:        list[dict],
    final_melody: list[str],
    timestamp:    str,
) -> Path:
    """
    Save a chain session to sessions/<timestamp>.json and return the path.

    JSON structure
    --------------
    {
      "timestamp": "2024-01-15T14:30:22",
      "seed": ["C4", "E4", "G4"],
      "steps": [
        {
          "step": 1,
          "input": ["C4", "E4", "G4"],
          "chord": "C_MAJOR",
          "contour": "ascending",
          "continuations": {
            "safe":       ["A4", "G4", "C4"],
            "creative":   ["B4", "A4", "G4"],
            "unexpected": ["D4", "F4", "A4"]
          },
          "chosen_mode":  "creative",
          "chosen_notes": ["B4", "A4", "G4"]
        }
      ],
      "final_melody": ["C4", "E4", "G4", "B4", "A4", "G4"]
    }
    """
    SESSIONS_DIR.mkdir(exist_ok=True)

    filepath = SESSIONS_DIR / f"{timestamp}.json"

    payload = {
        "timestamp":    timestamp,
        "seed":         seed,
        "steps":        steps,
        "final_melody": final_melody,
    }

    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)

    return filepath


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display(result: TuneGenResult, chain_step: int = 0) -> None:
    bar = "─" * 52
    step_label = f"  Step {chain_step}" if chain_step > 0 else "  TuneGen — Rule-Based Music Generator"
    print(f"\n{'=' * 52}")
    print(step_label)
    print(f"{'=' * 52}")
    print(f"  Input Sequence : {' -> '.join(result.input_sequence)}")
    print(f"  Chord Detected : {result.chord_detected}")
    print(f"  Chord Tones    : {', '.join(result.chord_tones)}")
    print(f"  Contour        : {result.contour}")
    print(f"  Avg Interval   : {result.avg_interval} scale steps")
    print(f"{bar}")
    print(f"  SAFE        : {' -> '.join(result.safe)}")
    print(f"  CREATIVE    : {' -> '.join(result.creative)}")
    print(f"  UNEXPECTED  : {' -> '.join(result.unexpected)}")
    print(f"{'=' * 52}")


# ---------------------------------------------------------------------------
# Chain history display
# ---------------------------------------------------------------------------

def display_chain_history(history: list[tuple[str, list[str]]]) -> None:
    if not history:
        return
    print("\n  -- Melody so far --------------------------------------------------")
    line_parts = []
    for mode, notes in history:
        line_parts.append(f"[{mode[:1]}] {' -> '.join(notes)}")
    print("  " + "  |  ".join(line_parts))
    print("  -------------------------------------------------------------------")


# ---------------------------------------------------------------------------
# Mode prompt
# ---------------------------------------------------------------------------

VALID_CHOICES = {"s": "safe", "c": "creative", "u": "unexpected", "q": "quit"}


def ask_chain_choice() -> str:
    while True:
        raw = input("\n  Chain into -> [S]afe  [C]reative  [U]nexpected  [Q]uit : ").strip().lower()
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
    seed:       list[str],
    max_chains: int,
    audio:      bool,
    player=None,
) -> None:
    current_seq = seed
    history: list[tuple[str, list[str]]] = []   # (mode_chosen, notes)
    session_steps: list[dict] = []               # full data for JSON log
    step = 0

    print(f"\n  Chain mode started -- max {max_chains} steps. Type Q to stop early.\n")

    while step < max_chains:
        step += 1
        result = generate(current_seq)
        display(result, chain_step=step)

        if audio and player:
            player.play_result(result)

        display_chain_history(history)

        if step == max_chains:
            print(f"\n  Reached max chains ({max_chains}). Stopping.")
            # Record the final step with no choice made
            session_steps.append({
                "step":    step,
                "input":   result.input_sequence,
                "chord":   result.chord_detected,
                "contour": result.contour,
                "continuations": {
                    "safe":       result.safe,
                    "creative":   result.creative,
                    "unexpected": result.unexpected,
                },
                "chosen_mode":  None,
                "chosen_notes": None,
            })
            break

        choice = ask_chain_choice()

        if choice == "quit":
            print("\n  Stopped by user.")
            session_steps.append({
                "step":    step,
                "input":   result.input_sequence,
                "chord":   result.chord_detected,
                "contour": result.contour,
                "continuations": {
                    "safe":       result.safe,
                    "creative":   result.creative,
                    "unexpected": result.unexpected,
                },
                "chosen_mode":  "quit",
                "chosen_notes": None,
            })
            break

        chosen_notes = get_continuation(result, choice)

        session_steps.append({
            "step":    step,
            "input":   result.input_sequence,
            "chord":   result.chord_detected,
            "contour": result.contour,
            "continuations": {
                "safe":       result.safe,
                "creative":   result.creative,
                "unexpected": result.unexpected,
            },
            "chosen_mode":  choice,
            "chosen_notes": chosen_notes,
        })

        history.append((choice.upper(), chosen_notes))
        current_seq = chosen_notes

    # Build final melody
    all_notes = list(seed)
    for _, notes in history:
        all_notes.extend(notes)

    # Final summary
    print(f"\n  -- Final melody ({len(history)} chain(s) completed) ----------------")
    print(f"  {' -> '.join(all_notes)}")
    print(f"  -------------------------------------------------------------------\n")

    # Shared timestamp so session JSON and MIDI always match up
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Save session JSON
    saved_path = save_session(
        seed         = seed,
        steps        = session_steps,
        final_melody = all_notes,
        timestamp    = timestamp,
    )
    print(f"  Session saved  -> {saved_path}")

    # Export MIDI
    midi_path = export_midi(
        seed      = seed,
        steps     = session_steps,
        timestamp = timestamp,
    )
    print(f"  MIDI exported  -> {midi_path}\n")


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


def run_demo(audio: bool, player=None) -> None:
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
        description="TuneGen -- Rule-Based Music Generator"
    )
    parser.add_argument(
        "--seq",
        nargs=3,
        metavar=("NOTE1", "NOTE2", "NOTE3"),
        help="Seed sequence, e.g. --seq C4 E4 G4",
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