"""
predict.py — TuneGen Inference
================================
Loads the trained Transformer model and predicts the next notes
given an input sequence of note names.

Usage
-----
    python predict.py --seq C4 E4 G4
    python predict.py --seq C4 E4 G4 --steps 5

Arguments
---------
    --seq     : Input note names (space separated), e.g. C4 E4 G4
    --steps   : Number of notes to predict (default: 3)

Output
------
    Prints the predicted next notes as note names.

Dependencies
------------
    pip install torch numpy
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from model import TuneGenTransformer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
SEQUENCE_LENGTH = 32      # must match the window size used in preprocessing


# ---------------------------------------------------------------------------
# Note name <-> MIDI number mapping
# ---------------------------------------------------------------------------

# Standard note names across octaves 0-9
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note_to_midi(note: str) -> int:
    """
    Convert a note name to a MIDI number.
    e.g. C4 -> 60, A4 -> 69, D#3 -> 51
    """
    note = note.strip()

    # Split note name and octave
    if len(note) >= 3 and note[1] in ("#", "b"):
        name   = note[:2]
        octave = int(note[2:])
    else:
        name   = note[0]
        octave = int(note[1:])

    # Handle flats by converting to sharps
    flat_to_sharp = {
        "Db": "C#", "Eb": "D#", "Fb": "E",
        "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"
    }
    name = flat_to_sharp.get(name, name)

    if name not in NOTE_NAMES:
        raise ValueError(f"Unknown note name: '{note}'")

    return (octave + 1) * 12 + NOTE_NAMES.index(name)


def midi_to_note(midi: int) -> str:
    """
    Convert a MIDI number to a note name.
    e.g. 60 -> C4, 69 -> A4
    """
    octave = (midi // 12) - 1
    name   = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(device: torch.device) -> TuneGenTransformer:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{CHECKPOINT_PATH}'.\n"
            f"Make sure best_model.pt is inside core/checkpoints/."
        )

    model = TuneGenTransformer().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  Model loaded from {CHECKPOINT_PATH}")
    print(f"  Trained for {checkpoint['epoch']} epoch(s), "
          f"val loss: {checkpoint['val_loss']:.4f}\n")

    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_next_notes(
    model:       TuneGenTransformer,
    seed_seq:    list[int],
    steps:       int,
    device:      torch.device,
    temperature: float = 1.0,
) -> list[int]:
    """
    Predict the next `steps` notes autoregressively.

    Parameters
    ----------
    model    : Trained TuneGenTransformer
    seed_seq : List of MIDI pitch integers (at least SEQUENCE_LENGTH long)
    steps    : Number of notes to predict
    device   : torch.device

    Returns
    -------
    List of predicted MIDI pitch integers.
    """
    model.eval()
    predictions = []

    # Use the last SEQUENCE_LENGTH notes as the initial context
    context = list(seed_seq[-SEQUENCE_LENGTH:])

    with torch.no_grad():
        for _ in range(steps):
            x      = torch.tensor([context], dtype=torch.long).to(device)
            logits    = model(x)

            # Sample from probability distribution with temperature
            # Temperature > 1.0 = more random, < 1.0 = more conservative
            logits_scaled = logits / temperature
            probs         = torch.softmax(logits_scaled, dim=-1)
            predicted     = torch.multinomial(probs, num_samples=1).item()
            predictions.append(predicted)

            # Slide the window forward
            context = context[1:] + [predicted]

    return predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TuneGen — Next Note Prediction"
    )
    parser.add_argument(
        "--seq",
        nargs="+",
        metavar="NOTE",
        required=True,
        help="Input note sequence e.g. --seq C4 E4 G4",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of notes to predict (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0). Higher = more creative, lower = more conservative.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n  Device : {device}")

    # Load model
    model = load_model(device)

    # Convert input note names to MIDI numbers
    try:
        input_midi = [note_to_midi(n) for n in args.seq]
    except ValueError as e:
        print(f"\n  [Error] {e}")
        return

    print(f"  Input  : {' -> '.join(args.seq)}")
    print(f"  MIDI   : {input_midi}\n")

    # If input is shorter than SEQUENCE_LENGTH repeat the input to fill context
    # This is more musical than padding with a fixed note
    if len(input_midi) < SEQUENCE_LENGTH:
        while len(input_midi) < SEQUENCE_LENGTH:
            input_midi = input_midi + input_midi
        input_midi = input_midi[-SEQUENCE_LENGTH:]

    # Predict
    predicted_midi = predict_next_notes(
        model       = model,
        seed_seq    = input_midi,
        steps       = args.steps,
        device      = device,
        temperature = args.temperature,
    )

    # Convert predictions back to note names
    predicted_notes = [midi_to_note(m) for m in predicted_midi]

    print(f"  Predicted next {args.steps} note(s):")
    print(f"  {' -> '.join(predicted_notes)}")
    print(f"  MIDI : {predicted_midi}\n")


if __name__ == "__main__":
    main()