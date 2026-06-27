"""
train.py — TuneGen LSTM Training Loop
=======================================
Trains the TuneGenLSTM model on preprocessed GiantMIDI data.

What this script does
---------------------
- Loads train and validation splits from data/
- Trains the LSTM with cross-entropy loss
- Validates after every epoch
- Saves the best model checkpoint based on validation loss
- Logs training progress to checkpoints/training_log.txt

Run
---
    python train.py

Output
------
    checkpoints/best_model.pt       — best model weights
    checkpoints/training_log.txt    — epoch-by-epoch loss log

Dependencies
------------
    pip install torch numpy
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import TuneGenLSTM


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR         = Path("data")
CHECKPOINT_DIR   = Path("checkpoints")

BATCH_SIZE       = 512
EPOCHS           = 5
LEARNING_RATE    = 0.001
GRAD_CLIP        = 1.0        # gradient clipping to prevent exploding gradients
SAMPLE_RATIO     = 0.20       # fraction of training data to use


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NoteSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping the preprocessed numpy arrays.

    Parameters
    ----------
    inputs  : np.ndarray, shape (N, seq_len)
    outputs : np.ndarray, shape (N,)
    """

    def __init__(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        self.inputs  = torch.tensor(inputs,  dtype=torch.long)
        self.outputs = torch.tensor(outputs, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(filename: str, sample_ratio: float = 1.0) -> NoteSequenceDataset:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"'{path}' not found. Run the preprocessing scripts first."
        )
    data    = np.load(path)
    inputs  = data["inputs"]
    outputs = data["outputs"]

    if sample_ratio < 1.0:
        n          = int(len(inputs) * sample_ratio)
        rng        = np.random.default_rng(42)
        indices    = rng.choice(len(inputs), size=n, replace=False)
        inputs     = inputs[indices]
        outputs    = outputs[indices]
        print(f"  Sampled {n:,} / {len(data['inputs']):,} pairs ({sample_ratio*100:.0f}%)")

    return NoteSequenceDataset(inputs, outputs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model:      TuneGenLSTM,
    loader:     DataLoader,
    criterion:  nn.CrossEntropyLoss,
    optimizer:  torch.optim.Adam,
    device:     torch.device,
    epoch:      int,
) -> float:
    """Run one full training epoch. Returns mean loss."""
    model.train()
    total_loss   = 0.0
    start_time   = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader, 1):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss      = criterion(logits, targets)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(
    model:     TuneGenLSTM,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
    device:    torch.device,
) -> float:
    """Run validation. Returns mean loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss      = criterion(logits, targets)
            total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n  Loading data...")
    train_dataset = load_split("train.npz", sample_ratio=SAMPLE_RATIO)
    val_dataset   = load_split("val.npz")

    train_loader  = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,     # MPS works best with num_workers=0
        pin_memory  = False, # not needed for MPS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
    )

    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val batches   : {len(val_loader):,}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model     = TuneGenLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters    : {total_params:,}")

    # ------------------------------------------------------------------
    # Resume from checkpoint if available
    # ------------------------------------------------------------------
    start_epoch   = 1
    best_val_loss = float("inf")
    log_lines     = ["epoch,train_loss,val_loss,time_s\n"]

    checkpoint_path = CHECKPOINT_DIR / "best_model.pt"
    if checkpoint_path.exists():
        print(f"\n  Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["val_loss"]
        start_epoch   = checkpoint["epoch"] + 1
        print(f"  Resumed from epoch {checkpoint['epoch']} "
              f"(val loss: {best_val_loss:.4f})")
    else:
        print("\n  No checkpoint found — starting from scratch.")

    remaining_epochs = EPOCHS - (start_epoch - 1)
    if remaining_epochs <= 0:
        print(f"\n  Already trained {checkpoint['epoch']} epoch(s). "
              f"Increase EPOCHS to train further.")
        return

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n  Starting training for {remaining_epochs} more epoch(s) "
          f"(epoch {start_epoch} → {start_epoch + remaining_epochs - 1})...\n")

    for epoch in range(start_epoch, start_epoch + remaining_epochs):
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch} complete")
        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Time       : {epoch_time:.1f}s\n")

        log_lines.append(f"{epoch},{train_loss:.4f},{val_loss:.4f},{epoch_time:.1f}\n")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":      epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":   val_loss,
                },
                CHECKPOINT_DIR / "best_model.pt",
            )
            print(f"  Checkpoint saved (val loss: {val_loss:.4f})\n")

    # ------------------------------------------------------------------
    # Save log
    # ------------------------------------------------------------------
    log_path = CHECKPOINT_DIR / "training_log.txt"
    with open(log_path, "w") as f:
        f.writelines(log_lines)

    print(f"  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Log saved to {log_path}\n")


if __name__ == "__main__":
    main()