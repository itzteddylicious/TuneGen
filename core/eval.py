"""
eval.py — TuneGen Model Evaluation
=====================================
Evaluates the trained LSTM model on the held-out test set.

Metrics
-------
- Loss        : Cross entropy loss on the full test set
- Top-1 Acc   : How often the model's top prediction is exactly correct
- Top-5 Acc   : How often the correct note is in the top 5 predictions

Why Top-5 matters
-----------------
In music, multiple notes can be valid continuations of a phrase.
Top-5 accuracy is a more musically meaningful measure than Top-1
since it captures whether the model understands the harmonic context
even when it doesn't predict the exact note.

Run
---
    python eval.py

Output
------
    Prints metrics to stdout
    Saves report to checkpoints/eval_report.txt

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

DATA_DIR        = Path("data")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
REPORT_PATH     = Path("checkpoints/eval_report.txt")

BATCH_SIZE      = 512


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NoteSequenceDataset(Dataset):
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        self.inputs  = torch.tensor(inputs,  dtype=torch.long)
        self.outputs = torch.tensor(outputs, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.outputs[idx]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model:     TuneGenLSTM,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
    device:    torch.device,
) -> dict:
    """
    Run full evaluation on the test set.

    Returns
    -------
    dict with keys: loss, top1_acc, top5_acc, total_samples
    """
    model.eval()

    total_loss    = 0.0
    top1_correct  = 0
    top5_correct  = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)
            loss      = criterion(logits, targets)
            total_loss += loss.item()

            # Top-1 accuracy
            top1_preds   = torch.argmax(logits, dim=-1)
            top1_correct += (top1_preds == targets).sum().item()

            # Top-5 accuracy
            top5_preds   = torch.topk(logits, k=5, dim=-1).indices
            top5_correct += sum(
                targets[i].item() in top5_preds[i].tolist()
                for i in range(len(targets))
            )

            total_samples += len(targets)

    return {
        "loss":          total_loss / len(loader),
        "top1_acc":      top1_correct  / total_samples * 100,
        "top5_acc":      top5_correct  / total_samples * 100,
        "total_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        print(f"[Error] Checkpoint not found at '{CHECKPOINT_PATH}'.")
        return

    model = TuneGenLSTM().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"  Model loaded — trained for {checkpoint['epoch']} epoch(s), "
          f"val loss: {checkpoint['val_loss']:.4f}\n")

    # ------------------------------------------------------------------
    # Load test set
    # ------------------------------------------------------------------
    test_path = DATA_DIR / "test.npz"
    if not test_path.exists():
        print(f"[Error] '{test_path}' not found. Run preprocessing first.")
        return

    print(f"  Loading test set...")
    data         = np.load(test_path)
    test_dataset = NoteSequenceDataset(data["inputs"], data["outputs"])
    test_loader  = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
    )
    print(f"  Test samples : {len(test_dataset):,}")
    print(f"  Test batches : {len(test_loader):,}\n")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    print("  Running evaluation...")
    start_time = time.time()
    metrics    = evaluate(model, test_loader, criterion, device)
    elapsed    = time.time() - start_time

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report = f"""
TuneGen — Model Evaluation Report
==================================
  Checkpoint               : {CHECKPOINT_PATH}
  Trained epochs           : {checkpoint['epoch']}
  Val loss (training)      : {checkpoint['val_loss']:.4f}

  Test Results
  ------------
  Test samples             : {metrics['total_samples']:,}
  Test loss                : {metrics['loss']:.4f}
  Top-1 accuracy           : {metrics['top1_acc']:.2f}%
  Top-5 accuracy           : {metrics['top5_acc']:.2f}%

  Time taken               : {elapsed:.1f}s
==================================
"""

    print(report)

    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"  Report saved to {REPORT_PATH}\n")


if __name__ == "__main__":
    main()