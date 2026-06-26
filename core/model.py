"""
model.py — TuneGen LSTM Model
==============================
Defines the LSTM architecture for next-note prediction.

Architecture
------------
  Input  : sequence of MIDI pitch values (integers 0-127)
  Embed  : each pitch mapped to a dense 64-dim vector
  LSTM   : 2 layers, 512 hidden units, dropout 0.3
  Output : linear projection to 128 classes (full MIDI pitch range)

The model treats next-note prediction as a classification problem —
for a given sequence of notes, predict which of the 128 possible
MIDI pitches comes next.
"""

import torch
import torch.nn as nn


class TuneGenLSTM(nn.Module):
    """
    LSTM-based next-note predictor.

    Parameters
    ----------
    vocab_size : int
        Number of possible pitch values. 128 covers the full MIDI range.
    embed_dim : int
        Dimension of the pitch embedding vectors.
    hidden_size : int
        Number of hidden units in each LSTM layer.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied between LSTM layers.
    """

    def __init__(
        self,
        vocab_size:  int   = 128,
        embed_dim:   int   = 64,
        hidden_size: int   = 512,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
    ) -> None:
        super().__init__()

        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Embedding: integer pitch → dense vector
        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embed_dim,
        )

        # LSTM: learns sequential patterns across the note window
        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,    # input shape: (batch, seq_len, features)
        )

        # Dropout before the output layer
        self.dropout = nn.Dropout(dropout)

        # Output: project hidden state to pitch class probabilities
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of pitch indices, shape (batch_size, seq_len).
        hidden : tuple | None
            Optional initial hidden and cell states.
            If None, defaults to zeros.

        Returns
        -------
        logits : torch.Tensor
            Raw class scores, shape (batch_size, vocab_size).
        hidden : tuple[torch.Tensor, torch.Tensor]
            Updated hidden and cell states for stateful inference.
        """
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # (batch, seq_len, embed_dim) → (batch, seq_len, hidden_size)
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Take only the last time step output
        last_out = lstm_out[:, -1, :]          # (batch, hidden_size)

        # Apply dropout and project to vocab
        logits = self.fc(self.dropout(last_out))  # (batch, vocab_size)

        return logits, hidden

    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise hidden and cell states to zeros.

        Parameters
        ----------
        batch_size : int
        device     : torch.device

        Returns
        -------
        Tuple of (h_0, c_0), each shape (num_layers, batch_size, hidden_size).
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h_0, c_0


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device : {device}")

    model = TuneGenLSTM().to(device)
    print(f"  Model  : {model}\n")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params:,}")

    # Forward pass with dummy input
    dummy_input = torch.randint(0, 128, (32, 32)).to(device)  # batch=32, seq=32
    logits, hidden = model(dummy_input)
    print(f"  Input shape      : {dummy_input.shape}")
    print(f"  Output shape     : {logits.shape}")
    print(f"  Hidden shape     : {hidden[0].shape}")
    print(f"\n  Sanity check passed.\n")