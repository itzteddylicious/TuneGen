"""
model.py — TuneGen Transformer Model
======================================
Defines the Transformer decoder architecture for next-note prediction.

Architecture
------------
  Input  : sequence of MIDI pitch values (integers 0-127)
  Embed  : each pitch mapped to a dense 64-dim vector
  PosEnc : sinusoidal positional encoding added to embeddings
  Trans  : 4 decoder layers, 4 attention heads, 256 feedforward dim
  Output : linear projection to 128 classes (full MIDI pitch range)

The model uses causal masking so each position can only attend
to previous positions — ensuring the model predicts the next note
based only on what came before it.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as described in 'Attention Is All You Need'.

    Adds position information to the embeddings since Transformers
    process all positions simultaneously and have no inherent sense of order.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension. Must match the model dimension.
    max_seq_len : int
        Maximum sequence length to pre-compute encodings for.
    dropout : float
        Dropout applied after adding positional encoding.
    """

    def __init__(
        self,
        embed_dim:   int,
        max_seq_len: int   = 512,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute sinusoidal encodings
        pe       = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it moves with the model to GPU but isn't a parameter
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, seq_len, embed_dim)

        Returns
        -------
        torch.Tensor
            Same shape as input with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TuneGenTransformer(nn.Module):
    """
    Transformer decoder for next-note prediction.

    Parameters
    ----------
    vocab_size   : int   — number of possible pitch values (128 for full MIDI range)
    embed_dim    : int   — embedding dimension
    num_heads    : int   — number of attention heads
    num_layers   : int   — number of Transformer decoder layers
    ff_dim       : int   — feedforward layer dimension
    max_seq_len  : int   — maximum input sequence length
    dropout      : float — dropout probability
    """

    def __init__(
        self,
        vocab_size:  int   = 128,
        embed_dim:   int   = 128,
        num_heads:   int   = 8,
        num_layers:  int   = 6,
        ff_dim:      int   = 512,
        max_seq_len: int   = 512,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        # Pitch embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model         = embed_dim,
            nhead           = num_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer = decoder_layer,
            num_layers    = num_layers,
        )

        # Output projection
        self.fc = nn.Linear(embed_dim, vocab_size)

        # Initialise weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate a causal mask so each position can only attend to
        previous positions and itself.

        Returns
        -------
        torch.Tensor of shape (seq_len, seq_len) with -inf above the diagonal.
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of pitch indices, shape (batch_size, seq_len).

        Returns
        -------
        logits : torch.Tensor
            Raw class scores, shape (batch_size, vocab_size).
        """
        seq_len = x.size(1)
        device  = x.device

        # Embed and add positional encoding
        # (batch, seq_len) → (batch, seq_len, embed_dim)
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoding(x)

        # Causal mask — prevents attending to future positions
        mask = self._causal_mask(seq_len, device)

        # Transformer layers
        x = self.transformer(x, mask=mask)

        # Take only the last position output for next-note prediction
        x = x[:, -1, :]           # (batch, embed_dim)

        # Project to vocab
        logits = self.fc(x)        # (batch, vocab_size)

        return logits


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device : {device}")

    model = TuneGenTransformer().to(device)
    print(f"  Model  : {model}\n")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {total_params:,}")

    # Forward pass with dummy input
    dummy_input = torch.randint(0, 128, (32, 32)).to(device)
    logits      = model(dummy_input)
    print(f"  Input shape      : {dummy_input.shape}")
    print(f"  Output shape     : {logits.shape}")
    print(f"\n  Sanity check passed.\n")