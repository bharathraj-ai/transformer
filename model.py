"""
model.py - Transformer Language Model
========================================
A small Transformer that predicts the next token in a sequence.

Architecture Overview:
----------------------
Input Token IDs
      │
      ▼
┌---------------------┐
│  Token Embedding     │  - Converts token IDs to vectors
│  + Positional Embed  │  - Adds position information
└----------┬----------┘
           │
    ┌------▼------┐
    │  Transformer │  ← One or more blocks, each containing:
    │    Block     │     • Layer Norm
    │              │     • Causal Self-Attention
    │              │     • Residual Connection
    │              │     • Layer Norm
    │              │     • Feed-Forward Network
    │              │     • Residual Connection
    └------┬------┘
           │
    ┌------▼------┐
    │  Layer Norm  │
    └------┬------┘
           │
    ┌------▼------┐
    │  Output Head │  - Projects to vocabulary size
    └------┬------┘
           │
           ▼
    Next-Token Probabilities

Why Two Types of Embeddings?
-----------------------------
• Token embeddings: Give each word a unique vector representation.
  "cat" → [0.2, -0.1, 0.8, ...]

• Positional embeddings: Tell the model WHERE each token is in the sequence.
  Without this, "the cat sat" and "sat cat the" would look identical because
  attention is permutation-invariant. Position embeddings break this symmetry.
"""

import torch
import torch.nn as nn
import json
import os

from attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """
    One Transformer block = Self-Attention + Feed-Forward Network.

    The block uses:
    • Pre-norm: Layer normalization BEFORE attention and FFN (more stable training)
    • Residual connections: Add input back to output (helps gradients flow)
    • Feed-forward network: Two linear layers with GELU activation

    Residual Connection Explained:
    ------------------------------
    Instead of output = f(input), we do output = input + f(input).
    This means gradients can flow directly through the + path,
    preventing the vanishing gradient problem in deep networks.
    """

    def __init__(self, embed_dim, seq_length, ff_dim=None, dropout=0.1):
        super().__init__()

        # Default feed-forward dimension is 4x the embedding dimension
        # (a common choice in Transformer literature)
        if ff_dim is None:
            ff_dim = embed_dim * 4

        # Layer normalization (normalizes activations for stable training)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Self-attention
        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            seq_length=seq_length,
            dropout=dropout,
        )

        # Feed-forward network (processes each position independently)
        # Two layers with a GELU activation in between.
        # GELU is a smooth version of ReLU used in modern Transformers.
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass through the Transformer block.

        x shape: (batch_size, seq_length, embed_dim)
        """
        # -- Self-Attention with residual connection --
        # 1. Normalize
        # 2. Apply self-attention
        # 3. Add back the original input (residual)
        x = x + self.attention(self.ln1(x))

        # -- Feed-Forward with residual connection --
        # 1. Normalize
        # 2. Apply feed-forward network
        # 3. Add back the input (residual)
        x = x + self.feed_forward(self.ln2(x))

        return x


class TransformerLM(nn.Module):
    """
    Transformer Language Model for next-token prediction.

    This model takes a sequence of token IDs and predicts the next token
    at each position.
    """

    def __init__(self, vocab_size, embed_dim=64, seq_length=16,
                 num_layers=2, dropout=0.1):
        super().__init__()

        self.seq_length = seq_length
        self.embed_dim = embed_dim

        # -- Token Embedding --
        # Lookup table: each token ID → a learned vector of size embed_dim.
        # vocab_size rows, each of embed_dim dimensions.
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # -- Positional Embedding --
        # Lookup table: each position (0, 1, 2, ...) → a learned vector.
        # This tells the model WHERE each token sits in the sequence.
        self.position_embedding = nn.Embedding(seq_length, embed_dim)

        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(dropout)

        # -- Transformer Blocks --
        # Stack of identical blocks (default: 2 blocks)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, seq_length, dropout=dropout)
            for _ in range(num_layers)
        ])

        # -- Final layer norm --
        self.final_ln = nn.LayerNorm(embed_dim)

        # -- Output head --
        # Projects from embed_dim back to vocab_size.
        # The output at each position is a score for every word in the vocabulary.
        # Higher score = model thinks that word is more likely to come next.
        self.output_head = nn.Linear(embed_dim, vocab_size)

        # Initialize weights (helps training converge faster)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform (good default for Transformers)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Forward pass: token IDs → next-token logits.

        Args:
            x: Token IDs of shape (batch_size, seq_length)

        Returns:
            logits: Raw scores of shape (batch_size, seq_length, vocab_size)
                    logits[b][t][v] = score for vocab word v being the next
                    token after position t in batch example b.
        """
        batch_size, seq_len = x.shape

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # Shape: (1, seq_len) - broadcasts across batch

        # -- Combine token + position embeddings --
        # Token embedding: what is this word?
        # Position embedding: where is this word?
        # Adding them merges both types of information.
        tok_emb = self.token_embedding(x)       # (batch, seq_len, embed_dim)
        pos_emb = self.position_embedding(positions)  # (1, seq_len, embed_dim)
        x = self.embed_dropout(tok_emb + pos_emb)

        # -- Pass through Transformer blocks --
        for block in self.blocks:
            x = block(x)

        # -- Final norm + output projection --
        x = self.final_ln(x)
        logits = self.output_head(x)  # (batch, seq_len, vocab_size)

        return logits

    def count_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_default_config():
    """Return default model configuration."""
    return {
        "embed_dim": 128,
        "seq_length": 8,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "num_epochs": 200,
        "seed": 42,
    }


def save_model(model, config, save_dir="saved_model"):
    """Save model weights and configuration."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


def load_model(save_dir="saved_model", device="cpu"):
    """Load model weights and configuration."""
    # Load config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create model with saved config
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        seq_length=config["seq_length"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    # Load weights
    model_path = os.path.join(save_dir, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    print(f"Model loaded from {save_dir} ({model.count_parameters():,} parameters)")
    return model, config


# -- Quick test when run directly --
if __name__ == "__main__":
    print("Testing TransformerLM...")

    vocab_size = 100
    config = get_default_config()

    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=config["embed_dim"],
        seq_length=config["seq_length"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    batch = torch.randint(0, vocab_size, (4, config["seq_length"]))
    logits = model(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected:     (4, {config['seq_length']}, {vocab_size})")
    assert logits.shape == (4, config["seq_length"], vocab_size)
    print("OK Forward pass works!")
