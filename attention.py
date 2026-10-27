"""
attention.py - Causal Self-Attention (Built from Scratch)
==========================================================
This is the heart of the Transformer. Self-attention lets each token in a
sequence look at all previous tokens and decide which ones are most relevant.

How Self-Attention Works (Simple Explanation):
----------------------------------------------
Imagine reading a sentence word by word. When you reach the word "it", you
look back to figure out what "it" refers to. Self-attention does something
similar - for each token, it computes how much attention to pay to every
other token.

The Process:
1. QUERY (Q): "What am I looking for?" - each token creates a query.
2. KEY   (K): "What do I contain?"     - each token creates a key.
3. VALUE (V): "What information do I carry?" - each token creates a value.

4. Attention Score = Q · K^T  (dot product measures similarity)
5. Scale by √d_k to prevent scores from getting too large
6. Apply causal mask (prevent looking at future tokens)
7. Softmax to get attention weights (probabilities that sum to 1)
8. Output = attention_weights · V (weighted combination of values)

Why "Causal"?
-------------
In language modeling, when predicting the next word, the model should only
see previous words - not future ones. The causal mask enforces this by
setting future positions to -infinity before softmax, making their
attention weight zero.
"""

import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """
    Single-head causal (masked) self-attention.

    This is a simplified version that uses one attention head instead of
    multiple heads. It's easier to understand and sufficient for our
    small model.

    Parameters:
        embed_dim: Size of the token embedding vectors
        seq_length: Maximum sequence length (for the causal mask)
        dropout: Dropout rate for attention weights (regularization)
    """

    def __init__(self, embed_dim, seq_length, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # -- Linear projections --
        # These transform the input into queries, keys, and values.
        # Think of them as learned "lenses" that extract different aspects
        # of each token's meaning.
        self.query_proj = nn.Linear(embed_dim, embed_dim)  # W_Q
        self.key_proj = nn.Linear(embed_dim, embed_dim)    # W_K
        self.value_proj = nn.Linear(embed_dim, embed_dim)  # W_V

        # Output projection: combines the attention output back
        self.output_proj = nn.Linear(embed_dim, embed_dim)  # W_O

        # Dropout for regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # -- Causal mask --
        # A lower-triangular matrix of ones.
        # Position (i, j) is 1 if j <= i (can attend), 0 if j > i (blocked).
        #
        # Example for seq_length=4:
        #   [[1, 0, 0, 0],    Token 0 can only see token 0
        #    [1, 1, 0, 0],    Token 1 can see tokens 0-1
        #    [1, 1, 1, 0],    Token 2 can see tokens 0-2
        #    [1, 1, 1, 1]]    Token 3 can see tokens 0-3
        #
        # register_buffer saves this tensor with the model but doesn't treat
        # it as a learnable parameter.
        mask = torch.tril(torch.ones(seq_length, seq_length))
        self.register_buffer("mask", mask)

    def forward(self, x):
        """
        Forward pass of causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_dim)

        Returns:
            Output tensor of same shape (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # -- Step 1: Create Q, K, V --
        # Each has shape: (batch_size, seq_length, embed_dim)
        Q = self.query_proj(x)  # What am I looking for?
        K = self.key_proj(x)    # What do I contain?
        V = self.value_proj(x)  # What information do I carry?

        # -- Step 2: Compute attention scores --
        # Score = Q · K^T
        # Shape: (batch_size, seq_length, seq_length)
        # Each entry (i, j) measures how much token i should attend to token j.
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # -- Step 3: Scale --
        # Divide by √d_k to prevent dot products from growing too large.
        # Large values push softmax into regions with tiny gradients,
        # making training unstable. Scaling fixes this.
        scale = math.sqrt(self.embed_dim)
        scores = scores / scale

        # -- Step 4: Apply causal mask --
        # Set future positions to -infinity so softmax gives them zero weight.
        # We use the pre-computed lower-triangular mask.
        causal_mask = self.mask[:seq_len, :seq_len]  # Trim to actual length
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # -- Step 5: Softmax --
        # Convert scores to probabilities (each row sums to 1).
        # After masking, future tokens get probability ≈ 0.
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights (regularization)
        attention_weights = self.attn_dropout(attention_weights)

        # -- Step 6: Weighted sum of values --
        # Multiply attention weights by values to get the output.
        # Each token's output is a weighted combination of all visible
        # tokens' values, where the weights reflect relevance.
        # Shape: (batch_size, seq_length, embed_dim)
        context = torch.matmul(attention_weights, V)

        # -- Step 7: Output projection --
        # Final linear transformation and dropout
        output = self.output_proj(context)
        output = self.output_dropout(output)

        return output


# -- Quick test when run directly --
if __name__ == "__main__":
    print("Testing CausalSelfAttention...")

    batch_size = 2
    seq_length = 8
    embed_dim = 32

    # Create random input (simulating embedded tokens)
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Create attention layer
    attn = CausalSelfAttention(embed_dim=embed_dim, seq_length=seq_length)

    # Forward pass
    output = attn(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shapes match: {x.shape == output.shape}")

    # Verify causal masking: check that output at position 0
    # is the same regardless of what's at positions 1+
    x2 = x.clone()
    x2[:, 1:, :] = torch.randn_like(x2[:, 1:, :])  # Change future tokens

    attn.eval()  # Disable dropout for deterministic comparison
    with torch.no_grad():
        out1 = attn(x)
        out2 = attn(x2)

    # Position 0 should produce the same output since it can't see positions 1+
    diff = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
    print(f"Causal mask check (should be ~0): {diff:.10f}")
    print("OK Causal masking works!" if diff < 1e-6 else "FAIL Causal masking broken!")
