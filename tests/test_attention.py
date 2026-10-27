"""
test_attention.py - Tests for Causal Self-Attention
=====================================================
Tests the attention mechanism for correct shapes, causal masking
(future tokens must not influence earlier positions), and output
consistency.
"""

import os
import sys
import torch
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention import CausalSelfAttention


class TestCausalSelfAttention:
    """Test suite for CausalSelfAttention."""

    def setup_method(self):
        """Set up common test fixtures."""
        self.embed_dim = 32
        self.seq_length = 8
        self.batch_size = 2

        self.attn = CausalSelfAttention(
            embed_dim=self.embed_dim,
            seq_length=self.seq_length,
            dropout=0.0,  # No dropout for deterministic tests
        )
        self.attn.eval()  # Evaluation mode

    def test_output_shape(self):
        """Output shape should match input shape."""
        x = torch.randn(self.batch_size, self.seq_length, self.embed_dim)
        output = self.attn(x)
        assert output.shape == x.shape

    def test_output_shape_shorter_sequence(self):
        """Should work with sequences shorter than max seq_length."""
        short_len = 4
        x = torch.randn(self.batch_size, short_len, self.embed_dim)
        output = self.attn(x)
        assert output.shape == (self.batch_size, short_len, self.embed_dim)

    def test_causal_masking_position_zero(self):
        """
        Position 0 should only depend on itself.

        If we change tokens at positions 1+, the output at position 0
        should remain exactly the same.
        """
        x1 = torch.randn(1, self.seq_length, self.embed_dim)
        x2 = x1.clone()
        x2[:, 1:, :] = torch.randn_like(x2[:, 1:, :])  # Change future

        with torch.no_grad():
            out1 = self.attn(x1)
            out2 = self.attn(x2)

        # Position 0 should be identical
        diff = (out1[:, 0, :] - out2[:, 0, :]).abs().max().item()
        assert diff < 1e-5, f"Position 0 changed when future tokens changed (diff={diff})"

    def test_causal_masking_all_positions(self):
        """
        For every position i, changing any token at position j > i
        should NOT change the output at position i.
        """
        x1 = torch.randn(1, self.seq_length, self.embed_dim)

        with torch.no_grad():
            out1 = self.attn(x1)

        # For each position, change all future tokens and verify
        for pos in range(self.seq_length - 1):
            x2 = x1.clone()
            x2[:, pos + 1:, :] = torch.randn_like(x2[:, pos + 1:, :])

            with torch.no_grad():
                out2 = self.attn(x2)

            diff = (out1[:, pos, :] - out2[:, pos, :]).abs().max().item()
            assert diff < 1e-5, (
                f"Position {pos} changed when future tokens changed (diff={diff})"
            )

    def test_causal_mask_shape(self):
        """The causal mask should be a lower-triangular matrix."""
        mask = self.attn.mask
        assert mask.shape == (self.seq_length, self.seq_length)

        # Check it's lower-triangular
        expected = torch.tril(torch.ones(self.seq_length, self.seq_length))
        assert torch.equal(mask, expected)

    def test_different_inputs_different_outputs(self):
        """Different inputs should generally produce different outputs."""
        x1 = torch.randn(1, self.seq_length, self.embed_dim)
        x2 = torch.randn(1, self.seq_length, self.embed_dim)

        with torch.no_grad():
            out1 = self.attn(x1)
            out2 = self.attn(x2)

        # Outputs should differ (with very high probability for random inputs)
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_batch_independence(self):
        """Each item in a batch should be processed independently."""
        x = torch.randn(2, self.seq_length, self.embed_dim)

        with torch.no_grad():
            # Process as batch
            batch_out = self.attn(x)

            # Process individually
            out0 = self.attn(x[0:1])
            out1 = self.attn(x[1:2])

        # Should match
        assert torch.allclose(batch_out[0], out0[0], atol=1e-5)
        assert torch.allclose(batch_out[1], out1[0], atol=1e-5)

    def test_single_token_input(self):
        """Should handle single-token sequences."""
        x = torch.randn(1, 1, self.embed_dim)
        output = self.attn(x)
        assert output.shape == (1, 1, self.embed_dim)

    def test_gradients_flow(self):
        """Gradients should flow through the attention layer."""
        self.attn.train()
        x = torch.randn(1, self.seq_length, self.embed_dim, requires_grad=True)
        output = self.attn(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.all(x.grad == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
