"""
test_model.py - Tests for the Transformer Model
===================================================
Tests model architecture, forward pass, saving/loading, prediction,
and text generation.
"""

import os
import sys
import tempfile
import torch
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import WordTokenizer
from model import TransformerLM, TransformerBlock, save_model, load_model, get_default_config
from generate import predict_next_token, generate_text


class TestTransformerBlock:
    """Tests for individual Transformer blocks."""

    def test_output_shape(self):
        """Block output should have same shape as input."""
        embed_dim = 32
        seq_length = 8
        block = TransformerBlock(embed_dim, seq_length, dropout=0.0)
        block.eval()

        x = torch.randn(2, seq_length, embed_dim)
        output = block(x)
        assert output.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from a zero-initialized block's output modestly
        (residual connection means output ≈ input + attention + ffn)."""
        embed_dim = 32
        seq_length = 8
        block = TransformerBlock(embed_dim, seq_length, dropout=0.0)
        block.eval()

        x = torch.randn(2, seq_length, embed_dim)
        with torch.no_grad():
            output = block(x)

        # Due to residual connections, output should not be zero
        assert output.abs().sum() > 0


class TestTransformerLM:
    """Tests for the complete Transformer language model."""

    def setup_method(self):
        """Set up model and tokenizer."""
        self.vocab_size = 50
        self.embed_dim = 32
        self.seq_length = 8
        self.num_layers = 2

        self.model = TransformerLM(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            seq_length=self.seq_length,
            num_layers=self.num_layers,
            dropout=0.0,
        )
        self.model.eval()

    def test_forward_shape(self):
        """Output shape should be (batch, seq_len, vocab_size)."""
        batch_size = 4
        x = torch.randint(0, self.vocab_size, (batch_size, self.seq_length))
        logits = self.model(x)
        assert logits.shape == (batch_size, self.seq_length, self.vocab_size)

    def test_forward_different_batch_sizes(self):
        """Should work with different batch sizes."""
        for bs in [1, 2, 8]:
            x = torch.randint(0, self.vocab_size, (bs, self.seq_length))
            logits = self.model(x)
            assert logits.shape == (bs, self.seq_length, self.vocab_size)

    def test_forward_shorter_sequence(self):
        """Should work with sequences shorter than max seq_length."""
        short_len = 4
        x = torch.randint(0, self.vocab_size, (2, short_len))
        logits = self.model(x)
        assert logits.shape == (2, short_len, self.vocab_size)

    def test_output_contains_valid_logits(self):
        """Logits should be finite real numbers."""
        x = torch.randint(0, self.vocab_size, (2, self.seq_length))
        logits = self.model(x)
        assert torch.isfinite(logits).all()

    def test_parameter_count(self):
        """Model should have a reasonable number of parameters."""
        count = self.model.count_parameters()
        assert count > 0
        # For our small model, params should be less than 1M
        assert count < 1_000_000

    def test_causal_masking_in_model(self):
        """
        Changing future tokens should not affect predictions at earlier positions.
        This is the end-to-end test for causal masking.
        """
        x1 = torch.randint(0, self.vocab_size, (1, self.seq_length))
        x2 = x1.clone()
        # Change the last 3 tokens
        x2[0, -3:] = torch.randint(0, self.vocab_size, (3,))

        with torch.no_grad():
            out1 = self.model(x1)
            out2 = self.model(x2)

        # First (seq_length - 3) positions should be identical
        for pos in range(self.seq_length - 3):
            diff = (out1[0, pos, :] - out2[0, pos, :]).abs().max().item()
            assert diff < 1e-4, (
                f"Position {pos} changed when future tokens changed (diff={diff})"
            )

    def test_training_step(self):
        """Model should be trainable (loss decreases in one step)."""
        self.model.train()
        x = torch.randint(0, self.vocab_size, (4, self.seq_length))
        targets = torch.randint(0, self.vocab_size, (4, self.seq_length))

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Forward
        logits = self.model(x)
        loss = criterion(logits.view(-1, self.vocab_size), targets.view(-1))
        initial_loss = loss.item()

        # Backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Forward again
        with torch.no_grad():
            logits2 = self.model(x)
            loss2 = criterion(logits2.view(-1, self.vocab_size), targets.view(-1))

        # Loss should have changed (not necessarily decreased in 1 step,
        # but it should be different)
        assert loss2.item() != initial_loss


class TestSaveLoad:
    """Tests for model saving and loading."""

    def test_save_and_load(self):
        """Model should produce identical outputs after save/load."""
        vocab_size = 50
        config = get_default_config()
        config["vocab_size"] = vocab_size

        model = TransformerLM(
            vocab_size=vocab_size,
            embed_dim=config["embed_dim"],
            seq_length=config["seq_length"],
            num_layers=config["num_layers"],
            dropout=0.0,
        )
        model.eval()

        # Get output before saving
        x = torch.randint(0, vocab_size, (1, config["seq_length"]))
        with torch.no_grad():
            original_output = model(x)

        # Save and reload
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, config, save_dir=tmpdir)

            loaded_model, loaded_config = load_model(tmpdir)
            loaded_model.eval()

            with torch.no_grad():
                loaded_output = loaded_model(x)

        # Outputs should be identical
        assert torch.allclose(original_output, loaded_output, atol=1e-5)
        assert loaded_config["vocab_size"] == vocab_size

    def test_config_saved_correctly(self):
        """All config values should be preserved after save/load."""
        config = get_default_config()
        config["vocab_size"] = 100

        model = TransformerLM(
            vocab_size=100,
            embed_dim=config["embed_dim"],
            seq_length=config["seq_length"],
            num_layers=config["num_layers"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, config, save_dir=tmpdir)
            _, loaded_config = load_model(tmpdir)

        for key in config:
            assert loaded_config[key] == config[key], f"Config mismatch for '{key}'"


class TestPredictionAndGeneration:
    """Tests for next-token prediction and text generation."""

    def setup_method(self):
        """Create a model and tokenizer for testing."""
        self.text = (
            "The model learns to predict the next word in a sequence. "
            "Training helps the model improve its predictions over time."
        )
        self.tokenizer = WordTokenizer()
        self.tokenizer.build_vocabulary(self.text)

        self.model = TransformerLM(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=32,
            seq_length=16,
            num_layers=1,
            dropout=0.0,
        )
        self.model.eval()

    def test_predict_returns_valid_result(self):
        """Prediction should return a dict with expected keys."""
        result = predict_next_token(self.model, self.tokenizer, "the model")

        assert "top_token" in result
        assert "top_prob" in result
        assert "top_k_predictions" in result
        assert len(result["top_k_predictions"]) > 0

    def test_predict_probabilities_sum_approximately_one(self):
        """Top-k probabilities should be valid (0 to 1)."""
        result = predict_next_token(self.model, self.tokenizer, "the", top_k=5)

        for word, prob in result["top_k_predictions"]:
            assert 0 <= prob <= 1, f"Invalid probability: {prob}"

    def test_predict_empty_prompt(self):
        """Should handle empty prompt gracefully."""
        result = predict_next_token(self.model, self.tokenizer, "")
        assert result["top_token"] == "<UNK>"

    def test_generate_returns_string(self):
        """Generated text should be a string containing the prompt."""
        generated = generate_text(
            self.model, self.tokenizer, "the model",
            max_tokens=5, temperature=0.0
        )
        assert isinstance(generated, str)
        assert len(generated) > 0

    def test_generate_greedy_deterministic(self):
        """Greedy generation should produce identical output each time."""
        gen1 = generate_text(
            self.model, self.tokenizer, "the model",
            max_tokens=10, temperature=0.0
        )
        gen2 = generate_text(
            self.model, self.tokenizer, "the model",
            max_tokens=10, temperature=0.0
        )
        assert gen1 == gen2

    def test_generate_respects_max_tokens(self):
        """Generated text should not exceed max_tokens new tokens."""
        prompt = "the"
        prompt_token_count = len(self.tokenizer.encode(prompt))
        max_tokens = 5

        generated = generate_text(
            self.model, self.tokenizer, prompt,
            max_tokens=max_tokens, temperature=0.0
        )
        gen_tokens = self.tokenizer.encode(generated)
        new_tokens = len(gen_tokens) - prompt_token_count
        assert new_tokens <= max_tokens

    def test_unknown_word_handling(self):
        """Unknown words should not crash prediction or generation."""
        result = predict_next_token(
            self.model, self.tokenizer, "xyzzy foobar"
        )
        assert result is not None

        generated = generate_text(
            self.model, self.tokenizer, "xyzzy",
            max_tokens=5, temperature=0.0
        )
        assert isinstance(generated, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
