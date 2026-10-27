"""
test_tokenizer.py - Tests for the Word-Level Tokenizer
=========================================================
Tests vocabulary building, encoding, decoding, unknown word handling,
punctuation treatment, and vocabulary save/load.
"""

import os
import sys
import tempfile
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import WordTokenizer


class TestWordTokenizer:
    """Test suite for WordTokenizer."""

    def setup_method(self):
        """Create a fresh tokenizer for each test."""
        self.tokenizer = WordTokenizer()
        self.sample_text = (
            "The cat sat on the mat. "
            "A dog ran in the park. "
            "The cat is happy."
        )
        self.tokenizer.build_vocabulary(self.sample_text)

    def test_vocabulary_has_special_tokens(self):
        """Vocabulary must include <PAD> and <UNK> tokens."""
        assert "<PAD>" in self.tokenizer.word2idx
        assert "<UNK>" in self.tokenizer.word2idx
        assert self.tokenizer.word2idx["<PAD>"] == 0
        assert self.tokenizer.word2idx["<UNK>"] == 1

    def test_vocabulary_size(self):
        """Vocabulary size should match number of unique tokens + special tokens."""
        assert self.tokenizer.vocab_size > 2  # More than just PAD and UNK
        assert self.tokenizer.vocab_size == len(self.tokenizer.word2idx)

    def test_encode_known_words(self):
        """Encoding known words should return valid IDs (not UNK)."""
        encoded = self.tokenizer.encode("the cat")
        unk_id = self.tokenizer.word2idx["<UNK>"]
        assert len(encoded) == 2
        assert all(idx != unk_id for idx in encoded)

    def test_encode_unknown_words(self):
        """Unknown words should be encoded as <UNK>."""
        encoded = self.tokenizer.encode("elephant zebra")
        unk_id = self.tokenizer.word2idx["<UNK>"]
        assert all(idx == unk_id for idx in encoded)

    def test_decode_roundtrip(self):
        """Encoding then decoding should return the original text (for known words)."""
        text = "the cat sat on the mat"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        assert decoded == text

    def test_punctuation_as_separate_tokens(self):
        """Punctuation should be treated as separate tokens."""
        encoded = self.tokenizer.encode("hello, world!")
        # 'hello' might be UNK, but ',' and '!' should be separate tokens
        assert len(encoded) >= 3  # At least: word, comma, word, exclamation

    def test_case_insensitivity(self):
        """Tokenizer should be case-insensitive (all lowercase)."""
        enc1 = self.tokenizer.encode("The Cat")
        enc2 = self.tokenizer.encode("the cat")
        assert enc1 == enc2

    def test_empty_input(self):
        """Empty input should return empty list."""
        encoded = self.tokenizer.encode("")
        assert encoded == []

    def test_decode_empty(self):
        """Decoding empty list should return empty string."""
        decoded = self.tokenizer.decode([])
        assert decoded == ""

    def test_decode_filters_padding(self):
        """Decoding should filter out <PAD> tokens."""
        pad_id = self.tokenizer.word2idx["<PAD>"]
        token_id = self.tokenizer.word2idx.get("the", 1)
        decoded = self.tokenizer.decode([token_id, pad_id, pad_id])
        assert "<PAD>" not in decoded

    def test_reverse_mapping(self):
        """idx2word should be the exact reverse of word2idx."""
        for word, idx in self.tokenizer.word2idx.items():
            assert self.tokenizer.idx2word[idx] == word

    def test_save_and_load_vocabulary(self):
        """Vocabulary should be identical after save/load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, "vocab.json")

            # Save
            self.tokenizer.save_vocabulary(vocab_path)
            assert os.path.exists(vocab_path)

            # Load into new tokenizer
            new_tokenizer = WordTokenizer()
            new_tokenizer.load_vocabulary(vocab_path)

            # Compare
            assert new_tokenizer.vocab_size == self.tokenizer.vocab_size
            assert new_tokenizer.word2idx == self.tokenizer.word2idx

            # Verify encoding produces same results
            text = "the cat sat"
            assert new_tokenizer.encode(text) == self.tokenizer.encode(text)

    def test_all_tokens_have_unique_ids(self):
        """Every token in vocabulary should have a unique ID."""
        ids = list(self.tokenizer.word2idx.values())
        assert len(ids) == len(set(ids)), "Duplicate IDs found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
