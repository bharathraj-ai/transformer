"""
tokenizer.py - Word-Level Tokenizer
=====================================
Converts raw text into numerical token IDs and back.

How it works:
1. Text is lowercased and split into words (punctuation treated as separate tokens).
2. A vocabulary maps each unique word to a unique integer ID.
3. Special tokens: <PAD> (padding), <UNK> (unknown words).
4. The vocabulary is saved/loaded as JSON so inference uses the same mapping.
"""

import re
import json
import os


class WordTokenizer:
    """A simple word-level tokenizer that splits text into words and punctuation."""

    def __init__(self):
        # Special tokens with reserved IDs
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        # word -> id  and  id -> word  mappings
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def _tokenize_text(self, text):
        """
        Split text into tokens (words + punctuation).

        We lowercase everything for consistency, then use a regex to extract
        sequences of letters/digits (words) and individual punctuation marks.

        Example: "Hello, world!" -> ["hello", ",", "world", "!"]
        """
        text = text.lower().strip()
        # Match word characters OR individual punctuation/special chars
        tokens = re.findall(r"[a-z0-9]+|[.,!?;:'\"-]", text)
        return tokens

    def build_vocabulary(self, text):
        """
        Build the vocabulary from a text string.

        Steps:
        1. Tokenize the entire text.
        2. Collect all unique tokens.
        3. Assign an ID to each token (starting after special tokens).
        """
        tokens = self._tokenize_text(text)
        unique_tokens = sorted(set(tokens))  # Sort for reproducibility

        # Start with special tokens
        self.word2idx = {
            self.pad_token: 0,
            self.unk_token: 1,
        }

        # Add each unique word
        for idx, token in enumerate(unique_tokens, start=2):
            self.word2idx[token] = idx

        # Build reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        print(f"Vocabulary built: {self.vocab_size} tokens")
        print(f"  Special tokens: {self.pad_token} (0), {self.unk_token} (1)")
        print(f"  Example words: {list(self.word2idx.items())[2:7]}")

    def encode(self, text):
        """
        Convert text string -> list of token IDs.

        Unknown words (not in vocabulary) get the <UNK> ID.
        """
        tokens = self._tokenize_text(text)
        unk_id = self.word2idx[self.unk_token]
        return [self.word2idx.get(token, unk_id) for token in tokens]

    def decode(self, token_ids):
        """
        Convert list of token IDs -> text string.

        Joins tokens with spaces (simple reconstruction).
        """
        unk_token = self.unk_token
        words = [self.idx2word.get(idx, unk_token) for idx in token_ids]
        # Filter out padding tokens
        words = [w for w in words if w != self.pad_token]
        return " ".join(words)

    def save_vocabulary(self, filepath):
        """Save vocabulary to a JSON file for later use during inference."""
        vocab_data = {
            "word2idx": self.word2idx,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2)
        print(f"Vocabulary saved to {filepath}")

    def load_vocabulary(self, filepath):
        """Load vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.word2idx = vocab_data["word2idx"]
        self.pad_token = vocab_data["pad_token"]
        self.unk_token = vocab_data["unk_token"]

        # Rebuild reverse mapping (JSON keys are strings, so convert IDs)
        self.idx2word = {int(idx): word for word, idx in self.word2idx.items()}
        # Also keep string keys for safety
        self.idx2word.update({idx: word for word, idx in self.word2idx.items()
                             if isinstance(idx, int)})
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary loaded: {self.vocab_size} tokens")


# -- Quick test when run directly --
if __name__ == "__main__":
    sample = "Artificial intelligence is transforming the world. AI is powerful!"
    tok = WordTokenizer()
    tok.build_vocabulary(sample)

    encoded = tok.encode("AI is powerful")
    print(f"\nEncoded 'AI is powerful': {encoded}")
    print(f"Decoded back: {tok.decode(encoded)}")

    # Test unknown word
    encoded_unk = tok.encode("deep learning is amazing")
    print(f"\nEncoded 'deep learning is amazing': {encoded_unk}")
    print(f"Decoded back: {tok.decode(encoded_unk)}")
