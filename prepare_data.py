"""
prepare_data.py - Dataset Loading & Training Sequence Creation
================================================================
Handles:
1. Loading and cleaning the text dataset.
2. Splitting into training and validation sets (by sentences, no leakage).
3. Creating fixed-length input/target sequence pairs for next-token prediction.

Key design choice: sequences never cross sentence boundaries, so the model
doesn't learn to connect unrelated sentences.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import WordTokenizer


def load_dataset(filepath="dataset.txt"):
    """
    Load the text file and return a list of cleaned sentences.

    Each non-empty line becomes one sentence.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Keep only non-empty lines, strip whitespace
    sentences = [line.strip() for line in lines if line.strip()]
    print(f"Loaded {len(sentences)} sentences from {filepath}")
    return sentences


def split_dataset(sentences, val_ratio=0.15, seed=42):
    """
    Split sentences into training and validation sets.

    We split by sentence (not by token) to prevent data leakage -
    no validation sentence appears in training data.
    """
    import random
    random.seed(seed)

    # Shuffle a copy so we don't modify the original
    shuffled = sentences.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - val_ratio))
    train_sentences = shuffled[:split_idx]
    val_sentences = shuffled[split_idx:]

    print(f"Train: {len(train_sentences)} sentences, Val: {len(val_sentences)} sentences")
    return train_sentences, val_sentences


def create_sequences(sentences, tokenizer, seq_length=16):
    """
    Create input-target pairs for next-token prediction.

    For each sentence:
    1. Encode it into token IDs.
    2. Slide a window of size (seq_length + 1) across the tokens.
    3. Input  = first seq_length tokens   [t0, t1, ..., t_{n-1}]
       Target = last  seq_length tokens   [t1, t2, ..., t_n]

    The target is the input shifted right by one - this teaches the model
    to predict the next token at every position.

    Sequences stay within a single sentence (no cross-boundary examples).
    """
    inputs = []
    targets = []

    for sentence in sentences:
        token_ids = tokenizer.encode(sentence)

        # Need at least seq_length + 1 tokens to make one example
        if len(token_ids) < seq_length + 1:
            # Pad short sentences to create at least one example
            if len(token_ids) >= 2:
                # Use what we have (shorter sequence)
                pad_length = seq_length + 1 - len(token_ids)
                padded = token_ids + [tokenizer.word2idx[tokenizer.pad_token]] * pad_length
                inputs.append(padded[:seq_length])
                targets.append(padded[1:seq_length + 1])
            continue

        # Slide window across the sentence
        for i in range(len(token_ids) - seq_length):
            input_seq = token_ids[i:i + seq_length]
            target_seq = token_ids[i + 1:i + seq_length + 1]
            inputs.append(input_seq)
            targets.append(target_seq)

    print(f"Created {len(inputs)} training sequences (seq_length={seq_length})")
    return inputs, targets


class TextDataset(Dataset):
    """PyTorch Dataset for our token sequences."""

    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_dataloaders(train_inputs, train_targets, val_inputs, val_targets,
                       batch_size=32, seed=42):
    """Create PyTorch DataLoaders for training and validation."""
    # Set generator seed for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataset = TextDataset(train_inputs, train_targets)
    val_dataset = TextDataset(val_inputs, val_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


# -- Quick test when run directly --
if __name__ == "__main__":
    sentences = load_dataset("dataset.txt")
    train_sents, val_sents = split_dataset(sentences)

    tokenizer = WordTokenizer()
    # Build vocab on ALL sentences (vocab should cover full dataset)
    tokenizer.build_vocabulary(" ".join(sentences))

    train_in, train_tgt = create_sequences(train_sents, tokenizer, seq_length=16)
    val_in, val_tgt = create_sequences(val_sents, tokenizer, seq_length=16)

    train_loader, val_loader = create_dataloaders(
        train_in, train_tgt, val_in, val_tgt, batch_size=32
    )

    # Show one batch
    for batch_inputs, batch_targets in train_loader:
        print(f"\nBatch input shape:  {batch_inputs.shape}")
        print(f"Batch target shape: {batch_targets.shape}")
        print(f"First input:  {batch_inputs[0].tolist()}")
        print(f"First target: {batch_targets[0].tolist()}")
        print(f"Decoded input:  {tokenizer.decode(batch_inputs[0].tolist())}")
        print(f"Decoded target: {tokenizer.decode(batch_targets[0].tolist())}")
        break
