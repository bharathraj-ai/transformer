"""
train.py - Model Training Script
===================================
Trains the Transformer language model on the text dataset.

Training Process:
1. Load and tokenize the dataset.
2. Create training/validation sequences.
3. Build the Transformer model.
4. Train using cross-entropy loss and AdamW optimizer.
5. Track and plot training/validation loss.
6. Save the trained model, config, and vocabulary.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works without display)
import matplotlib.pyplot as plt
import time
import json
import os
import random

from tokenizer import WordTokenizer
from prepare_data import load_dataset, split_dataset, create_sequences, create_dataloaders
from model import TransformerLM, get_default_config, save_model


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train for one epoch (one pass through all training data).

    Returns the average training loss for this epoch.
    """
    model.train()
    total_loss = 0
    num_batches = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass: get predictions
        logits = model(inputs)  # (batch, seq_len, vocab_size)

        # Reshape for cross-entropy loss
        # Cross-entropy expects: (N, C) predictions and (N,) targets
        # where C = number of classes (vocab_size)
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)   # (batch*seq_len, vocab_size)
        targets_flat = targets.view(-1)              # (batch*seq_len,)

        # Compute loss
        loss = criterion(logits_flat, targets_flat)

        # Backward pass: compute gradients
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation data.

    Returns the average validation loss.
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():  # No gradients needed for evaluation
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = targets.view(-1)

            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def plot_losses(train_losses, val_losses, save_path="saved_model/loss_plot.png"):
    """Plot and save the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="#4A90D9", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", color="#E74C3C", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Progress", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss plot saved to {save_path}")


def main():
    """Main training function."""
    print("=" * 60)
    print("  Transformer Next-Token Prediction - Training")
    print("=" * 60)

    # -- Configuration --
    config = get_default_config()
    set_seed(config["seed"])

    # Device selection (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # -- Step 1: Load dataset --
    print("\n-- Loading Dataset --")
    sentences = load_dataset("dataset.txt")

    # -- Step 2: Split into train/validation --
    print("\n-- Splitting Dataset --")
    train_sentences, val_sentences = split_dataset(sentences, val_ratio=0.15,
                                                   seed=config["seed"])

    # -- Step 3: Build tokenizer --
    print("\n-- Building Tokenizer --")
    tokenizer = WordTokenizer()
    # Build vocabulary on ALL sentences (both train and val need to be encodable)
    all_text = " ".join(sentences)
    tokenizer.build_vocabulary(all_text)

    # Save vocabulary for inference
    vocab_path = os.path.join("saved_model", "vocab.json")
    tokenizer.save_vocabulary(vocab_path)

    # -- Step 4: Create training sequences --
    print("\n-- Creating Sequences --")
    seq_length = config["seq_length"]

    train_inputs, train_targets = create_sequences(
        train_sentences, tokenizer, seq_length=seq_length
    )
    val_inputs, val_targets = create_sequences(
        val_sentences, tokenizer, seq_length=seq_length
    )

    if len(train_inputs) == 0:
        print("ERROR: No training sequences created. Check dataset and seq_length.")
        return

    # -- Step 5: Create DataLoaders --
    print("\n-- Creating DataLoaders --")
    train_loader, val_loader = create_dataloaders(
        train_inputs, train_targets,
        val_inputs, val_targets,
        batch_size=config["batch_size"],
        seed=config["seed"],
    )

    # -- Step 6: Build model --
    print("\n-- Building Model --")
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config["embed_dim"],
        seq_length=seq_length,
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Vocabulary size:  {tokenizer.vocab_size}")
    print(f"Embedding dim:    {config['embed_dim']}")
    print(f"Sequence length:  {seq_length}")
    print(f"Transformer layers: {config['num_layers']}")

    # -- Step 7: Set up training --
    # Cross-entropy loss: measures how well predicted probabilities
    # match the actual next tokens. Lower = better predictions.
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <PAD> tokens

    # AdamW optimizer: Adam with proper weight decay (regularization).
    # Adapts learning rate per-parameter for faster convergence.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=0.01,
    )

    # Learning rate scheduler: reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"], eta_min=1e-5
    )

    # -- Step 8: Training loop --
    print(f"\n-- Training ({config['num_epochs']} epochs) --")
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, config["num_epochs"] + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Step the scheduler
        scheduler.step()

        # Save best model (silently)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            config_to_save = {
                **config,
                "vocab_size": tokenizer.vocab_size,
            }
            # Save silently (no print each time)
            os.makedirs("saved_model", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("saved_model", "model.pth"))
            config_path = os.path.join("saved_model", "config.json")
            with open(config_path, "w") as f:
                json.dump(config_to_save, f, indent=2)

        # Print progress every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{config['num_epochs']}  |  "
                  f"Train Loss: {train_loss:.4f}  |  "
                  f"Val Loss: {val_loss:.4f}  |  "
                  f"LR: {lr:.6f}  |  "
                  f"Time: {elapsed:.1f}s")

    total_time = time.time() - start_time

    # -- Step 9: Save results --
    print(f"\n-- Training Complete --")
    print(f"Total time: {total_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final training loss:  {train_losses[-1]:.4f}")

    # Save the final model (for tiny datasets, the fully trained model
    # gives better predictions even if val loss is higher)
    config_to_save = {**config, "vocab_size": tokenizer.vocab_size}
    save_model(model, config_to_save, save_dir="saved_model")

    # Plot losses
    plot_losses(train_losses, val_losses)

    # Save loss history
    loss_history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "total_time": total_time,
    }
    with open("saved_model/loss_history.json", "w") as f:
        json.dump(loss_history, f, indent=2)
    print("Loss history saved to saved_model/loss_history.json")

    # -- Step 10: Quick test --
    print("\n-- Quick Prediction Test --")
    model.eval()
    test_prompt = "the model"
    tokens = tokenizer.encode(test_prompt)
    if len(tokens) > 0:
        input_tensor = torch.tensor([tokens], device=device)
        with torch.no_grad():
            logits = model(input_tensor)
        # Get prediction for last position
        last_logits = logits[0, -1, :]  # (vocab_size,)
        probs = torch.softmax(last_logits, dim=-1)
        top5 = torch.topk(probs, min(5, probs.shape[0]))

        print(f"Prompt: '{test_prompt}'")
        print("Top 5 predictions:")
        for i in range(len(top5.indices)):
            token_id = top5.indices[i].item()
            prob = top5.values[i].item()
            word = tokenizer.idx2word.get(token_id, "<UNK>")
            print(f"  {i+1}. '{word}' ({prob:.4f})")

    print("\n" + "=" * 60)
    print("  Training complete! Run 'streamlit run app.py' to use the model.")
    print("=" * 60)


if __name__ == "__main__":
    main()
