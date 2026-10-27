"""
generate.py - Next-Token Prediction & Text Generation
========================================================
This module provides two key capabilities:

1. Next-Token Prediction:
   Given a prompt, predict the most likely next word.

2. Autoregressive Text Generation:
   Generate a sequence of tokens by repeatedly predicting the next token
   and appending it to the input.

Decoding Strategies:
--------------------
• Greedy: Always pick the highest-probability token. Deterministic but
  can be repetitive.

• Temperature Sampling: Scale logits by a temperature value before
  sampling. Temperature > 1 = more random/creative. Temperature < 1 =
  more focused/deterministic. Temperature = 1 = standard sampling.

Usage:
    python generate.py --prompt "artificial intelligence" --max_tokens 20
"""

import torch
import argparse
import os

from tokenizer import WordTokenizer
from model import load_model


def predict_next_token(model, tokenizer, prompt, device="cpu", top_k=5):
    """
    Predict the next token given a text prompt.

    Args:
        model: Trained TransformerLM model
        tokenizer: WordTokenizer with loaded vocabulary
        prompt: Input text string
        device: 'cpu' or 'cuda'
        top_k: Number of top predictions to return

    Returns:
        dict with:
            - 'top_token': most likely next word
            - 'top_prob': probability of the top token
            - 'top_k_predictions': list of (word, probability) tuples
    """
    model.eval()

    # Tokenize the prompt
    token_ids = tokenizer.encode(prompt)

    if len(token_ids) == 0:
        return {
            "top_token": "<UNK>",
            "top_prob": 0.0,
            "top_k_predictions": [],
        }

    # Trim to model's maximum sequence length
    seq_length = model.seq_length
    if len(token_ids) > seq_length:
        token_ids = token_ids[-seq_length:]

    # Create input tensor
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Get model predictions
    with torch.no_grad():
        logits = model(input_tensor)  # (1, seq_len, vocab_size)

    # Get logits for the LAST position (predicting what comes next)
    last_logits = logits[0, -1, :]  # (vocab_size,)

    # Convert to probabilities
    probs = torch.softmax(last_logits, dim=-1)

    # Get top-k predictions
    actual_k = min(top_k, probs.shape[0])
    top_values, top_indices = torch.topk(probs, actual_k)

    top_k_predictions = []
    for i in range(actual_k):
        token_id = top_indices[i].item()
        prob = top_values[i].item()
        word = tokenizer.idx2word.get(token_id, "<UNK>")
        top_k_predictions.append((word, prob))

    return {
        "top_token": top_k_predictions[0][0],
        "top_prob": top_k_predictions[0][1],
        "top_k_predictions": top_k_predictions,
    }


def generate_text(model, tokenizer, prompt, max_tokens=20,
                  temperature=1.0, device="cpu"):
    """
    Generate text autoregressively.

    Process:
    1. Tokenize the prompt.
    2. Feed tokens into the model.
    3. Get the prediction for the next token.
    4. Append the predicted token to the sequence.
    5. Repeat steps 2-4 for max_tokens iterations.

    Args:
        model: Trained TransformerLM model
        tokenizer: WordTokenizer with loaded vocabulary
        prompt: Starting text
        max_tokens: Number of new tokens to generate
        temperature: Controls randomness (0.1=focused, 1.0=normal, 2.0=creative)
        device: 'cpu' or 'cuda'

    Returns:
        The complete generated text (prompt + generated tokens)
    """
    model.eval()

    # Tokenize the prompt
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        return prompt

    generated_ids = list(token_ids)
    seq_length = model.seq_length

    for _ in range(max_tokens):
        # Use only the last seq_length tokens as input
        # (the model has a fixed context window)
        context = generated_ids[-seq_length:]
        input_tensor = torch.tensor([context], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor)  # (1, len(context), vocab_size)

        # Get logits for the last position
        next_logits = logits[0, -1, :]  # (vocab_size,)

        if temperature <= 0.01:
            # -- Greedy decoding --
            # Simply pick the highest-scoring token.
            next_token_id = next_logits.argmax().item()
        else:
            # -- Temperature sampling --
            # Divide logits by temperature before softmax:
            # - Low temp (0.1): sharper distribution → more deterministic
            # - High temp (2.0): flatter distribution → more random
            scaled_logits = next_logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)

            # Sample from the distribution
            next_token_id = torch.multinomial(probs, num_samples=1).item()

        # Stop if we generate a padding token
        if next_token_id == tokenizer.word2idx.get(tokenizer.pad_token, 0):
            break

        generated_ids.append(next_token_id)

    # Decode back to text
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def main():
    """Command-line interface for text generation."""
    parser = argparse.ArgumentParser(description="Generate text with the trained model")
    parser.add_argument("--prompt", type=str, default="the model",
                        help="Starting text for generation")
    parser.add_argument("--max_tokens", type=int, default=20,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0=greedy, 1=normal, >1=creative)")
    parser.add_argument("--model_dir", type=str, default="saved_model",
                        help="Directory containing saved model")
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(os.path.join(args.model_dir, "model.pth")):
        print("Error: No trained model found. Run 'python train.py' first.")
        return

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model and tokenizer
    print("Loading model...")
    model, config = load_model(args.model_dir, device=device)

    tokenizer = WordTokenizer()
    tokenizer.load_vocabulary(os.path.join(args.model_dir, "vocab.json"))

    # -- Next-Token Prediction --
    print("\n-- Next-Token Prediction --")
    result = predict_next_token(model, tokenizer, args.prompt, device=device)
    print(f"Prompt: '{args.prompt}'")
    print(f"Predicted next token: '{result['top_token']}' "
          f"(probability: {result['top_prob']:.4f})")
    print("\nTop 5 predictions:")
    for i, (word, prob) in enumerate(result["top_k_predictions"]):
        bar = "#" * int(prob * 40)
        print(f"  {i+1}. '{word}' {bar} {prob:.4f}")

    # -- Text Generation --
    print(f"\n-- Text Generation (temperature={args.temperature}) --")
    generated = generate_text(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=device,
    )
    print(f"Generated: {generated}")

    # Also show greedy generation for comparison
    print(f"\n-- Greedy Generation --")
    greedy = generate_text(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        temperature=0.0,
        device=device,
    )
    print(f"Generated: {greedy}")


if __name__ == "__main__":
    main()
