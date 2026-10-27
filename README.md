# 🤖 Transformer Next-Token Prediction

A complete, from-scratch Transformer-based language model that learns to predict the next token using manually implemented causal self-attention.

---

## 📋 Abstract

This project implements a small Transformer language model entirely from scratch using PyTorch. The model learns patterns from a text dataset about AI/ML concepts and predicts the next word in a sequence. Every component — from tokenization to self-attention to text generation — is built manually without pre-trained models or external AI APIs.

## 🎯 Problem Statement

Build a system that:
1. Takes a text prompt as input (e.g., "Artificial intelligence is")
2. Predicts the most likely next word
3. Generates coherent text by repeatedly predicting the next token

## 🏆 Objectives

- Implement causal self-attention from scratch
- Build a working Transformer architecture
- Train a next-token prediction model
- Create an interactive web interface
- Make everything beginner-friendly and educational

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Programming language |
| PyTorch | Deep learning framework |
| NumPy | Numerical computations |
| Matplotlib | Loss visualization |
| Streamlit | Web interface |

**No pre-trained models, Hugging Face, or external AI APIs are used.**

## 🏗️ System Architecture

```
Text Dataset
     │
     ▼
┌─────────────────┐
│  Tokenization    │  Word-level: text → token IDs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Training Data   │  Sliding window: input/target pairs
│  Preparation     │  (no cross-sentence boundaries)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Token Embedding │  Token IDs → dense vectors
│  + Position Emb  │  + positional information
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         Transformer Block (×2)       │
│  ┌─────────────────────────────┐    │
│  │   Layer Norm                 │    │
│  │   Causal Self-Attention      │    │
│  │   Residual Connection        │    │
│  │   Layer Norm                 │    │
│  │   Feed-Forward Network       │    │
│  │   Residual Connection        │    │
│  └─────────────────────────────┘    │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Layer Norm      │
│  Output Head     │  → vocabulary-sized logits
└────────┬────────┘
         │
         ▼
   Next-Token Probabilities
```

## 🔍 How Self-Attention Works

Self-attention allows each token to "look at" every previous token and decide which ones are most relevant for prediction.

### The Process:

1. **Query (Q)**: "What am I looking for?" — each token creates a query vector
2. **Key (K)**: "What do I contain?" — each token creates a key vector
3. **Value (V)**: "What information do I carry?" — each token creates a value vector
4. **Attention Score**: `Q · K^T` — dot product measures similarity
5. **Scaling**: Divide by `√d_k` to prevent extreme values
6. **Causal Mask**: Block future tokens (set to -infinity)
7. **Softmax**: Convert to probabilities (weights sum to 1)
8. **Output**: Weighted sum of values

### Why Causal?

In language modeling, when predicting the next word, the model should only see *previous* words. The causal mask enforces this by making future positions invisible.

```
Token:  [the]  [cat]  [sat]  [on]
         ↓      ↓      ↓      ↓
Sees:   [the]  [the]  [the]  [the]
               [cat]  [cat]  [cat]
                      [sat]  [sat]
                             [on]
```

## 🔄 Transformer Workflow

```
Input: "the model learns to"

1. Tokenize: [the, model, learns, to] → [45, 32, 28, 41]

2. Embed:    Each ID → 64-dimensional vector
             + positional embedding added

3. Attention: Each token attends to previous tokens
              "learns" pays attention to "model" and "the"

4. FFN:      Process each position independently

5. Output:   Predict probability distribution over vocabulary
             → "predict" has highest probability

6. Result:   "the model learns to predict"
```

## 📁 Project Structure

```
transformer/
├── dataset.txt          # Training text (72 sentences about AI/ML)
├── tokenizer.py         # Word-level tokenizer
├── prepare_data.py      # Dataset loading & sequence creation
├── attention.py         # Causal self-attention implementation
├── model.py             # Transformer model architecture
├── train.py             # Training script
├── generate.py          # Prediction & text generation
├── app.py               # Streamlit web interface
├── requirements.txt     # Dependencies
├── README.md            # This file
├── tests/
│   ├── test_tokenizer.py
│   ├── test_attention.py
│   └── test_model.py
└── saved_model/
    ├── model.pth        # Trained model weights
    ├── config.json      # Model configuration
    ├── vocab.json       # Tokenizer vocabulary
    ├── loss_history.json
    └── loss_plot.png
```

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

```bash
# 1. Navigate to project directory
cd transformer

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## 🎓 Training

```bash
# Train the model (takes ~1-2 minutes on CPU)
python train.py
```

This will:
- Load and tokenize the dataset
- Split into training/validation sets
- Train for 50 epochs
- Save the model to `saved_model/`
- Generate a loss plot

### Expected Output
```
── Training (50 epochs) ──
  Epoch   1/50  |  Train Loss: 5.xxxx  |  Val Loss: 5.xxxx
  Epoch   5/50  |  Train Loss: 4.xxxx  |  Val Loss: 4.xxxx
  ...
  Epoch  50/50  |  Train Loss: 2.xxxx  |  Val Loss: 3.xxxx

── Training Complete ──
```

## 🎮 Using the Model

### Command Line

```bash
# Generate text with default settings
python generate.py --prompt "the model" --max_tokens 20

# Greedy generation (deterministic)
python generate.py --prompt "neural networks" --temperature 0.0

# Creative generation
python generate.py --prompt "deep learning" --temperature 1.5
```

### Web Interface

```bash
# Launch Streamlit app
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tokenizer.py -v
python -m pytest tests/test_attention.py -v
python -m pytest tests/test_model.py -v
```

## 📊 Sample Outputs

> **Note**: These are illustrative. Actual outputs depend on training. This is a small model trained on a tiny dataset — it will learn patterns from the training data but won't produce human-quality text.

**Next-Token Prediction:**
```
Prompt: "the model"
Top predictions:
  1. 'learns'     ██████████████  0.15
  2. 'to'         ████████████    0.12
  3. 'performance' ████████       0.09
```

**Text Generation:**
```
Prompt: "neural networks"
Generated: "neural networks are inspired by the structure of the human brain"
```

## ⚠️ Limitations

- **Small model**: ~50K parameters (vs. billions in GPT). Limited language understanding.
- **Tiny dataset**: Only 72 sentences. The model can only generate text using words from this vocabulary.
- **Word-level tokenization**: Cannot handle subwords or out-of-vocabulary words well.
- **Single-head attention**: Uses one attention head instead of multi-head for simplicity.
- **Limited context**: 16-token context window.
- **Not a chatbot**: This model predicts the next word — it doesn't understand questions or instructions.

## 🔮 Future Enhancements

- Multi-head attention
- Larger dataset with more diverse text
- Subword tokenization (BPE/WordPiece)
- Longer context window
- Learning rate warmup
- Beam search decoding
- Top-k / top-p (nucleus) sampling
- Attention visualization
- Model evaluation with perplexity

## 📚 References

- Vaswani et al., "Attention Is All You Need" (2017)
- PyTorch Documentation: https://pytorch.org/docs/
- Streamlit Documentation: https://docs.streamlit.io/

---

*Built from scratch for educational purposes. No pre-trained models used.*
#   t r a n s f o r m e r  
 