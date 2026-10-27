"""
app.py - Streamlit Web Interface
===================================
A beautiful, interactive web app for the Transformer next-token prediction model.

Features:
- Text input for prompts
- Next-token prediction with top-5 probabilities
- Autoregressive text generation
- Training loss visualization
- Model information display

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
import json
import os

from tokenizer import WordTokenizer
from model import load_model
from generate import predict_next_token, generate_text

# -- Page Configuration --
st.set_page_config(
    page_title="Transformer Next-Token Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS for Premium Look --
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* Header styling */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }

    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: #888;
        font-size: 1.1rem;
    }

    /* Card styling */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .prediction-item {
        display: flex;
        align-items: center;
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .prediction-rank {
        font-size: 1.2rem;
        font-weight: 700;
        color: #667eea;
        min-width: 2rem;
    }

    .prediction-word {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e0e0e0;
        min-width: 120px;
    }

    .prediction-bar {
        flex: 1;
        height: 24px;
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        overflow: hidden;
        margin: 0 1rem;
    }

    .prediction-bar-fill {
        height: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.5s ease;
    }

    .prediction-prob {
        font-size: 0.95rem;
        color: #aaa;
        min-width: 60px;
        text-align: right;
    }

    /* Generated text box */
    .generated-text {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border: 1px solid #4a4a8a;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        font-size: 1.2rem;
        line-height: 1.8;
        color: #e8e8ff;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .prompt-text {
        color: #667eea;
        font-weight: 700;
    }

    .new-text {
        color: #a8edea;
        font-weight: 500;
    }

    /* Info cards */
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .info-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .info-card .label {
        color: #888;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-card .value {
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }

    /* Sidebar */
    .sidebar .sidebar-content {
        background: #0f0f1a;
    }

    /* Button override */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_cached():
    """Load model and tokenizer (cached to avoid reloading on every interaction)."""
    model_dir = "saved_model"

    if not os.path.exists(os.path.join(model_dir, "model.pth")):
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(model_dir, device=device)

    tokenizer = WordTokenizer()
    tokenizer.load_vocabulary(os.path.join(model_dir, "vocab.json"))

    return model, tokenizer, config


def render_header():
    """Render the app header."""
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Transformer Next-Token Predictor</h1>
        <p>A from-scratch Transformer model with manual causal self-attention</p>
    </div>
    """, unsafe_allow_html=True)


def render_predictions(predictions, prompt):
    """Render top-k predictions as styled cards."""
    if not predictions:
        return

    st.markdown("### 🎯 Top Predictions")

    html = '<div class="result-card">'
    max_prob = predictions[0][1] if predictions else 1.0

    for i, (word, prob) in enumerate(predictions):
        bar_width = (prob / max_prob) * 100 if max_prob > 0 else 0
        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] if i < 5 else f"{i+1}."

        html += f"""
        <div class="prediction-item">
            <span class="prediction-rank">{medal}</span>
            <span class="prediction-word">{word}</span>
            <div class="prediction-bar">
                <div class="prediction-bar-fill" style="width: {bar_width}%"></div>
            </div>
            <span class="prediction-prob">{prob:.2%}</span>
        </div>
        """

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_generated_text(prompt, full_text):
    """Render generated text with prompt highlighted."""
    # Find where the new text starts
    prompt_lower = prompt.lower()
    prompt_tokens = prompt_lower.split()
    full_tokens = full_text.split()

    # The prompt part
    prompt_display = " ".join(full_tokens[:len(prompt_tokens)])
    new_display = " ".join(full_tokens[len(prompt_tokens):])

    st.markdown("### ✨ Generated Text")
    st.markdown(f"""
    <div class="generated-text">
        <span class="prompt-text">{prompt_display}</span>
        <span class="new-text"> {new_display}</span>
    </div>
    """, unsafe_allow_html=True)


def render_model_info(config, tokenizer):
    """Render model information cards."""
    st.markdown("### 📊 Model Information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Vocabulary</div>
            <div class="value">{tokenizer.vocab_size}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Embed Dim</div>
            <div class="value">{config.get('embed_dim', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Layers</div>
            <div class="value">{config.get('num_layers', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="info-card">
            <div class="label">Seq Length</div>
            <div class="value">{config.get('seq_length', 'N/A')}</div>
        </div>
        """, unsafe_allow_html=True)


def render_loss_plot():
    """Render the training loss plot."""
    loss_plot_path = "saved_model/loss_plot.png"
    if os.path.exists(loss_plot_path):
        st.markdown("### 📉 Training Loss")
        st.image(loss_plot_path, use_container_width=True)


def main():
    """Main Streamlit application."""
    render_header()

    # Load model
    model, tokenizer, config = load_model_cached()

    if model is None:
        st.error(" No trained model found! Please run `python train.py` first.")
        st.info("Run the following command in your terminal:")
        st.code("python train.py", language="bash")
        return

    device = next(model.parameters()).device

    # -- Sidebar --
    with st.sidebar:
        st.markdown("## Settings")

        max_tokens = st.slider("Max Tokens to Generate", 5, 50, 20, step=5)
        temperature = st.slider("Temperature", 0.0, 2.0, 0.8, step=0.1,
                                help="0 = greedy (deterministic), 1 = normal, >1 = more creative")

        st.markdown("---")
        st.markdown("##  How It Works")
        st.markdown("""
        1. **Enter a prompt** in the text box
        2. **Predict** shows the top-5 next tokens
        3. **Generate** creates new text autoregressively
        4. **Temperature** controls creativity:
           - `0.0`: Always picks the most likely token
           - `0.8`: Balanced (recommended)
           - `1.5+`: More creative but less coherent
        """)

        st.markdown("---")
        st.markdown("##  Architecture")
        st.markdown("""
        - **Token + Positional Embeddings**
        - **Causal Self-Attention** (single-head)
        - **Feed-Forward Network** (GELU)
        - **Layer Normalization**
        - **Residual Connections**
        - **2 Transformer Blocks**
        """)

    # -- Main Content --
    # Prompt input
    prompt = st.text_input(
        " Enter your prompt",
        value="the model",
        placeholder="Type a starting phrase...",
        help="Enter a few words and the model will predict what comes next"
    )

    # Action buttons
    col1, col2, _ = st.columns([1, 1, 2])

    with col1:
        predict_btn = st.button(" Predict Next Token", use_container_width=True)

    with col2:
        generate_btn = st.button("✨ Generate Text", use_container_width=True)

    # -- Prediction --
    if predict_btn and prompt.strip():
        with st.spinner("Predicting..."):
            result = predict_next_token(model, tokenizer, prompt.strip(),
                                        device=str(device))

        # Show the predicted token prominently
        st.markdown(f"""
        <div style="text-align: center; margin: 1.5rem 0;">
            <span style="font-size: 1.1rem; color: #888;">Next token: </span>
            <span style="font-size: 1.6rem; font-weight: 700;
                         background: linear-gradient(135deg, #667eea, #764ba2);
                         -webkit-background-clip: text;
                         -webkit-text-fill-color: transparent;">
                {result['top_token']}
            </span>
            <span style="font-size: 1rem; color: #666;"> ({result['top_prob']:.2%})</span>
        </div>
        """, unsafe_allow_html=True)

        render_predictions(result["top_k_predictions"], prompt.strip())

    # -- Generation --
    if generate_btn and prompt.strip():
        with st.spinner("Generating text..."):
            generated = generate_text(
                model, tokenizer, prompt.strip(),
                max_tokens=max_tokens,
                temperature=temperature,
                device=str(device),
            )

        render_generated_text(prompt.strip(), generated)

    # -- Bottom sections --
    st.markdown("---")

    tab1, tab2 = st.tabs([" Model Info", " Training Loss"])

    with tab1:
        render_model_info(config, tokenizer)

        # Show some dataset info
        if os.path.exists("dataset.txt"):
            with open("dataset.txt", "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            st.markdown(f"**Dataset:** {len(lines)} sentences from `dataset.txt`")

        # Show loss history
        loss_path = "saved_model/loss_history.json"
        if os.path.exists(loss_path):
            with open(loss_path, "r") as f:
                loss_data = json.load(f)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Validation Loss",
                          f"{loss_data.get('best_val_loss', 'N/A'):.4f}")
            with col2:
                st.metric("Training Time",
                          f"{loss_data.get('total_time', 0):.1f}s")

    with tab2:
        render_loss_plot()

    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #555; font-size: 0.85rem;'>"
        "Built from scratch with PyTorch • Manual Causal Self-Attention • "
        "No pre-trained models used</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
