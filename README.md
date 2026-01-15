# GPT-256M: Custom Causal Transformer Implementation

A 256-million parameter Large Language Model (LLM) engineered from scratch using **PyTorch**. This project implements a decoder-only Transformer architecture designed for autoregressive text generation, following the design principles of GPT-3 and LLaMA.

The implementation avoids high-level abstractions (like HuggingFace `AutoModel`) to demonstrate a first-principles understanding of deep learning architectures, tensor manipulations, and attention mechanisms.

## 🚀 Key Features
*   **Scale:** ~256M trainable parameters.
*   **Architecture:** Decoder-only Transformer with Causal Masking.
*   **Attention:** Multi-Head Self-Attention (MHSA) with scaled dot-product.
*   **Activation:** Gaussian Error Linear Units (GELU) for smoother gradient flow.
*   **Normalization:** Pre-Layer Normalization for stable convergence in deep networks.
*   **Tokenization:** Byte-Pair Encoding (BPE) integration.

---

## 📂 Project Structure & File Modules

The architecture is modularized into distinct components to ensure readability and easy debugging. Here is the breakdown of the codebase:

### 1. Core Architecture
*   **`model.py`**
    *   The main entry point of the LLM.
    *   Assembles the Embeddings, Positional Encodings, stacked Transformer Blocks, and the final Linear Head.
    *   Contains the `forward()` pass logic and weight initialization methods.

*   **`transformer_block.py`**
    *   Defines a single Transformer Layer.
    *   Orchestrates the flow of data: `Input -> LayerNorm -> Attention -> Residual -> LayerNorm -> FeedForward -> Residual`.

### 2. Neural Components
*   **`attention.py`**
    *   Implements **Multi-Head Self-Attention**.
    *   Handles query/key/value projections and the scaled dot-product calculation.
    *   **Crucial:** Applies the **Causal Mask** (lower triangular matrix) to ensure the model cannot "see" future tokens during training.

*   **`feed_forward.py`**
    *   The Position-wise Feed-Forward Network (FFN).
    *   Implements the expansion and contraction of the hidden dimension (typically 4x the model dimension) with non-linearity in between.

*   **`activation.py`**
    *   Contains the implementation of the **GELU (Gaussian Error Linear Unit)** activation function.
    *   Used within the feed-forward blocks to introduce non-linearity.

*   **`layer_norm.py`**
    *   A custom implementation of Layer Normalization.
    *   Normalizes input across the feature dimension to stabilize hidden state dynamics.

### 3. Utilities & Config
*   **`configuration.py`**
    *   A central configuration class (DataClass) to manage hyperparameters.
    *   Controls `vocab_size`, `n_layer` (depth), `n_head` (attention heads), `n_embd` (hidden size), and `dropout` rates.
    *   Ensures the model size scales correctly to the target 256M parameters.

*   **`tokenizer.py`**
    *   Handles the text-to-integer and integer-to-text conversion.
    *   Implements/Wraps the **Byte-Pair Encoding (BPE)** logic to handle subword tokenization efficiently.
