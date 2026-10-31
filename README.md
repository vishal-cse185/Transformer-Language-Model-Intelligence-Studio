# TRANSFORMER INTELLIGENCE STUDIO
### *Self-Attention Based Next-Token Prediction & Language Modeling Platform*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-8+-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-v4-38B2AC?logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)

---

## 🌟 Executive Overview

**Transformer Intelligence Studio** is a research and language modeling platform designed to demystify the internal mechanics of Transformers. Built from the ground up without using high-level black-box abstractions like `nn.MultiheadAttention`, pretrained GPT/BERT checkpoints, or external API wrappers, every single attention calculation, causal mask, positional encoding, and residual layer is implemented in **pure PyTorch tensor operations**.

---

## 🔬 Core Architectural Specifications

| Component | Specification |
| :--- | :--- |
| **Model Type** | Autoregressive Decoder-Only Transformer |
| **Embedding Dimension ($d_{model}$)** | **128** |
| **Attention Heads ($H$)** | **4** (Head dimension $d_k = 32$) |
| **Transformer Blocks ($N$)** | **2 Layers** (Pre-LayerNorm with Residuals) |
| **Feed-Forward Dimension ($d_{ff}$)** | **512** (GELU non-linearity) |
| **Maximum Sequence Length ($T$)** | **32 Tokens** |
| **Positional Encoding** | Sinusoidal geometric encoding |
| **Causal Attention Masking** | Strict lower-triangular $-\infty$ masking |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999, \text{lr}=10^{-3}$) |
| **Loss Function** | Cross-Entropy Loss with padding ignore |
| **Target Hardware** | Apple Silicon (MPS) / CUDA / CPU fallback |

---

## 📐 Mathematical Formulation

### 1. Manual Multi-Head Self-Attention
For input representations $X \in \mathbb{R}^{B \times T \times d_{model}}$, linear projections create Queries, Keys, and Values:
$$Q = X W_q, \quad K = X W_k, \quad V = X W_v \quad (W_q, W_k, W_v \in \mathbb{R}^{d_{model} \times d_{model}})$$

Reshaping into $H=4$ heads with $d_k=32$:
$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} + M \right) V$$

### 2. Causal Mask Matrix ($M$)
To preserve the autoregressive property during next-token prediction, token $i$ must not look ahead to future tokens $j > i$:
$$M_{i,j} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
Because $e^{-\infty} = 0$, attention weights for all future positions evaluate to exactly zero after softmax normalization.

### 3. Scaling Factor ($1/\sqrt{d_k}$)
Assuming independent random components with zero mean and unit variance:
$$\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
Dividing by $\sqrt{d_k}$ normalizes the variance back to $1$, preventing gradient vanishing in the softmax tails.

### 4. Language Modeling Objective & Perplexity
$$\mathcal{L}_{CE} = -\frac{1}{T} \sum_{t=1}^T \log P_\theta(w_t \mid w_{<t})$$
$$\mathcal{PPL} = \exp(\mathcal{L}_{CE})$$

---

## 🖥️ Application Modules

1. **MAIN DASHBOARD**
   - Telemetry status, vocabulary size, dataset sequence counts, training epochs, validation loss, perplexity, and Top-1/Top-5 accuracy.
2. **DATASET AND TOKENIZER STUDIO**
   - Documented corpus library (*TinyStories*, *AI Science*, *Shakespeare*, *Combined* or Custom text editor).
   - Regex-based tokenization with special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
   - Interactive Tokenizer Inspector: test custom sentences with chip visualizations and unknown word badges.
   - Sliding window sequence pairs ($X \to Y$) preview table.
3. **TRANSFORMER ARCHITECTURE STUDIO**
   - Interactive sequential computational dataflow diagram.
   - Component Inspector: click any component to inspect tensor shapes, parameter counts, LaTeX formulas, NLP explanations, and PyTorch code.
4. **SELF-ATTENTION VISUALIZATION**
   - Interactive Attention Heatmap with layer selector (Block 1, Block 2) and head selector (Head 1-4, Average).
   - Token Focus Inspector: click any query token to view its attention bar chart on preceding context tokens.
   - Causal mask visualizer explaining $-\infty$ lookahead prevention.
5. **MODEL TRAINING CENTER**
   - Live hyperparameter tuning (Epochs, Batch Size, Learning Rate, Weight Decay).
   - Real-time training progress bar, step counter, and epoch timer.
   - Dynamic Recharts loss curves plotting Train Loss vs. Validation Loss.
   - Automated model checkpointing (`backend/outputs/best_model.pt`).
6. **MODEL EVALUATION**
   - Held-out test set evaluation: Cross-Entropy loss, Perplexity, Top-1, Top-3, and Top-5 accuracy.
   - Top-K accuracy comparison bar chart.
   - Prediction examples table comparing Expected Target vs Model Top-1 and Top-5 candidates.
7. **NEXT-TOKEN PREDICTION PLAYGROUND**
   - Interactive prompt editor with preset prompts.
   - Candidate tokens cards showing rank, probability percentage, and raw logits.
   - Recharts Probability Distribution chart.
   - Decoding methods: Greedy (Argmax) or Temperature-based sampling ($0.1 - 2.0$) with Top-K cutoff.
   - Step-by-step "Append Next Token" button for incremental sentence building.
8. **LANGUAGE GENERATION WORKSPACE**
   - Autoregressive text generation with adjustable length, temperature, and top-K sampling.
   - Token-by-token generation trace log showing step number, emitted token, probability, and alternative candidates considered.
9. **RESEARCH AND DOCUMENTATION**
   - Theoretical encyclopedia covering Transformers vs RNNs, self-attention mechanics, mathematical proofs, causal masking, and sampling techniques.

---

## 📁 Project Structure

```text
transformer-intelligence-studio/
├── backend/
│   ├── datasets/
│   │   ├── __init__.py
│   │   └── corpus.py          # Documented English text corpora (Stories, Science, Shakespeare)
│   ├── models/
│   │   ├── __init__.py
│   │   └── transformer.py     # Pure PyTorch manual Multi-Head Self-Attention & TransformerLM
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   └── tokenizer.py       # Custom Tokenizer with special tokens & serialization
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py         # Thread-safe TrainingManager, evaluation & inference engine
│   ├── outputs/
│   │   ├── best_model.pt      # Saved PyTorch checkpoint weights
│   │   ├── vocab.json         # Persisted vocabulary mapping
│   │   ├── config.json        # Model hyperparameters
│   │   └── metrics.json       # Persisted training & evaluation metrics
│   ├── __init__.py
│   └── main.py                # FastAPI RESTful API server
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── Dashboard.tsx
│   │   │   ├── DatasetTokenizerStudio.tsx
│   │   │   ├── ArchitectureStudio.tsx
│   │   │   ├── AttentionVisualization.tsx
│   │   │   ├── TrainingCenter.tsx
│   │   │   ├── ModelEvaluation.tsx
│   │   │   ├── PredictionPlayground.tsx
│   │   │   ├── GenerationWorkspace.tsx
│   │   │   └── ResearchDocumentation.tsx
│   │   ├── services/
│   │   │   └── api.ts         # Frontend API client
│   │   ├── types/
│   │   │   └── index.ts       # TypeScript interfaces
│   │   ├── App.tsx            # Main shell & tab router
│   │   ├── main.tsx           # React entry point
│   │   └── index.css          # Tailwind CSS & glassmorphic styling
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── tests/
│   ├── __init__.py
│   └── test_transformer.py   # Unit & integration test suite (6 passing tests)
└── README.md
```

---

## 🚀 Running Locally

### 1. Backend Setup & Startup
```bash
# In the project root directory
source venv/bin/activate

# Run test suite to verify PyTorch tensor operations and strict causality
PYTHONPATH=. pytest tests/test_transformer.py -v

# Launch FastAPI backend server
PYTHONPATH=. uvicorn backend.main:app --host 127.0.0.1 --port 8008
```
*Backend API will be live at `http://127.0.0.1:8008` (Interactive Swagger docs at `http://127.0.0.1:8008/docs`).*

### 2. Frontend Setup & Startup
```bash
# In a new terminal tab
cd frontend

# Install npm dependencies (if not already installed)
npm install

# Start Vite development server
npm run dev
```
*Frontend studio will be accessible at `http://localhost:5180`.*

---

## 🧪 Automated Test Suite

Run the full PyTorch verification suite:
```bash
PYTHONPATH=. pytest tests/test_transformer.py -v
```
All 6 tests verify:
1. `test_tokenizer_encoding_decoding`: Special tokens, token2id, unknown word handling.
2. `test_positional_encoding`: Geometric sinusoidal values and shape determinism.
3. `test_manual_multi_head_attention_causal_mask`: Attention weights sum to 1.0; strict $0.00$ future weights.
4. `test_causality_strict_property`: Modifying future tokens does not affect representations of previous tokens.
5. `test_transformer_end_to_end_forward`: Forward pass and layer shapes.
6. `test_training_manager_integration`: Dataset sliding window generation, sampling, and attention inspection.
