"""FastAPI Application Server for Transformer Intelligence Studio.

Exposes RESTful endpoints for:
- Main dashboard status & telemetry
- Dataset management & custom corpus loading
- Tokenizer inspection & sequence generation
- Interactive Transformer architecture metadata
- Model training & live monitoring
- Model evaluation on held-out test sets
- Next-token prediction with top-5 distributions
- Autoregressive text generation workspace
- Multi-head self-attention heatmap inspection
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.datasets.corpus import CORPORA, get_corpus, list_corpora
from backend.training.trainer import TrainingManager

app = FastAPI(
    title="Transformer Intelligence Studio API",
    description="Backend API for Self-Attention Based Next-Token Prediction & Language Modeling Platform",
    version="1.0.0"
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate global TrainingManager
trainer_manager = TrainingManager(outputs_dir="backend/outputs")

# Initialize automatically on startup with default curated combined corpus
default_corpus = get_corpus("combined")
init_stats = trainer_manager.initialize_with_corpus(
    corpus_text=default_corpus["text"],
    corpus_id="combined"
)


# ----------------- Request / Response Models -----------------

class CorpusSelectRequest(BaseModel):
    corpus_id: str = Field(default="combined", description="Corpus identifier")
    custom_text: Optional[str] = Field(default=None, description="Optional custom corpus text")
    seq_len: int = Field(default=32, ge=8, le=64)


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input string to tokenize")


class TrainStartRequest(BaseModel):
    epochs: int = Field(default=15, ge=1, le=100)
    batch_size: int = Field(default=32, ge=4, le=128)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    weight_decay: float = Field(default=0.0001, ge=0)


class PredictNextTokenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Input prompt for next-token prediction")
    temperature: float = Field(default=1.0, ge=0.01, le=3.0)
    top_k: int = Field(default=5, ge=1, le=50)
    greedy: bool = Field(default=False)


class GenerateTextRequest(BaseModel):
    prompt: str = Field(default="Once upon a time", description="Input prompt for text generation")
    max_new_tokens: int = Field(default=20, ge=1, le=80)
    temperature: float = Field(default=0.8, ge=0.01, le=3.0)
    top_k: int = Field(default=10, ge=1, le=50)


class InspectAttentionRequest(BaseModel):
    text: str = Field(default="Attention is all you need", min_length=1)


# ----------------- API Endpoints -----------------

@app.get("/api/status")
def get_system_status() -> Dict[str, Any]:
    """Retrieve overall model status, vocabulary size, dataset splits, and latest metrics."""
    return {
        "status": "online",
        "device": str(trainer_manager.device),
        "active_corpus": trainer_manager.active_corpus_id,
        "is_training": trainer_manager.is_training,
        "training_state": trainer_manager.training_state,
        "vocab_size": trainer_manager.tokenizer.vocab_size if trainer_manager.tokenizer else 0,
        "dataset_sizes": {
            "train": len(trainer_manager.dataset_splits["train"]),
            "val": len(trainer_manager.dataset_splits["val"]),
            "test": len(trainer_manager.dataset_splits["test"])
        },
        "model_config": trainer_manager.model_config,
        "model_parameters": trainer_manager.model.count_parameters() if trainer_manager.model else {},
        "latest_evaluation": trainer_manager.latest_evaluation
    }


@app.get("/api/corpora")
def get_available_corpora() -> Dict[str, Any]:
    """List available corpora with metadata and current selection."""
    return {
        "available_corpora": list_corpora(),
        "active_corpus_id": trainer_manager.active_corpus_id
    }


@app.post("/api/dataset/select")
def select_or_upload_corpus(req: CorpusSelectRequest) -> Dict[str, Any]:
    """Switch active corpus or upload custom text."""
    if req.custom_text and req.custom_text.strip():
        corpus_text = req.custom_text.strip()
        cid = "custom"
    else:
        corpus_data = get_corpus(req.corpus_id)
        corpus_text = corpus_data["text"]
        cid = req.corpus_id

    stats = trainer_manager.initialize_with_corpus(
        corpus_text=corpus_text,
        corpus_id=cid,
        seq_len=req.seq_len
    )
    return {
        "message": f"Successfully initialized dataset with corpus '{cid}'",
        "stats": stats
    }


@app.get("/api/dataset/details")
def get_dataset_details() -> Dict[str, Any]:
    """Vocabulary statistics and sample training sequence pairs."""
    if not trainer_manager.tokenizer:
        raise HTTPException(status_code=400, detail="Tokenizer not initialized")

    vocab_stats = trainer_manager.tokenizer.get_vocab_stats()

    # Create sample sequences representation
    sample_pairs = []
    train_splits = trainer_manager.dataset_splits.get("train", [])
    for idx in range(min(5, len(train_splits))):
        x_ids, y_ids = train_splits[idx]
        x_toks = [trainer_manager.tokenizer.id2token.get(i, "<unk>") for i in x_ids[:10]]
        y_toks = [trainer_manager.tokenizer.id2token.get(i, "<unk>") for i in y_ids[:10]]
        sample_pairs.append({
            "sample_index": idx + 1,
            "input_preview": " ".join(x_toks),
            "target_preview": " ".join(y_toks),
            "input_ids_preview": x_ids[:10],
            "target_ids_preview": y_ids[:10],
            "full_length": len(x_ids)
        })

    return {
        "vocab_stats": vocab_stats,
        "corpus_id": trainer_manager.active_corpus_id,
        "sample_sequence_pairs": sample_pairs,
        "splits_count": {
            "train": len(trainer_manager.dataset_splits["train"]),
            "val": len(trainer_manager.dataset_splits["val"]),
            "test": len(trainer_manager.dataset_splits["test"])
        }
    }


@app.post("/api/tokenizer/test")
def test_tokenizer(req: TokenizeRequest) -> Dict[str, Any]:
    """Tokenize arbitrary text and return token details and statistics."""
    if not trainer_manager.tokenizer:
        raise HTTPException(status_code=400, detail="Tokenizer not initialized")

    token_details = trainer_manager.tokenizer.get_token_details(req.text)
    encoded_ids = trainer_manager.tokenizer.encode(req.text)
    decoded_text = trainer_manager.tokenizer.decode(encoded_ids)
    unk_count = sum(1 for item in token_details if item["is_unk"])

    return {
        "raw_text": req.text,
        "tokens_count": len(token_details),
        "unknown_tokens_count": unk_count,
        "token_details": token_details,
        "encoded_ids": encoded_ids,
        "reconstructed_text": decoded_text
    }


@app.get("/api/architecture")
def get_architecture_details() -> Dict[str, Any]:
    """Complete interactive architecture documentation and layer specification."""
    if not trainer_manager.model:
        raise HTTPException(status_code=400, detail="Model not initialized")

    param_info = trainer_manager.model.count_parameters()

    components = [
        {
            "id": "token_embedding",
            "name": "Token Embedding",
            "layer_type": "nn.Embedding(vocab_size, d_model)",
            "input_shape": "(batch_size, seq_len)",
            "output_shape": "(batch_size, seq_len, 128)",
            "parameters": param_info["embedding_parameters"],
            "math_formula": "X_{emb} = \\text{Embedding}(W) \\cdot \\sqrt{d_{model}}",
            "explanation": "Maps discrete integer token IDs into dense continuous vector representations in a 128-dimensional semantic space. Scaled by sqrt(d_model) to maintain gradient variance.",
            "pytorch_code": "tok_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)"
        },
        {
            "id": "positional_encoding",
            "name": "Sinusoidal Positional Encoding",
            "layer_type": "PositionalEncoding(d_model=128, max_seq_len=32)",
            "input_shape": "(batch_size, seq_len, 128)",
            "output_shape": "(batch_size, seq_len, 128)",
            "parameters": 0,
            "math_formula": "PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right), \\quad PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)",
            "explanation": "Injects sequence order into embeddings. Without positional encodings, self-attention is entirely permutation-invariant and cannot distinguish token position.",
            "pytorch_code": "x = self.pos_encoding(tok_emb)"
        },
        {
            "id": "qkv_projections",
            "name": "Query, Key, Value Linear Projections",
            "layer_type": "3x nn.Linear(128, 128)",
            "input_shape": "(batch_size, seq_len, 128)",
            "output_shape": "(batch_size, num_heads=4, seq_len, d_k=32)",
            "parameters": 128 * 128 * 3 + 128 * 3,
            "math_formula": "Q = X W_q, \\quad K = X W_k, \\quad V = X W_v",
            "explanation": "Projects inputs into query, key, and value subspaces for 4 parallel attention heads, each with d_k = 32 dimensions.",
            "pytorch_code": "q = self.w_q(x).view(B, T, 4, 32).transpose(1, 2)"
        },
        {
            "id": "scaled_dot_product_attention",
            "name": "Scaled Dot-Product Self-Attention",
            "layer_type": "Manual Tensor Operations",
            "input_shape": "Q, K, V: (batch_size, 4, seq_len, 32)",
            "output_shape": "(batch_size, 4, seq_len, 32)",
            "parameters": 0,
            "math_formula": "\\text{Attention}(Q, K, V) = \\text{softmax}\\left( \\frac{Q K^T}{\\sqrt{d_k}} + M \\right) V",
            "explanation": "Computes affinity scores between tokens. Scaled by 1/sqrt(32) to prevent the softmax gradient from vanishing at large dot product magnitudes.",
            "pytorch_code": "scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(32))"
        },
        {
            "id": "causal_masking",
            "name": "Causal Attention Mask",
            "layer_type": "torch.triu(..., diagonal=1) == -inf",
            "input_shape": "(seq_len, seq_len)",
            "output_shape": "(1, 1, seq_len, seq_len)",
            "parameters": 0,
            "math_formula": "M_{i,j} = \\begin{cases} 0 & \\text{if } j \\le i \\\\ -\\infty & \\text{if } j > i \\end{cases}",
            "explanation": "Ensures the model cannot look into the future during autoregressive next-token prediction. Position i can only attend to positions j <= i.",
            "pytorch_code": "mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)\nscores = scores + mask"
        },
        {
            "id": "feed_forward",
            "name": "Position-Wise Feed-Forward Network",
            "layer_type": "Linear(128, 512) -> GELU -> Linear(512, 128)",
            "input_shape": "(batch_size, seq_len, 128)",
            "output_shape": "(batch_size, seq_len, 128)",
            "parameters": (128 * 512 + 512) + (512 * 128 + 128),
            "math_formula": "\\text{FFN}(x) = \\text{GELU}(x W_1 + b_1) W_2 + b_2",
            "explanation": "Applied to each position separately and identically. Expands dimension to 512 with GELU non-linearity to learn rich non-linear token representations.",
            "pytorch_code": "ffn_out = self.linear2(F.gelu(self.linear1(x)))"
        },
        {
            "id": "residual_and_norm",
            "name": "Residual Connections & Layer Normalization",
            "layer_type": "nn.LayerNorm(128) + Residual Additions",
            "input_shape": "(batch_size, seq_len, 128)",
            "output_shape": "(batch_size, seq_len, 128)",
            "parameters": (128 * 2) * 2,
            "math_formula": "x = x + \\text{SubLayer}(\\text{LayerNorm}(x))",
            "explanation": "Pre-LayerNorm stabilizes deep gradient flow. Residual connections create highway paths allowing gradients to backpropagate unimpeded.",
            "pytorch_code": "x = x + self.attn(self.ln_1(x))\nx = x + self.ffn(self.ln_2(x))"
        },
        {
            "id": "lm_head",
            "name": "Linear Output Projection (LM Head)",
            "layer_type": "nn.Linear(128, vocab_size)",
            "input_shape": "(batch_size, seq_len, 128)",
            "output_shape": "(batch_size, seq_len, vocab_size)",
            "parameters": 128 * param_info["vocab_size"],
            "math_formula": "\\text{Logits} = x \\cdot W_{\\text{vocab}}^T",
            "explanation": "Projects the final 128-dimensional contextual token representation into unnormalized log-probabilities over every word in the vocabulary.",
            "pytorch_code": "logits = self.lm_head(self.ln_f(x))"
        }
    ]

    return {
        "parameters": param_info,
        "config": trainer_manager.model_config,
        "components": components
    }


@app.post("/api/train/start")
def start_model_training(req: TrainStartRequest) -> Dict[str, Any]:
    """Launch model training with specified hyperparameters."""
    if trainer_manager.is_training:
        raise HTTPException(status_code=400, detail="Training is already in progress")

    success = trainer_manager.start_training(
        epochs=req.epochs,
        batch_size=req.batch_size,
        learning_rate=req.learning_rate,
        weight_decay=req.weight_decay
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start training thread")

    return {
        "message": f"Training initiated for {req.epochs} epochs with batch size {req.batch_size}.",
        "hyperparameters": req.dict()
    }


@app.post("/api/train/stop")
def stop_model_training() -> Dict[str, Any]:
    """Gracefully request training loop to stop."""
    if not trainer_manager.is_training:
        return {"message": "No active training process to stop"}
    trainer_manager.stop_training()
    return {"message": "Stop signal sent to training worker thread"}


@app.get("/api/train/status")
def get_training_status() -> Dict[str, Any]:
    """Live polling endpoint for training progress, loss curves, and history."""
    return trainer_manager.training_state


@app.post("/api/evaluate")
def run_model_evaluation() -> Dict[str, Any]:
    """Calculate actual held-out test split results: Cross-Entropy Loss, Perplexity, Top-1/3/5 accuracy."""
    try:
        eval_result = trainer_manager.evaluate_test_split()
        return eval_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/next-token")
def predict_next_token(req: PredictNextTokenRequest) -> Dict[str, Any]:
    """Predict next-token candidate distribution and return attention map."""
    try:
        result = trainer_manager.predict_next_token(
            prompt=req.prompt,
            temperature=req.temperature,
            top_k=req.top_k,
            greedy=req.greedy
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
def generate_text_sequence(req: GenerateTextRequest) -> Dict[str, Any]:
    """Autoregressively generate next tokens with full step history."""
    try:
        result = trainer_manager.generate_text(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/attention/inspect")
def inspect_attention_map(req: InspectAttentionRequest) -> Dict[str, Any]:
    """Inspect attention matrices across all layers and heads for the input sequence."""
    try:
        result = trainer_manager.inspect_self_attention(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
