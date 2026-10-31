"""Manual Transformer Language Model implementation in PyTorch.

Built from scratch using basic tensor operations for:
- Token & Positional Embeddings (Sinusoidal or Learnable)
- Manual Multi-Head Self-Attention with Causal Masking
- Feed-Forward Networks with GELU activation
- Residual Connections & Layer Normalization
- Linear Output Projection for Next-Token Prediction
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding as introduced in 'Attention Is All You Need'."""

    def __init__(self, d_model: int, max_seq_len: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Compute sinusoidal positional encodings once in log space
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to token embeddings.
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ManualMultiHeadSelfAttention(nn.Module):
    """Manual Multi-Head Self-Attention with Causal Masking using raw PyTorch tensor ops."""

    def __init__(self, d_model: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 128 // 4 = 32

        # Distinct Q, K, V linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)

        # Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute multi-head self-attention with strict causal masking.
        Args:
            x: (batch_size, seq_len, d_model)
            return_attention_weights: bool whether to return raw attention matrix
        Returns:
            output: (batch_size, seq_len, d_model)
            (optional) attn_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Step 1: Project inputs to Query, Key, Value
        q = self.w_q(x)  # (B, T, D)
        k = self.w_k(x)  # (B, T, D)
        v = self.w_v(x)  # (B, T, D)

        # Step 2: Reshape for multi-head attention: (B, T, H, d_k) -> (B, H, T, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Scaled Dot-Product: (B, H, T, d_k) @ (B, H, d_k, T) -> (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Step 4: Causal Attention Masking (Autoregressive constraint)
        # Upper triangular indices (excluding diagonal) are masked to -inf
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1
        )
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Step 5: Softmax normalization over the last dimension
        attn_weights = torch.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn_weights)

        # Step 6: Attention weighted values: (B, H, T, T) @ (B, H, T, d_k) -> (B, H, T, d_k)
        context = torch.matmul(attn_dropped, v)

        # Step 7: Concatenate heads back into (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Step 8: Final linear output projection
        output = self.w_o(context)

        if return_attention_weights:
            return output, attn_weights
        return output


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.linear2(self.act(self.linear1(x))))


class TransformerBlock(nn.Module):
    """A single Transformer Decoder Block with Pre-LayerNorm and Residual connections."""

    def __init__(self, d_model: int = 128, num_heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = ManualMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Pre-LN Self-Attention with Residual Connection
        normed_1 = self.ln_1(x)
        if return_attention_weights:
            attn_out, weights = self.attn(normed_1, return_attention_weights=True)
            x = x + attn_out
            # Pre-LN Feed-Forward with Residual Connection
            x = x + self.ffn(self.ln_2(x))
            return x, weights
        else:
            x = x + self.attn(normed_1)
            x = x + self.ffn(self.ln_2(x))
            return x


class TransformerLM(nn.Module):
    """Full Transformer Language Model for Next-Token Prediction.

    Configuration:
    - Embedding Dimension: 128
    - Attention Heads: 4
    - Transformer Blocks: 2
    - Maximum Sequence Length: 32
    - Output: Logits over vocabulary size
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 512,
        max_seq_len: int = 32,
        dropout: float = 0.1,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        self.emb_dropout = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final Layer Normalization
        self.ln_f = nn.LayerNorm(d_model)

        # Linear Output Projection to Vocabulary Logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017) boosts parameter efficiency
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights with standard normal (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_attentions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass.
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
            return_attentions: bool whether to return attention matrices
        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, vocab_size)
            (optional) all_attentions: List of (batch_size, num_heads, seq_len, seq_len)
        """
        _, seq_len = input_ids.size()
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"

        # 1. Token Embeddings scaled by sqrt(d_model) + Positional Encoding
        tok_emb = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.emb_dropout(self.pos_encoding(tok_emb))

        all_attentions = []
        for block in self.blocks:
            if return_attentions:
                x, attn_weights = block(x, return_attention_weights=True)
                all_attentions.append(attn_weights)
            else:
                x = block(x)

        # 2. Final LayerNorm
        x = self.ln_f(x)

        # 3. Output Logits
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        if return_attentions:
            return logits, all_attentions
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Detailed parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embeddings = sum(p.numel() for p in self.token_embedding.parameters())

        blocks_params = []
        for i, block in enumerate(self.blocks):
            b_total = sum(p.numel() for p in block.parameters())
            b_attn = sum(p.numel() for p in block.attn.parameters())
            b_ffn = sum(p.numel() for p in block.ffn.parameters())
            blocks_params.append({
                "block_idx": i,
                "total": b_total,
                "attention": b_attn,
                "feed_forward": b_ffn
            })

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "embedding_parameters": embeddings,
            "blocks": blocks_params,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_k": self.d_model // self.num_heads,
            "num_layers": self.num_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size
        }
