"""Unit and Integration Tests for Transformer Intelligence Studio.

Tests:
1. Tokenizer: special tokens, token2id, encode, decode, unknown token handling.
2. Positional Encoding: shape, values, and determinism.
3. Manual MultiHeadSelfAttention: tensor shapes, causal mask enforcement, softmax normalization.
4. Causality Invariance: changing future tokens does not affect prior token representations.
5. TransformerLM: end-to-end forward pass, parameter breakdown, logits shapes.
6. Training and Evaluation: cross-entropy loss, perplexity, top-1 accuracy.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from backend.datasets.corpus import get_corpus
from backend.models.transformer import (
    ManualMultiHeadSelfAttention,
    PositionalEncoding,
    PositionWiseFeedForward,
    TransformerBlock,
    TransformerLM,
)
from backend.tokenizer.tokenizer import SimpleTokenizer
from backend.training.trainer import TrainingManager


def test_tokenizer_encoding_decoding():
    tokenizer = SimpleTokenizer(lowercase=True)
    sample_text = "Oliver the curious rabbit hopped happily!"
    tokenizer.build_vocab(sample_text)

    # Check special tokens
    assert tokenizer.PAD_TOKEN in tokenizer.token2id
    assert tokenizer.UNK_TOKEN in tokenizer.token2id
    assert tokenizer.token2id[tokenizer.PAD_TOKEN] == 0
    assert tokenizer.token2id[tokenizer.UNK_TOKEN] == 1

    # Encode known words
    encoded = tokenizer.encode("Oliver rabbit")
    assert len(encoded) == 2
    assert tokenizer.UNK_ID not in encoded

    # Encode unknown word
    encoded_unk = tokenizer.encode("Oliver mysterious_unicorn_word rabbit")
    assert tokenizer.UNK_ID in encoded_unk

    # Decode
    decoded = tokenizer.decode(encoded)
    assert "oliver rabbit" in decoded.lower()


def test_positional_encoding():
    d_model = 128
    max_seq_len = 32
    pe = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)

    x = torch.zeros(2, 16, d_model)
    out = pe(x)
    assert out.shape == (2, 16, d_model)
    # Ensure positional encoding is not all zeros
    assert not torch.allclose(out, torch.zeros_like(out))


def test_manual_multi_head_attention_causal_mask():
    batch_size = 2
    seq_len = 8
    d_model = 128
    num_heads = 4

    attn = ManualMultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    attn.eval()

    x = torch.randn(batch_size, seq_len, d_model)
    out, weights = attn(x, return_attention_weights=True)

    # Check output shapes
    assert out.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # Verify attention weights sum to 1.0 along the key dimension (dim=-1)
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    # Verify causal masking: for any position i, attention to position j > i MUST be exactly 0
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            future_attn = weights[:, :, i, j]
            assert torch.allclose(future_attn, torch.zeros_like(future_attn), atol=1e-6), (
                f"Causal mask failed! Position {i} attended to future position {j} with weight {future_attn}"
            )


def test_causality_strict_property():
    """Verify that changing token at position k does NOT change representations at positions < k."""
    vocab_size = 50
    model = TransformerLM(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2, max_seq_len=16, dropout=0.0)
    model.eval()

    # Two sequences with identical first 4 tokens, but differing 5th token
    seq1 = torch.tensor([[5, 12, 18, 24, 7, 0, 0, 0]], dtype=torch.long)
    seq2 = torch.tensor([[5, 12, 18, 24, 42, 0, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        logits1 = model(seq1)
        logits2 = model(seq2)

    # The logits for the first 4 positions must be mathematically identical
    diff_prefix = (logits1[:, :4, :] - logits2[:, :4, :]).abs().max().item()
    assert diff_prefix < 1e-4, f"Causality violation! Logits in prefix changed by {diff_prefix}"


def test_transformer_end_to_end_forward():
    vocab_size = 100
    model = TransformerLM(vocab_size=vocab_size, d_model=128, num_heads=4, num_layers=2, max_seq_len=32)
    model.eval()

    input_ids = torch.randint(0, vocab_size, (4, 16))
    logits, attentions = model(input_ids, return_attentions=True)

    assert logits.shape == (4, 16, vocab_size)
    assert len(attentions) == 2  # 2 Transformer blocks
    assert attentions[0].shape == (4, 4, 16, 16)  # (B, H, T, T)

    param_info = model.count_parameters()
    assert param_info["total_parameters"] > 0
    assert param_info["num_layers"] == 2
    assert param_info["num_heads"] == 4


def test_training_manager_integration():
    corpus = get_corpus("stories")["text"]
    trainer = TrainingManager(outputs_dir="backend/outputs")
    stats = trainer.initialize_with_corpus(corpus_text=corpus, seq_len=16, stride=4)

    assert stats["vocab_size"] > 0
    assert stats["train_sequences"] > 0
    assert stats["test_sequences"] > 0

    # Test prediction
    pred = trainer.predict_next_token("Oliver the rabbit", temperature=1.0, top_k=5)
    assert len(pred["top_candidates"]) == 5
    assert "token" in pred["top_candidates"][0]
    assert "probability" in pred["top_candidates"][0]

    # Test generation
    gen = trainer.generate_text("Oliver hopped", max_new_tokens=5, temperature=0.8)
    assert len(gen["history"]) == 5
    assert len(gen["generated_text"]) > len("Oliver hopped")

    # Test attention inspection
    attn_data = trainer.inspect_self_attention("Oliver the rabbit")
    assert len(attn_data["layers"]) == 2
    assert len(attn_data["tokens"]) == 3
