"""Training, Evaluation, and Inference Engine for Transformer Language Model.

Provides:
- Dataset creation with sliding window input/target sequences
- Train/Validation/Test splitting
- Training loop with Adam optimizer and CrossEntropyLoss
- Accurate metrics computation: Cross-Entropy Loss, Perplexity, Top-1/Top-3/Top-5 Accuracy
- Live training state tracking for real-time frontend dashboard
- Next-token prediction and autoregressive text generation
- Attention visualization inspection data extraction
"""

import copy
import json
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from backend.models.transformer import TransformerLM
from backend.tokenizer.tokenizer import SimpleTokenizer


class SequenceDataset(Dataset):
    """Dataset producing (input_ids, target_ids) pairs."""

    def __init__(self, sequences: List[Tuple[List[int], List[int]]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class TrainingManager:
    """Manages model lifecycle, training jobs, evaluation, and inference."""

    def __init__(self, outputs_dir: str = "backend/outputs"):
        self.outputs_dir = outputs_dir
        os.makedirs(outputs_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.tokenizer: Optional[SimpleTokenizer] = None
        self.model: Optional[TransformerLM] = None
        self.model_config: Dict[str, Any] = {
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 2,
            "d_ff": 512,
            "max_seq_len": 32,
            "dropout": 0.1
        }

        # Dataset storage
        self.dataset_splits: Dict[str, List[Tuple[List[int], List[int]]]] = {
            "train": [],
            "val": [],
            "test": []
        }
        self.raw_corpus_text: str = ""
        self.active_corpus_id: str = "combined"

        # Thread-safe training state
        self._lock = threading.Lock()
        self.is_training: bool = False
        self.stop_requested: bool = False
        self.training_thread: Optional[threading.Thread] = None

        self.training_state: Dict[str, Any] = {
            "status": "idle",  # idle, training, completed, stopped, error
            "current_epoch": 0,
            "total_epochs": 0,
            "current_step": 0,
            "total_steps": 0,
            "progress_percent": 0.0,
            "current_train_loss": 0.0,
            "current_val_loss": 0.0,
            "history": [],
            "best_val_loss": float("inf"),
            "best_epoch": 0,
            "elapsed_seconds": 0.0,
            "message": "Model ready for training"
        }

        # Held-out evaluation metrics
        self.latest_evaluation: Optional[Dict[str, Any]] = None

    def initialize_with_corpus(
        self,
        corpus_text: str,
        corpus_id: str = "combined",
        seq_len: int = 32,
        stride: int = 2,
        vocab_size_limit: int = 1200
    ) -> Dict[str, Any]:
        """Tokenize corpus, build vocabulary, create train/val/test splits, and initialize model."""
        with self._lock:
            self.raw_corpus_text = corpus_text
            self.active_corpus_id = corpus_id

            # 1. Build Tokenizer
            self.tokenizer = SimpleTokenizer(lowercase=True, min_freq=1, max_vocab_size=vocab_size_limit)
            self.tokenizer.build_vocab(corpus_text)

            # 2. Encode corpus to tokens
            token_ids = self.tokenizer.encode(corpus_text)

            # 3. Create sliding window sequences: x = [w_i ... w_{i+seq_len-1}], y = [w_{i+1} ... w_{i+seq_len}]
            sequences: List[Tuple[List[int], List[int]]] = []
            if len(token_ids) > seq_len:
                for i in range(0, len(token_ids) - seq_len, stride):
                    x = token_ids[i : i + seq_len]
                    y = token_ids[i + 1 : i + seq_len + 1]
                    sequences.append((x, y))
            else:
                # If short, pad to seq_len
                pad_len = seq_len - len(token_ids) + 1
                padded = token_ids + [self.tokenizer.PAD_ID] * max(0, pad_len)
                x = padded[:seq_len]
                y = padded[1 : seq_len + 1]
                sequences.append((x, y))

            # 4. Split into Train (80%), Val (10%), Test (10%)
            total_seqs = len(sequences)
            train_end = max(1, int(total_seqs * 0.8))
            val_end = max(train_end + 1, int(total_seqs * 0.9))

            self.dataset_splits["train"] = sequences[:train_end]
            self.dataset_splits["val"] = sequences[train_end:val_end]
            self.dataset_splits["test"] = sequences[val_end:]
            if not self.dataset_splits["test"]:
                # Ensure test has at least some sequences for evaluation
                self.dataset_splits["test"] = sequences[-max(1, int(total_seqs * 0.1)):]

            # 5. Initialize PyTorch Model
            self.model_config["vocab_size"] = self.tokenizer.vocab_size
            self.model_config["max_seq_len"] = seq_len
            self.model = TransformerLM(**self.model_config).to(self.device)

            # Save initial vocab
            self.tokenizer.save_vocab(os.path.join(self.outputs_dir, "vocab.json"))
            with open(os.path.join(self.outputs_dir, "config.json"), "w") as f:
                json.dump(self.model_config, f, indent=2)

            return {
                "vocab_size": self.tokenizer.vocab_size,
                "total_tokens": len(token_ids),
                "total_sequences": total_seqs,
                "train_sequences": len(self.dataset_splits["train"]),
                "val_sequences": len(self.dataset_splits["val"]),
                "test_sequences": len(self.dataset_splits["test"]),
                "model_parameters": self.model.count_parameters()
            }

    def start_training(
        self,
        epochs: int = 15,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        on_epoch_end_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> bool:
        """Start model training in a background worker thread."""
        with self._lock:
            if self.is_training:
                return False
            self.is_training = True
            self.stop_requested = False

        self.training_thread = threading.Thread(
            target=self._run_training_loop,
            args=(epochs, batch_size, learning_rate, weight_decay, on_epoch_end_callback),
            daemon=True
        )
        self.training_thread.start()
        return True

    def stop_training(self) -> None:
        """Signal training loop to stop gracefully after current batch."""
        self.stop_requested = True

    def _run_training_loop(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> None:
        try:
            train_ds = SequenceDataset(self.dataset_splits["train"])
            val_ds = SequenceDataset(self.dataset_splits["val"])

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.PAD_ID)

            total_steps = epochs * len(train_loader)
            start_time = time.time()

            with self._lock:
                self.training_state.update({
                    "status": "training",
                    "current_epoch": 0,
                    "total_epochs": epochs,
                    "current_step": 0,
                    "total_steps": total_steps,
                    "progress_percent": 0.0,
                    "history": [],
                    "best_val_loss": float("inf"),
                    "best_epoch": 0,
                    "message": "Training in progress..."
                })

            best_model_weights = None
            step_count = 0

            for epoch in range(1, epochs + 1):
                if self.stop_requested:
                    break

                self.model.train()
                epoch_train_loss = 0.0
                train_correct_top1 = 0
                train_total_tokens = 0

                for batch_idx, (x, y) in enumerate(train_loader):
                    if self.stop_requested:
                        break

                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()

                    logits = self.model(x)  # (B, T, V)
                    loss = criterion(logits.view(-1, self.model.vocab_size), y.view(-1))
                    loss.backward()

                    # Gradient clipping for training stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_train_loss += loss.item()
                    step_count += 1

                    # Track top 1 token accuracy
                    preds = logits.argmax(dim=-1)
                    mask = (y != self.tokenizer.PAD_ID)
                    train_correct_top1 += (preds[mask] == y[mask]).sum().item()
                    train_total_tokens += mask.sum().item()

                    # Update step progress periodically
                    if batch_idx % max(1, len(train_loader) // 5) == 0:
                        with self._lock:
                            pct = round((step_count / max(1, total_steps)) * 100, 1)
                            self.training_state["current_step"] = step_count
                            self.training_state["progress_percent"] = pct
                            self.training_state["elapsed_seconds"] = round(time.time() - start_time, 2)

                avg_train_loss = epoch_train_loss / max(1, len(train_loader))
                train_top1_acc = (train_correct_top1 / max(1, train_total_tokens)) * 100

                # Validation phase
                val_metrics = self._evaluate_split(val_loader, criterion)
                avg_val_loss = val_metrics["loss"]
                val_ppl = val_metrics["perplexity"]
                val_top1 = val_metrics["top1_accuracy"]

                epoch_summary = {
                    "epoch": epoch,
                    "train_loss": round(avg_train_loss, 4),
                    "train_perplexity": round(math.exp(min(avg_train_loss, 20)), 2),
                    "train_top1_acc": round(train_top1_acc, 2),
                    "val_loss": round(avg_val_loss, 4),
                    "val_perplexity": round(val_ppl, 2),
                    "val_top1_acc": round(val_top1, 2),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_time": round(time.time() - start_time, 2)
                }

                # Save best checkpoint
                with self._lock:
                    self.training_state["history"].append(epoch_summary)
                    self.training_state["current_epoch"] = epoch
                    self.training_state["current_train_loss"] = round(avg_train_loss, 4)
                    self.training_state["current_val_loss"] = round(avg_val_loss, 4)

                    if avg_val_loss < self.training_state["best_val_loss"]:
                        self.training_state["best_val_loss"] = round(avg_val_loss, 4)
                        self.training_state["best_epoch"] = epoch
                        best_model_weights = copy.deepcopy(self.model.state_dict())
                        # Save best checkpoint to disk
                        torch.save(best_model_weights, os.path.join(self.outputs_dir, "best_model.pt"))

                if callback:
                    callback(epoch_summary)

            # Finalize training
            if best_model_weights is not None:
                self.model.load_state_dict(best_model_weights)

            # Automatically compute test evaluation on held-out test split
            test_eval = self.evaluate_test_split()

            with self._lock:
                status_str = "stopped" if self.stop_requested else "completed"
                self.training_state["status"] = status_str
                self.training_state["progress_percent"] = 100.0
                self.training_state["elapsed_seconds"] = round(time.time() - start_time, 2)
                self.training_state["message"] = (
                    f"Training {status_str}! Best Val Loss: {self.training_state['best_val_loss']} (Epoch {self.training_state['best_epoch']}). "
                    f"Test Perplexity: {test_eval['perplexity']}."
                )

                # Persist metrics to file
                with open(os.path.join(self.outputs_dir, "metrics.json"), "w") as f:
                    json.dump({
                        "training_state": self.training_state,
                        "latest_evaluation": self.latest_evaluation
                    }, f, indent=2)

        except Exception as e:
            with self._lock:
                self.training_state["status"] = "error"
                self.training_state["message"] = f"Error during training: {str(e)}"
        finally:
            with self._lock:
                self.is_training = False

    def _evaluate_split(self, data_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Compute exact evaluation metrics over a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct_top1 = 0
        correct_top3 = 0
        correct_top5 = 0

        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)  # (B, T, V)

                loss = criterion(logits.view(-1, self.model.vocab_size), y.view(-1))
                total_loss += loss.item() * x.size(0)

                # Accuracy metrics on valid non-pad tokens
                mask = (y != self.tokenizer.PAD_ID)
                target_flat = y[mask]
                logits_flat = logits[mask]

                if target_flat.numel() > 0:
                    # Top-1
                    top1_preds = logits_flat.argmax(dim=-1)
                    correct_top1 += (top1_preds == target_flat).sum().item()

                    # Top-3 & Top-5
                    _, top5_preds = logits_flat.topk(min(5, logits_flat.size(-1)), dim=-1)
                    correct_top5 += (top5_preds == target_flat.unsqueeze(1)).any(dim=1).sum().item()

                    _, top3_preds = logits_flat.topk(min(3, logits_flat.size(-1)), dim=-1)
                    correct_top3 += (top3_preds == target_flat.unsqueeze(1)).any(dim=1).sum().item()

                    total_tokens += target_flat.numel()

        avg_loss = total_loss / max(1, len(data_loader.dataset))
        ppl = math.exp(min(avg_loss, 20))
        top1_acc = (correct_top1 / max(1, total_tokens)) * 100
        top3_acc = (correct_top3 / max(1, total_tokens)) * 100
        top5_acc = (correct_top5 / max(1, total_tokens)) * 100

        return {
            "loss": avg_loss,
            "perplexity": ppl,
            "top1_accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "top5_accuracy": top5_acc,
            "total_tokens_evaluated": total_tokens
        }

    def evaluate_test_split(self) -> Dict[str, Any]:
        """Run full evaluation on held-out test split and collect prediction examples."""
        if not self.dataset_splits["test"]:
            return {"error": "No test split available"}

        test_ds = SequenceDataset(self.dataset_splits["test"])
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.PAD_ID)

        metrics = self._evaluate_split(test_loader, criterion)

        # Generate sample inspection predictions on test sequences
        self.model.eval()
        prediction_examples = []
        with torch.no_grad():
            for i in range(min(8, len(self.dataset_splits["test"]))):
                x_ids, y_ids = self.dataset_splits["test"][i]
                x_tensor = torch.tensor([x_ids], dtype=torch.long, device=self.device)
                logits = self.model(x_tensor)  # (1, T, V)

                # Look at the last token position
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                top5_probs, top5_ids = torch.topk(probs, 5)

                expected_target_id = y_ids[-1]
                expected_target_token = self.tokenizer.id2token.get(expected_target_id, self.tokenizer.UNK_TOKEN)

                top5_candidates = [
                    {
                        "token": self.tokenizer.id2token.get(idx.item(), self.tokenizer.UNK_TOKEN),
                        "id": idx.item(),
                        "probability": round(prob.item() * 100, 2)
                    }
                    for prob, idx in zip(top5_probs, top5_ids)
                ]

                context_tokens = [self.tokenizer.id2token.get(tid, self.tokenizer.UNK_TOKEN) for tid in x_ids]
                prediction_examples.append({
                    "sample_index": i + 1,
                    "context_preview": " ".join(context_tokens[-8:]),
                    "expected_target": expected_target_token,
                    "top1_prediction": top5_candidates[0]["token"],
                    "is_correct": top5_candidates[0]["id"] == expected_target_id,
                    "in_top5": any(c["id"] == expected_target_id for c in top5_candidates),
                    "top5_candidates": top5_candidates
                })

        eval_result = {
            "cross_entropy_loss": round(metrics["loss"], 4),
            "perplexity": round(metrics["perplexity"], 2),
            "top1_accuracy": round(metrics["top1_accuracy"], 2),
            "top3_accuracy": round(metrics["top3_accuracy"], 2),
            "top5_accuracy": round(metrics["top5_accuracy"], 2),
            "total_tokens_evaluated": metrics["total_tokens_evaluated"],
            "prediction_examples": prediction_examples,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        self.latest_evaluation = eval_result
        return eval_result

    def predict_next_token(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_k: int = 5,
        greedy: bool = False
    ) -> Dict[str, Any]:
        """Predict the next token candidates and probabilities for a given prompt."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be initialized")

        self.model.eval()
        tokens = self.tokenizer.tokenize_raw(prompt)
        if not tokens:
            tokens = ["the"]

        # Truncate to maximum sequence length
        if len(tokens) > self.model_config["max_seq_len"]:
            tokens = tokens[-self.model_config["max_seq_len"]:]

        input_ids = [self.tokenizer.token2id.get(t, self.tokenizer.UNK_ID) for t in tokens]
        x_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits, all_attentions = self.model(x_tensor, return_attentions=True)
            # Logits for the final sequence token
            next_logits = logits[0, -1, :]  # (vocab_size,)

            if greedy or temperature <= 0.01:
                probs = F.softmax(next_logits, dim=-1)
                selected_id = torch.argmax(next_logits).item()
            else:
                scaled_logits = next_logits / max(0.01, temperature)
                probs = F.softmax(scaled_logits, dim=-1)

                # Filter top_k
                if top_k > 0:
                    topk_vals, topk_indices = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    filtered_logits = torch.full_like(scaled_logits, float("-inf"))
                    filtered_logits[topk_indices] = topk_vals
                    sampling_probs = F.softmax(filtered_logits, dim=-1)
                    selected_id = torch.multinomial(sampling_probs, num_samples=1).item()
                else:
                    selected_id = torch.multinomial(probs, num_samples=1).item()

            # Retrieve top 5 candidate tokens
            top5_probs, top5_indices = torch.topk(probs, 5)
            candidates = []
            for rank, (prob, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                tok = self.tokenizer.id2token.get(idx.item(), self.tokenizer.UNK_TOKEN)
                candidates.append({
                    "rank": rank,
                    "token": tok,
                    "token_id": idx.item(),
                    "probability": round(prob.item() * 100, 2),
                    "raw_logit": round(next_logits[idx].item(), 3),
                    "is_selected": idx.item() == selected_id
                })

            selected_token = self.tokenizer.id2token.get(selected_id, self.tokenizer.UNK_TOKEN)

            # Format attention maps for visualization
            # all_attentions is list of (1, H, T, T)
            attention_summary = []
            for layer_idx, layer_attn in enumerate(all_attentions):
                layer_matrix = layer_attn[0].cpu().numpy().tolist()  # (H, T, T)
                attention_summary.append({
                    "layer": layer_idx + 1,
                    "heads": layer_matrix
                })

            return {
                "prompt": prompt,
                "input_tokens": tokens,
                "input_ids": input_ids,
                "selected_token": selected_token,
                "selected_token_id": selected_id,
                "top_candidates": candidates,
                "temperature": temperature,
                "sampling_mode": "greedy" if greedy else f"temperature ({temperature})",
                "attentions": attention_summary
            }

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Autoregressively generate subsequent tokens token-by-token."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be initialized")

        self.model.eval()
        tokens = self.tokenizer.tokenize_raw(prompt)
        if not tokens:
            tokens = ["once", "upon", "a", "time"]

        generated_history: List[Dict[str, Any]] = []
        current_tokens = list(tokens)

        with torch.no_grad():
            for step in range(1, max_new_tokens + 1):
                # Window to max_seq_len
                window = current_tokens[-self.model_config["max_seq_len"]:]
                input_ids = [self.tokenizer.token2id.get(t, self.tokenizer.UNK_ID) for t in window]
                x_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

                logits = self.model(x_tensor)
                next_logits = logits[0, -1, :]

                # Apply temperature
                scaled_logits = next_logits / max(0.01, temperature)
                probs = F.softmax(scaled_logits, dim=-1)

                # Top-K filtering
                if top_k > 0:
                    topk_vals, topk_indices = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    filtered_logits = torch.full_like(scaled_logits, float("-inf"))
                    filtered_logits[topk_indices] = topk_vals
                    filtered_probs = F.softmax(filtered_logits, dim=-1)
                    next_token_id = torch.multinomial(filtered_probs, num_samples=1).item()
                else:
                    next_token_id = torch.multinomial(probs, num_samples=1).item()

                next_token = self.tokenizer.id2token.get(next_token_id, self.tokenizer.UNK_TOKEN)
                current_tokens.append(next_token)

                # Alternatives considered
                top3_probs, top3_indices = torch.topk(probs, 3)
                alternatives = [
                    {
                        "token": self.tokenizer.id2token.get(idx.item(), self.tokenizer.UNK_TOKEN),
                        "prob": round(p.item() * 100, 1)
                    }
                    for p, idx in zip(top3_probs, top3_indices)
                ]

                token_prob = probs[next_token_id].item()
                generated_history.append({
                    "step": step,
                    "token": next_token,
                    "token_id": next_token_id,
                    "probability": round(token_prob * 100, 2),
                    "alternatives": alternatives
                })

                if next_token == self.tokenizer.EOS_TOKEN:
                    break

        full_text = self.tokenizer.decode(
            [self.tokenizer.token2id.get(t, self.tokenizer.UNK_ID) for t in current_tokens],
            skip_special_tokens=True
        )

        return {
            "prompt": prompt,
            "generated_text": full_text,
            "tokens_added": len(generated_history),
            "history": generated_history,
            "hyperparameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k
            }
        }

    def inspect_self_attention(self, text: str) -> Dict[str, Any]:
        """Compute detailed attention matrices, query-key dot products, and causal mask info."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be initialized")

        self.model.eval()
        tokens = self.tokenizer.tokenize_raw(text)
        if not tokens:
            tokens = ["attention", "is", "all", "you", "need"]

        tokens = tokens[:self.model_config["max_seq_len"]]
        input_ids = [self.tokenizer.token2id.get(t, self.tokenizer.UNK_ID) for t in tokens]
        x_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            _, all_attentions = self.model(x_tensor, return_attentions=True)

        seq_len = len(tokens)
        layers_data = []

        for layer_idx, layer_attn in enumerate(all_attentions):
            # layer_attn: (1, num_heads, seq_len, seq_len)
            heads_matrices = layer_attn[0].cpu().numpy().tolist()

            # Compute average attention across all heads
            avg_matrix = []
            for i in range(seq_len):
                row = []
                for j in range(seq_len):
                    val = sum(heads_matrices[h][i][j] for h in range(len(heads_matrices))) / len(heads_matrices)
                    row.append(round(val, 4))
                avg_matrix.append(row)

            layers_data.append({
                "layer_idx": layer_idx + 1,
                "heads": [
                    [[round(v, 4) for v in row] for row in head]
                    for head in heads_matrices
                ],
                "average_head": avg_matrix
            })

        # Causal mask representation for UI
        causal_mask_matrix = [
            [1 if j <= i else 0 for j in range(seq_len)]
            for i in range(seq_len)
        ]

        return {
            "tokens": tokens,
            "token_ids": input_ids,
            "seq_len": seq_len,
            "num_layers": len(layers_data),
            "num_heads": self.model_config["num_heads"],
            "d_k": self.model_config["d_model"] // self.model_config["num_heads"],
            "layers": layers_data,
            "causal_mask": causal_mask_matrix,
            "math_explanation": {
                "formula": "Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) + M ) * V",
                "scale_factor": round(1.0 / math.sqrt(self.model_config["d_model"] // self.model_config["num_heads"]), 4),
                "causal_property": "Causal mask M sets positions j > i to -inf, forcing softmax to yield 0 attention weight on future tokens."
            }
        }
