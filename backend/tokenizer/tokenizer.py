"""Custom Tokenizer for Transformer Intelligence Studio.

Handles text preprocessing, special tokens (<pad>, <unk>, <bos>, <eos>),
token-to-id mapping, vocabulary statistics, and serialization.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union


class SimpleTokenizer:
    """Word and punctuation level tokenizer with special token support."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, lowercase: bool = True, min_freq: int = 1, max_vocab_size: Optional[int] = 1200):
        self.lowercase = lowercase
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self.is_built: bool = False

        # Initialize special tokens
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        self.token2id = {token: idx for idx, token in enumerate(self.SPECIAL_TOKENS)}
        self.id2token = {idx: token for idx, token in enumerate(self.SPECIAL_TOKENS)}

    def clean_text(self, text: str) -> str:
        """Normalize whitespace and control characters."""
        if self.lowercase:
            text = text.lower()
        # Normalize various quotes and dashes
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = text.replace("—", " - ").replace("–", " - ")
        # Collapse multiple whitespaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_raw(self, text: str) -> List[str]:
        """Split text into words and punctuation tokens."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return []
        # Pattern matches words (including apostrophe like don't) or single punctuation
        tokens = re.findall(r"\b\w+(?:'\w+)?\b|[.,!?;:\"()\-]", cleaned)
        return tokens

    def build_vocab(self, corpus_text: str) -> Dict[str, int]:
        """Build vocabulary from raw text corpus respecting min_freq and max_vocab_size."""
        self._init_special_tokens()
        tokens = self.tokenize_raw(corpus_text)
        self.word_counts = Counter(tokens)

        # Sort tokens by frequency descending, then alphabetically
        sorted_tokens = sorted(
            [token for token, count in self.word_counts.items() if count >= self.min_freq],
            key=lambda t: (-self.word_counts[t], t)
        )

        current_id = len(self.SPECIAL_TOKENS)
        for token in sorted_tokens:
            if token in self.token2id:
                continue
            if self.max_vocab_size and current_id >= self.max_vocab_size:
                break
            self.token2id[token] = current_id
            self.id2token[current_id] = token
            current_id += 1

        self.is_built = True
        return self.token2id

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Convert text into token IDs."""
        tokens = self.tokenize_raw(text)
        ids: List[int] = []

        if add_bos:
            ids.append(self.BOS_ID)

        for token in tokens:
            token_id = self.token2id.get(token, self.UNK_ID)
            ids.append(token_id)

        if add_eos:
            ids.append(self.EOS_ID)

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Convert token IDs back into human readable text."""
        tokens: List[str] = []
        for token_id in ids:
            if token_id in self.id2token:
                token = self.id2token[token_id]
                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.UNK_TOKEN)

        # Join tokens intelligently handling punctuation
        text = ""
        for i, token in enumerate(tokens):
            if i == 0 or token in {".", ",", "!", "?", ";", ":", "'", ")"}:
                text += token
            elif i > 0 and tokens[i - 1] in {"(", '"'}:
                text += token
            else:
                text += " " + token
        return text.strip()

    def get_token_details(self, text: str) -> List[Dict[str, Union[str, int, bool]]]:
        """Detailed tokenization information for UI visualization."""
        tokens = self.tokenize_raw(text)
        details = []
        for i, token in enumerate(tokens):
            token_id = self.token2id.get(token, self.UNK_ID)
            is_unk = token_id == self.UNK_ID
            freq = self.word_counts.get(token, 0)
            details.append({
                "position": i,
                "token": token,
                "id": token_id,
                "is_unk": is_unk,
                "frequency": freq
            })
        return details

    def get_vocab_stats(self) -> Dict[str, Union[int, List[Dict[str, Union[str, int]]]]]:
        """Return statistics about current vocabulary."""
        top_words = []
        # Get top 25 most frequent vocabulary words
        for token, count in self.word_counts.most_common(25):
            if token in self.token2id:
                top_words.append({
                    "token": token,
                    "id": self.token2id[token],
                    "count": count
                })

        return {
            "vocab_size": len(self.token2id),
            "special_tokens_count": len(self.SPECIAL_TOKENS),
            "total_corpus_tokens": sum(self.word_counts.values()),
            "unique_raw_tokens": len(self.word_counts),
            "top_words": top_words,
            "special_tokens": [
                {"token": tok, "id": self.token2id[tok]} for tok in self.SPECIAL_TOKENS
            ]
        }

    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary mapping and metadata to a JSON file."""
        data = {
            "lowercase": self.lowercase,
            "min_freq": self.min_freq,
            "max_vocab_size": self.max_vocab_size,
            "token2id": self.token2id,
            "word_counts": dict(self.word_counts)
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_vocab(cls, filepath: str) -> "SimpleTokenizer":
        """Load vocabulary mapping from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(
            lowercase=data.get("lowercase", True),
            min_freq=data.get("min_freq", 1),
            max_vocab_size=data.get("max_vocab_size", 1200)
        )
        tokenizer.token2id = data["token2id"]
        tokenizer.id2token = {int(v): k for k, v in data["token2id"].items()}
        tokenizer.word_counts = Counter(data.get("word_counts", {}))
        tokenizer.is_built = True
        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)
