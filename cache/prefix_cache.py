"""Prefix-based KV cache for reusing computation across requests with shared prefixes."""

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from transformers import DynamicCache


@dataclass
class PrefixCacheEntry:
    prefix_text: str
    prefix_tokens: torch.Tensor          # shape: [1, T]
    past_key_values: Tuple                # HF-style tuple (stored on CPU)
    last_logits: torch.Tensor             # shape: [1, vocab]
    token_count: int


class PrefixCache:
    def __init__(self, min_tokens: int = 20):
        self.min_tokens = min_tokens
        self._store: Dict[str, PrefixCacheEntry] = {}
        self.hits = 0
        self.misses = 0

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def add(
        self,
        prefix_text: str,
        prefix_tokens: torch.Tensor,
        past_key_values,
        last_logits: torch.Tensor,
    ):
        token_count = prefix_tokens.shape[1]
        if token_count < self.min_tokens:
            return

        # Convert DynamicCache to tuple for storage
        if hasattr(past_key_values, 'to_legacy_cache'):
            pkv_tuple = past_key_values.to_legacy_cache()
        else:
            pkv_tuple = past_key_values

        key = self._hash(prefix_text)
        self._store[key] = PrefixCacheEntry(
            prefix_text=prefix_text,
            prefix_tokens=prefix_tokens.cpu(),
            past_key_values=self._to_cpu(pkv_tuple),
            last_logits=last_logits.cpu(),
            token_count=token_count,
        )

    def find_longest_prefix(
        self,
        prompt: str,
        device: torch.device,
    ) -> Optional[PrefixCacheEntry]:
        """
        Longest-prefix match over stored prefixes.
        Returns entry with DynamicCache format for transformers 4.57.3+
        """
        best = None
        best_len = -1

        for entry in self._store.values():
            if prompt.startswith(entry.prefix_text):
                if entry.token_count > best_len:
                    best = entry
                    best_len = entry.token_count

        if best is None:
            self.misses += 1
            return None

        self.hits += 1
        
        # Convert tuple back to DynamicCache for transformers
        pkv_on_device = self._to_device(best.past_key_values, device)
        dynamic_cache = DynamicCache.from_legacy_cache(pkv_on_device)
        
        return PrefixCacheEntry(
            prefix_text=best.prefix_text,
            prefix_tokens=best.prefix_tokens.to(device),
            past_key_values=dynamic_cache,
            last_logits=best.last_logits.to(device),
            token_count=best.token_count,
        )
    
    def _to_cpu(self, pkv):
        return tuple(
            tuple(t.cpu() for t in layer)
            for layer in pkv
        )

    def _to_device(self, pkv, device):
        return tuple(
            tuple(t.to(device) for t in layer)
            for layer in pkv
        )

    def clear(self):
        """Clear all cached entries."""
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "size": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_requests": total
        }
