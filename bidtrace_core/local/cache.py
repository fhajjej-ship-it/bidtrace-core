# SPDX-License-Identifier: MPL-2.0
"""Optional bounded, RAM-only exact-prefix caching for serial text-only trials.

The caller supplies a verified shared token prefix and the exact complete input
token IDs for each request. Keep one instance per loaded model/source group.
No prompt, grammar, label, reasoning budget, or generation output is changed.
Hybrid Qwen state must be snapshotted at the shared boundary, never rewound.
Cache admission may decline under memory pressure; inspect cached_tokens/stats.
Numerical output parity and actual speed still require a model trial.
"""
from __future__ import annotations


class LocalPrefixCache:
    """One bounded checkpoint; no model load or disk writes.

    Trials retain the 128 MiB default. The service may explicitly admit a larger
    source checkpoint, capped at 768 MiB, without changing the token boundary.
    """

    def __init__(self, shared_prefix_ids, *, memory_max_gb=0.125):
        if (type(memory_max_gb) not in (int, float)
                or not 0 < memory_max_gb <= 0.75):
            raise ValueError("Prefix cache budget must be positive and at most 0.75 GiB")
        from mlx_vlm.apc import APCManager

        if (not isinstance(shared_prefix_ids, (list, tuple))
                or len(shared_prefix_ids) < 16
                or any(type(token) is not int or token < 0 for token in shared_prefix_ids)):
            raise ValueError("Shared prefix requires at least 16 integer token IDs")
        self.shared_prefix_ids = tuple(shared_prefix_ids)
        self.memory_max_gb = memory_max_gb
        self.manager = APCManager(num_blocks=1, block_size=16, disk=None, overrides={
            "checkpoint_entries": 1, "checkpoint_interval_tokens": 0,
            "checkpoint_guard_tokens": 1, "memory_max_gb": memory_max_gb})
        if len(self.shared_prefix_ids) < self.manager.exact_cache_min_tokens:
            raise ValueError("Shared prefix is shorter than the installed APC minimum")

    def generation_kwargs(self, prompt_ids, *, has_media=False):
        """Return generate kwargs only for an exact matching text-only prefix.

        Call immediately before serial generate; this updates its checkpoint
        boundary. The IDs must match the model's actual prepared text input.
        """
        if has_media:
            return {}
        if (not isinstance(prompt_ids, (list, tuple))
                or any(type(token) is not int or token < 0 for token in prompt_ids)):
            raise ValueError("Prompt requires integer token IDs")
        length = len(self.shared_prefix_ids)
        if len(prompt_ids) <= length or tuple(prompt_ids[:length]) != self.shared_prefix_ids:
            return {}
        # Native APC computes checkpoint_len = prompt length - guard tokens.
        self.manager.exact_cache_guard_tokens = len(prompt_ids) - length
        return {"apc_manager": self.manager}

    def stats(self):
        return self.manager.stats_snapshot()

    def clear(self):
        self.manager.clear()
