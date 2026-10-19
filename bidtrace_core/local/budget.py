# SPDX-License-Identifier: MPL-2.0
"""Bound reasoning for the isolated MLX trial without changing the final grammar.

Use a fresh factory result per generation and a prompt prefilled with ``<think>``.
Supports installed MLX's single-sequence autoregressive calls: first call is
prefill, then exactly one sampled token per call; no speculative/chunk decoding.
The budget excludes prefill and </think>. EOS or the overall output limit can
still truncate generation; the runner must validate the final JSON and sources.
"""
from __future__ import annotations


class BudgetedThinkingProcessor:
    requires_immediate_decode_yield = True

    def __init__(self, adapter, max_reasoning_tokens):
        self.adapter = adapter
        self.max_reasoning_tokens = max_reasoning_tokens
        self.reset()

    def reset(self):
        self.adapter.reset()
        self.reasoning_tokens = 0
        self.forced_closure = False
        self._started = self._closed = False

    def clone(self):
        return type(self)(self.adapter.clone(), self.max_reasoning_tokens)

    def process_last_token(self, last_token, logits):
        import mlx.core as mx

        if logits.ndim not in (1, 2) or (logits.ndim == 2 and logits.shape[0] != 1):
            raise ValueError("Reasoning budget requires single-sequence logits")
        closing = self.adapter.thinking_end_token_id
        if not self._started:
            self._started = True  # First invocation sees prompt, not generation.
        elif not self._closed:
            if last_token == closing:
                self._closed = True
            else:
                self.reasoning_tokens += 1
        if self._closed:
            return self.adapter.process_last_token(last_token, logits)
        if self.reasoning_tokens >= self.max_reasoning_tokens:
            self.forced_closure = True
            forced = mx.full_like(logits, -float("inf"))
            forced[..., closing] = 0  # Even a previously masked marker is forced.
            return forced
        return logits

    def __call__(self, input_ids, logits):
        if input_ids.ndim > 2 or (input_ids.ndim == 2 and input_ids.shape[0] != 1):
            raise ValueError("Reasoning budget requires a single token sequence")
        if input_ids.size == 0:
            raise ValueError("Reasoning budget requires prompt prefill first")
        return self.process_last_token(int(input_ids.reshape(-1)[-1].item()), logits)


def build_budgeted_thinking_processor(tokenizer, schema: dict, max_reasoning_tokens=256):
    """Build fresh budget and grammar state; load neither weights nor tokenizer."""
    from bidtrace_core.local.thinking import build_thinking_processor

    if type(max_reasoning_tokens) is not int or max_reasoning_tokens < 0:
        raise ValueError("max_reasoning_tokens must be a nonnegative integer")
    return BudgetedThinkingProcessor(build_thinking_processor(tokenizer, schema),
                                     max_reasoning_tokens)
