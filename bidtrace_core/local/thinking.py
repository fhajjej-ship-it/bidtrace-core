# SPDX-License-Identifier: MPL-2.0
"""Thinking-aware adapter for the isolated local MLX assessment trial.

Use with a prompt rendered using enable_thinking=True. The pinned Qwen template
prefills ``<think>\n``; generated text can therefore begin with reasoning without
another opening tag. The runner must require ``</think>`` before accepting the
final JSON and must still run the existing source validator on that suffix.

This helper loads neither model weights nor a tokenizer. Heavy decoder imports
are delayed until the caller supplies its already loaded local tokenizer.
"""
from __future__ import annotations


def build_thinking_processor(tokenizer, schema: dict):
    """Leave reasoning unmasked, then apply the unchanged schema after </think>."""
    from mlx_vlm.structured import (ThinkingAwareLogitsProcessor,
                                    build_json_schema_logits_processor)

    opening = tokenizer.encode("<think>", add_special_tokens=False)
    closing = tokenizer.encode("</think>", add_special_tokens=False)
    # The installed adapter activates by one token ID, not by matching sequences.
    if len(opening) != 1 or len(closing) != 1 or opening == closing:
        raise ValueError("Thinking adapter requires distinct single-token thinking markers")
    processor = build_json_schema_logits_processor(tokenizer, schema)
    return ThinkingAwareLogitsProcessor(processor=processor, tokenizer=tokenizer,
                                        enable_thinking=True)
