# SPDX-License-Identifier: MPL-2.0
"""Local MLX runtime extracted unchanged from the development trial harness."""
from __future__ import annotations
import hashlib, os, platform, resource
from pathlib import Path
from bidtrace_v2.models import ModelFailure

SETTINGS = {"temperature": 0.0, "enable_thinking": False, "max_new_tokens": 4096,
            "mlx_memory_limit_bytes": 8 * 1024**3, "mlx_cache_limit_bytes": 256 * 1024**2,
            "max_concurrency": 1, "automatic_retries": 0, "warmup_generation_calls": 0}


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def peak_rss_gb(raw=None, system=None) -> float:
    """getrusage uses bytes on macOS and KiB on Linux; report decimal GB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if raw is None else raw
    system = platform.system() if system is None else system
    if system not in {"Darwin", "Linux"}:
        raise RuntimeError("Unsupported platform for peak RSS units")
    return raw * (1 if system == "Darwin" else 1024) / 1e9


def extract_answer(raw: str, *, thinking_prefilled=False) -> tuple[str, str]:
    """Remove only explicit complete wrappers; never search for or repair JSON."""
    text, wrapper = raw.strip(), "none"
    if thinking_prefilled:
        if text.count("</think>") != 1:
            raise ModelFailure("Expected one thinking closing marker; no complete final answer")
        text, wrapper = text.split("</think>", 1)[1].strip(), "prefilled_think_wrapper"
    elif text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise ModelFailure("Unclosed thinking wrapper; no final answer")
        text, wrapper = text[end + len("</think>"):].strip(), "complete_think_wrapper"
    if text.startswith("```json\n") and text.endswith("\n```"):
        text, wrapper = text[len("```json\n"):-len("\n```")].strip(), wrapper + "+json_fence"
    return text, wrapper


class MLXBackend:
    """Heavy imports are lazy: offline tests inject a fake backend without MLX."""
    def __init__(self, model_path: Path):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
        self.mx, self._generate, self._template = mx, generate, apply_chat_template
        mx.set_memory_limit(SETTINGS["mlx_memory_limit_bytes"])
        mx.set_cache_limit(SETTINGS["mlx_cache_limit_bytes"])
        self.config = load_config(str(model_path))
        self.model, self.processor = load(str(model_path), lazy=False, local_files_only=True,
                                          trust_remote_code=False)
        self._prefix_cache = None
        mx.synchronize()

    def memory(self):
        return {"mlx_active_gb": self.mx.get_active_memory() / 1e9,
                "mlx_peak_gb": self.mx.get_peak_memory() / 1e9,
                "mlx_cache_gb": self.mx.get_cache_memory() / 1e9,
                "process_peak_rss_gb": peak_rss_gb()}

    def _text_prompt_ids(self, prompt):
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        # Match installed generate's text-only preparation, including markers.
        inputs = prepare_inputs(self.processor, prompts=prompt,
            add_special_tokens=should_add_special_tokens(self.model.config.model_type, self.processor))
        return inputs["input_ids"].reshape(-1).tolist()

    def configure_prefix_cache(self, prepared_cases, *, enable_thinking):
        from bidtrace_core.local.cache import LocalPrefixCache

        previous = getattr(self, "_prefix_cache", None)
        if previous is not None:
            previous.clear()
        self._prefix_cache = None
        sequences = [self._text_prompt_ids(self._template(self.processor, self.config, case["messages"],
            num_images=0, enable_thinking=enable_thinking)) for case in prepared_cases if not case["images"]]
        prefix = []
        if len(sequences) >= 2:
            prefix = sequences[0][:-1]  # Always retain an uncached final token.
            for tokens in sequences[1:]:
                boundary = next((i for i, pair in enumerate(zip(prefix, tokens)) if pair[0] != pair[1]),
                                min(len(prefix), len(tokens) - 1))
                prefix = prefix[:boundary]
            if len(prefix) >= 16:
                self._prefix_cache = LocalPrefixCache(prefix)
        return {"enabled": self._prefix_cache is not None, "eligible_text_requests": len(sequences),
                "shared_prefix_tokens": len(prefix) if self._prefix_cache is not None else 0}

    def generate(self, messages, images, max_new_tokens, *, response_schema=None, enable_thinking=False, thinking_budget=None):
        prompt = self._template(self.processor, self.config, messages, num_images=len(images),
                                enable_thinking=enable_thinking)
        thinking_prefilled = prompt.endswith("<think>\n")
        if enable_thinking and not thinking_prefilled:
            raise ModelFailure("Thinking-enabled template did not prefill the expected opening marker")
        processors = None
        if response_schema is not None:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            if enable_thinking:
                if thinking_budget is not None:
                    from bidtrace_core.local.budget import build_budgeted_thinking_processor
                    processors = [build_budgeted_thinking_processor(tokenizer, response_schema, thinking_budget)]
                else:
                    from bidtrace_core.local.thinking import build_thinking_processor
                    processors = [build_thinking_processor(tokenizer, response_schema)]
            else:
                from mlx_vlm.structured import build_json_schema_logits_processor
                processors = [build_json_schema_logits_processor(tokenizer, response_schema)]
        prefix_cache = getattr(self, "_prefix_cache", None)
        cache_kwargs = (prefix_cache.generation_kwargs(self._text_prompt_ids(prompt))
                        if prefix_cache is not None and not images else {})
        response = self._generate(self.model, self.processor, prompt=prompt, image=images or None,
                                  max_tokens=max_new_tokens, temperature=0.0, verbose=False,
                                  logits_processors=processors, enable_thinking=enable_thinking,
                                  skip_special_tokens=False, **cache_kwargs)
        self.mx.synchronize()
        stats = {name: getattr(response, name, None) for name in ("prompt_tokens", "generation_tokens",
            "total_tokens", "prompt_tps", "generation_tps", "peak_memory", "cached_tokens", "finish_reason")}
        stats["formatted_prompt_sha256"] = sha256(prompt.encode())
        stats["thinking_prefilled"] = thinking_prefilled
        if prefix_cache is not None:
            stats["prefix_cache"] = {"request_eligible": bool(cache_kwargs),
                "shared_prefix_tokens": len(prefix_cache.shared_prefix_ids),
                "hit_tokens": stats["cached_tokens"], "counters": prefix_cache.stats()}
        if thinking_budget is not None:
            stats['reasoning_tokens'] = getattr(processors[0], 'reasoning_tokens', None)
            stats['forced_closure'] = getattr(processors[0], 'forced_closure', None)
        return response.text, stats

