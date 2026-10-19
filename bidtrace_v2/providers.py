# SPDX-License-Identifier: MPL-2.0
"""Explicit inference identity; provider changes create independent runs and indexes."""
from __future__ import annotations

LOCAL_MODELS = {
    "embedding": "Qwen/Qwen3-Embedding-0.6B",
    "reranker": "Qwen/Qwen3-Reranker-0.6B",
    "visual": "vidore/colqwen2-v1.0",
    "generation": "Qwen/Qwen3-VL-4B-Instruct",
}

MLX_STRATEGIES = {
    "condition_nli": {"model": "mlx-community/Qwen3.5-2B-MLX-4bit",
        "revision": "93760be4f1f69842a46bc13dbdc0f19e291392a3", "batch_limit": 1,
        "max_new_tokens": 2048, "thinking_budget": 128, "prefix_cache": False},
    "grounded": {"model": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "revision": "938d8919941c6e7efd3c7150eff7fe9d12afa631", "batch_limit": 12,
        "max_new_tokens": 4096, "thinking_budget": 512, "prefix_cache": True},
}
MLX_MODELS = {"embedding": "BAAI/bge-small-en-v1.5", "reranker": "BAAI/bge-small-en-v1.5:cosine",
              "visual": None, "generation": MLX_STRATEGIES["condition_nli"]["model"]}


def mlx_strategy(config):
    name = getattr(config, "mlx_assessment_strategy", "condition_nli")
    if name not in MLX_STRATEGIES:
        raise ValueError("Choose the condition_nli or grounded MLX assessment strategy.")
    return name, MLX_STRATEGIES[name]


def model_ids(config):
    if getattr(config, 'standalone_core', False) and config.inference_provider != 'mlx':
        raise ValueError('Hosted adapters are unavailable in the standalone core.')
    if getattr(config, "inference_provider", "local") == "codex":
        return {**MLX_MODELS, "generation": config.codex_model}
    if getattr(config, "inference_provider", "local") == "mlx":
        return {**MLX_MODELS, "generation": mlx_strategy(config)[1]["model"]}
    if getattr(config, "inference_provider", "local") == "gemini":
        return {"embedding": config.gemini_embedding_model, "visual": config.gemini_embedding_model,
                "reranker": config.gemini_generation_model, "generation": config.gemini_generation_model}
    return dict(LOCAL_MODELS)


def provider_identity(config):
    identity = {"provider": getattr(config, "inference_provider", "local"), "models": model_ids(config)}
    if identity["provider"] == "codex":
        from development.codex.client import implementation_identity
        identity.update(development_only=True, hosted_inference=True,
                        authentication="chatgpt_subscription", embedding_dimensions=384,
                        visual_strategy=None, embedding_recipe="fastembed_bge_cpu_v1",
                        assessment_contract="source_obligations_v1",
                        generation_settings={"reasoning_effort": config.codex_reasoning_effort,
                                             "recipe": "complete_requirement_supplier_catalog_v1",
                                             "implementation": implementation_identity()},
                        execution_settings={"max_concurrent_requests": config.codex_concurrency})
    if identity["provider"] == "gemini":
        identity.update(embedding_dimensions=config.gemini_embedding_dimensions,
                        visual_strategy="dense_cosine", embedding_recipe="gemini_embedding2_search_v1",
                        assessment_contract="source_obligations_v1",
                        execution_settings={"max_concurrent_requests": getattr(config, "gemini_max_concurrency", 8)},
                        generation_settings={"max_output_tokens": config.gemini_max_output_tokens,
                                             "rerank_max_output_tokens": min(config.gemini_max_output_tokens, 2048),
                                             "thinking_level": "low", "temperature": 0})
    elif identity["provider"] == "mlx":
        strategy, generation = mlx_strategy(config)
        identity.update(embedding_dimensions=384, visual_strategy=None,
                        embedding_recipe="fastembed_bge_cpu_v1", assessment_contract="source_obligations_v1",
                        generation_settings={"strategy": strategy, "revision": generation["revision"],
                                             "max_group_size": generation["batch_limit"],
                                             "max_request_group_size": 12,
                                             "condition_prompt_target_chars": 24_000 if strategy == "condition_nli" else None,
                                             "max_output_tokens": generation["max_new_tokens"],
                                             "condition_supplier_passages": 2 if strategy == "condition_nli" else None,
                                             "condition_premise_selection": "explicit_term_product_service_with_literal_corroboration_v5" if strategy == "condition_nli" else None,
                                             "thinking_budget": generation["thinking_budget"], "temperature": 0,
                                             "prescreen": {"model": "cross-encoder/nli-deberta-v3-xsmall",
                                                 "revision": "a150876415327c80daeff35ca6f68f5ed8cf5c24",
                                                 "recipe": "complete_sources_reject_only_v3_recovery_point_scope", "neutral_threshold": .995}
                                                 if strategy == "condition_nli" else None},
                        execution_settings={"max_concurrent_requests": 12 if strategy == 'condition_nli' else 1,
                                            "native_prompt_batch_size": 8 if strategy == 'condition_nli' else 1,
                                            "mlx_memory_limit_bytes": int(getattr(config, "mlx_memory_limit_gib", 10) * 1024**3),
                                            "source_prefix_cache_limit_bytes": 768 * 1024**2 if generation["prefix_cache"] else 0})
    return identity


def create_model_client(config, run_id=None):
    if getattr(config, 'standalone_core', False) and config.inference_provider != 'mlx':
        raise ValueError('Hosted adapters are unavailable in the standalone core.')
    if getattr(config, "inference_provider", "local") == "codex":
        from development.codex.client import CodexClient
        return CodexClient(config, run_id=run_id)
    if getattr(config, "inference_provider", "local") == "mlx":
        from .mlx_client import MLXClient
        return MLXClient(config.model_url, run_id=run_id, assessment_strategy=mlx_strategy(config)[0])
    if getattr(config, "inference_provider", "local") == "gemini":
        from .gemini import GeminiClient
        return GeminiClient(config, run_id=run_id)
    from .models import ModelClient
    return ModelClient(config.model_url, run_id=run_id)
