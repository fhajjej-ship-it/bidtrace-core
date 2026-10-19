# SPDX-License-Identifier: MPL-2.0
"""Persistent local-only MLX service; no weights are fetched at runtime.

Start with .venv-mlx/bin/python -m uvicorn bidtrace_v2.mlx_server:app
  --host 127.0.0.1 --port 8295 --workers 1
The three-ID checkpoint endpoint remains compatible with the production contract;
group requests accept twelve obligations and use serial generations of up to eight.
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time
import traceback
import uuid
from contextlib import nullcontext

from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, Field, model_validator

from .model_server import (AssessmentBatchRequest, AssessmentRequest, decode_images,
                           localhost_only)
from .models import (ModelFailure, build_citation_catalog, inference_context,
                     normalize_image_sources, select_obligation_batch,
                     validate_obligation_selection)
from .obligations import decompose_requirement
from .providers import MLX_STRATEGIES
from bidtrace_core.local.context import obligation_context
from bidtrace_core.local.batching import select_local_obligations
from bidtrace_core.local.grounding import (GROUNDING_INSTRUCTION, build_grounding_schema,
                                         validate_grounded_response)

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = MLX_STRATEGIES["condition_nli"]["model"]
MODEL_REVISION = MLX_STRATEGIES["condition_nli"]["revision"]
DEFAULT_MODEL_PATH = ROOT / "data-v2/local-models" / MODEL_ID.split("/")[-1]
SOURCE_PREFIX_CACHE_GIB = 0.75
CONDITION_PROMPT_TARGET_CHARS = 24_000


def prepare_assessment(requirement, contract, catalog, evidence, images, image_sources):
    """Build a model input exclusively from supplied sources and source contracts."""
    spans, contexts, span_ids, context_ids, obligations = {}, {}, {}, {}, []

    def intern(value, values, identifiers, prefix):
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in identifiers:
            identifier = prefix + str(len(values) + 1)
            identifiers[key] = identifier
            values[identifier] = value
        return identifiers[key]

    for obligation in contract["obligations"]:
        context = obligation_context(contract, obligation)
        buyer = {key: value for key, value in context.items()
                 if key not in {"obligation_id", "selected_span", "enclosing_span", "governing_spans"}}
        compact = {
            "selected_span_id": intern(context["selected_span"], spans, span_ids, "S"),
            "enclosing_span_id": intern(context["enclosing_span"], spans, span_ids, "S"),
            "governing_span_ids": [intern(span, spans, span_ids, "S") for span in context["governing_spans"]],
            "buyer_context_id": intern(buyer, contexts, context_ids, "B"),
        }
        obligations.append({"obligation_id": obligation["obligation_id"], "text": obligation["source_text"],
            "source_context": compact, "requirement_type": obligation["requirement_type"],
            "named_features": obligation["named_features"],
            "scope_review_reasons": obligation["decomposition_issues"]})
    source = {
        "requirement_role": "buyer_context_only",
        "requirement": {key: requirement[key] for key in
                        ("text", "source_text", "title")
                        if key in requirement} if isinstance(requirement, dict) else requirement,
        "context_reference_instruction": "Resolve every S ID in source_spans and B ID in buyer_context_catalog. "
            "Their exact text and paired questions are inherited buyer context for the selected obligation.",
        "source_spans": spans, "buyer_context_catalog": contexts, "obligation_batch": obligations,
    }
    # The supplier catalog is identical across buyer requests for the same pack.
    # Keep every existing catalog byte and ID; cache only this shared message,
    # never buyer conditions, schema state, reasoning, or a previous answer.
    supplier = {"citation_catalog": [{"citation_id": c["citation_id"], "text": c["quote"],
        "source_title": c["title"], "role": c["role"]} for c in catalog]}
    text = "\n".join([*("Source image provenance: " + json.dumps(s, ensure_ascii=False)
                         for s in image_sources), json.dumps(source, ensure_ascii=False)])
    return {"contract": contract, "catalog": catalog, "evidence": evidence,
            "images": images, "image_receipts": [{"normalized_source": s} for s in image_sources],
            "messages": [{"role": "system", "content": GROUNDING_INSTRUCTION},
                         {"role": "user", "content": json.dumps(supplier, ensure_ascii=False)},
                         {"role": "user", "content": text}]}


def canonical_grounded_selections(raw, prepared, *, max_quote_chars=600):
    """Keep the public E-ID contract after applying exact-span/scope validation.

    Raw labels and selected narrow spans remain in grounding_observations. The
    public proposal records the effective grounded decision against its original
    citation catalog, so existing storage and assembly validators can recheck it.
    """
    grounded = validate_grounded_response(raw, prepared, max_quote_chars=max_quote_chars)
    observations = {o["obligation_id"]: o for o in grounded["grounding_observations"]}
    selections = []
    for check in grounded["obligation_checks"]:
        observation = observations[check["obligation_id"]]
        selections.append({"obligation_id": check["obligation_id"],
            "assessment": {"gap": "contradicted"}.get(check["assessment"], check["assessment"]),
            "rationale": check["rationale"][:240],
            "citation_ids": list(dict.fromkeys(e["citation_id"] for e in observation["evidence"]))})
    return selections, grounded["grounding_observations"]


class MLXRuntime:
    def __init__(self, model_path=None, *, backend_factory=None, strategy=None, batch_limit=None,
                 max_new_tokens=None, thinking_budget=None, prefix_cache=None, memory_limit_gib=None):
        strategy = strategy or os.getenv("BIDTRACE_MLX_ASSESSMENT_STRATEGY", "condition_nli")
        if strategy not in MLX_STRATEGIES:
            raise ValueError("Choose the condition_nli or grounded MLX assessment strategy.")
        profile = MLX_STRATEGIES[strategy]
        batch_limit = profile["batch_limit"] if batch_limit is None else batch_limit
        max_new_tokens = profile["max_new_tokens"] if max_new_tokens is None else max_new_tokens
        thinking_budget = profile["thinking_budget"] if thinking_budget is None else thinking_budget
        prefix_cache = profile["prefix_cache"] if prefix_cache is None else prefix_cache
        allowed_batch_limits = (1, 3, 6, 8, 12) if strategy == 'condition_nli' else (3, 6, 8, 12)
        if batch_limit not in allowed_batch_limits:
            raise ValueError("Invalid MLX batch limit for the selected assessment strategy")
        if strategy == "condition_nli" and (prefix_cache):
            raise ValueError("Condition NLI requires no prefix cache")
        if not 1 <= max_new_tokens <= 4096 or not 1 <= thinking_budget <= 1024:
            raise ValueError("Invalid local generation budget")
        memory_limit_gib = (os.getenv("BIDTRACE_MLX_MEMORY_LIMIT_GIB", "10")
                            if memory_limit_gib is None else memory_limit_gib)
        if isinstance(memory_limit_gib, bool) or not 8 <= float(memory_limit_gib) <= 10:
            raise ValueError("Local MLX allocator must be bounded between 8 and 10 GiB")
        self.memory_limit_bytes = int(float(memory_limit_gib) * 1024**3)
        self.strategy, self.model_id, self.model_revision = strategy, profile["model"], profile["revision"]
        default_path = ROOT / "data-v2/local-models" / self.model_id.split("/")[-1]
        self.model_path = Path(model_path or os.getenv("BIDTRACE_MLX_MODEL_PATH", str(default_path)))
        self.batch_limit, self.max_new_tokens = batch_limit, max_new_tokens
        self.thinking_budget, self.prefix_cache = thinking_budget, prefix_cache
        self.backend_factory, self.backend = backend_factory, None
        self.runtime_id, self._lock = uuid.uuid4().hex, threading.RLock()
        self._calls, self._busy, self._memory = [], False, {}
        self._active_requests = 0
        from .local_generation_queue import LocalGenerationQueue
        self._native_run_id = None
        self._generation_queue = LocalGenerationQueue(self._decode_jobs)
        self._prefix_configured = False
        self._prefix_source_sha256, self._prefix_configuration = None, {}

    def _configure_source_prefix(self, case):
        """Bind one exact checkpoint to the instruction and supplier catalog."""
        if not self.prefix_cache or case["images"]:
            return
        source_messages = case["messages"][:2]
        identity = hashlib.sha256(json.dumps({"model": self.model_id, "revision": self.model_revision,
            "messages": source_messages}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if self._prefix_configured and identity == self._prefix_source_sha256:
            return
        # The empty buyer boundary is tokenized only, never sent for generation.
        boundary = {"messages": [*source_messages, {"role": "user", "content": ""}], "images": []}
        configuration = self.backend.configure_prefix_cache([case, boundary], enable_thinking=True) or {}
        cache = getattr(self.backend, "_prefix_cache", None)
        if cache is not None:
            prefix_ids = cache.shared_prefix_ids
            cache.clear()
            from bidtrace_core.local.cache import LocalPrefixCache
            self.backend._prefix_cache = LocalPrefixCache(prefix_ids, memory_max_gb=SOURCE_PREFIX_CACHE_GIB)
        self._prefix_configuration = {**configuration, "source_sha256": identity,
            "memory_max_bytes": int(SOURCE_PREFIX_CACHE_GIB * 1024**3)}
        self._prefix_source_sha256, self._prefix_configured = identity, True

    def weights_ready(self):
        try:
            required = ["config.json", "tokenizer.json", "tokenizer_config.json",
                        "preprocessor_config.json", "model.safetensors.index.json"]
            index = json.loads((self.model_path / "model.safetensors.index.json").read_text())
            shards = set(index["weight_map"].values())
            if not shards or any(not isinstance(name, str) or Path(name).name != name for name in shards):
                return False
            complete = all((self.model_path / name).is_file() and (self.model_path / name).stat().st_size > 0
                           for name in [*required, *shards])
            if not complete or self.backend_factory is not None:
                return complete
            # Cached download provenance prevents a stale path override from
            # loading a different model under the selected strategy's identity.
            metadata = self.model_path / ".cache/huggingface/download"
            return all((metadata / (name + ".metadata")).read_text().splitlines()[0] == self.model_revision
                       for name in ["config.json", *shards])
        except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError):
            return False

    def _condition_cases(self, contract, evidence, candidates):
        """Pack complete independent pairs by count and prompt size, never truncate."""
        from .condition_assessment import prepare_condition_case
        prepared, group, group_rows = [], [], []

        def finish():
            part = {**contract, "obligations": list(group)}
            prepared.append(prepare_condition_case(part, evidence,
                {o["obligation_id"]: candidates[o["obligation_id"]] for o in group}))

        for obligation in contract["obligations"]:
            single = prepare_condition_case({**contract, "obligations": [obligation]}, evidence,
                {obligation["obligation_id"]: candidates[obligation["obligation_id"]]})
            rows = json.loads(single["messages"][-1]["content"])
            combined_size = len(json.dumps([*group_rows, *rows], ensure_ascii=False))
            if group and (len(group) >= self.batch_limit or combined_size > CONDITION_PROMPT_TARGET_CHARS):
                finish()
                group, group_rows = [], []
            group.append(obligation)
            group_rows.extend(rows)
        if group:
            finish()
        return prepared

    def _decode_jobs(self, jobs):
        """The single queue worker owns all native GPU operations."""
        run_ids = {job[3].get('_run_id') for job in jobs}
        if len(run_ids) > 1:
            # Different analyses must not share retained native model state.
            return [self._decode_jobs([job])[0] for job in jobs]
        run_id = next(iter(run_ids))
        if run_id is not None and run_id != self._native_run_id:
            if self.backend is not None:
                mx = getattr(self.backend, 'mx', None)
                if mx is not None:
                    mx.synchronize()
                self.backend = None
                gc.collect()
                if mx is not None:
                    mx.clear_cache()
            self._native_run_id = run_id
        started = time.perf_counter()
        self._load()
        load_seconds = time.perf_counter() - started
        try:
            if self.backend_factory is None:
                from .local_mlx_batch_decode import generate_many
                results = generate_many(self.backend, jobs)
            else:
                results = [self.backend.generate(messages, images, limit, **options)
                           for messages, images, limit, options in jobs]
            for _, stats in results:
                stats['load_seconds'] = load_seconds / len(jobs)
            self._memory = self.backend.memory()
            return results
        except Exception:
            self.backend = None
            raise

    def _load(self):
        if self.backend is not None:
            return
        if not self.weights_ready():
            raise ModelFailure("MLX weights unavailable; an existing local model directory is required")
        factory = self.backend_factory
        if factory is None:
            from bidtrace_core.local.runtime import MLXBackend
            factory = MLXBackend
        try:
            self.backend = factory(self.model_path)
            mx = getattr(self.backend, "mx", None)
            if mx is not None:
                # Full supplier context measured 8.36 GB peak on the 24 GB Mac.
                # Long buyer tails need headroom beyond the trial's 8 GiB cap;
                # keep this service allocation bounded and the trial unchanged.
                mx.set_memory_limit(self.memory_limit_bytes)
        except Exception as exc:
            self.backend = None
            raise ModelFailure(f"Local MLX model loading failed: {type(exc).__name__}") from exc

    def status(self):
        return {"service": "local-mlx", "provider": "mlx", "hosted_inference": False,
            "runtime_id": self.runtime_id, "busy": self._busy,
            "loaded_model": self.model_id if self.backend is not None else None,
            "assessment_strategy": self.strategy,
            "device": "metal", "dtype": "4bit", "memory": dict(self._memory),
            "models": [{"id": self.model_id, "kind": "reasoning",
                "weights_ready": self.weights_ready(),
                "revision": self.model_revision,
                "successful_invocations": sum(c["status"] == "completed" for c in self._calls)}],
            "limits": {"heavy_model_concurrency": 1, "native_prompt_batch_size": 8,
                "batch_limit": self.batch_limit,
                "max_new_tokens": self.max_new_tokens, "thinking_budget": self.thinking_budget,
                "mlx_memory_limit_bytes": self.memory_limit_bytes,
                "condition_prompt_target_chars": CONDITION_PROMPT_TARGET_CHARS if self.strategy == "condition_nli" else None,
                "automatic_retries": 0, "prefix_cache": self.prefix_cache,
                "prefix_cache_max_bytes": int(SOURCE_PREFIX_CACHE_GIB * 1024**3) if self.prefix_cache else 0},
            "prefix_configuration": dict(self._prefix_configuration),
            "invocations": [dict(call) for call in self._calls]}

    def unload(self):
        with self._lock:
            if self.backend is not None:
                cache = getattr(self.backend, "_prefix_cache", None)
                if cache is not None:
                    cache.clear()
                mx = getattr(self.backend, "mx", None)
                self.backend = None
                gc.collect()
                if mx is not None:
                    mx.clear_cache()
            self._memory = {}
            self._prefix_configured = False
            self._prefix_source_sha256, self._prefix_configuration = None, {}

    def assess(self, requirement, evidence, images=None, *, image_sources=None, condition_candidates=None):
        return self._assess(requirement, evidence, images, image_sources=image_sources,
                            condition_candidates=condition_candidates)

    def assess_batch(self, requirement, evidence, images=None, *, image_sources=None, obligation_ids,
                     condition_candidates=None):
        selected = select_obligation_batch(requirement, obligation_ids)
        return self._assess(requirement, evidence, images, image_sources=image_sources, selected=selected,
                            condition_candidates=condition_candidates)

    def assess_group(self, requirement, evidence, images=None, *, image_sources=None, obligation_ids,
                     condition_candidates=None):
        selected = select_local_obligations(requirement, obligation_ids, 12)
        return self._assess(requirement, evidence, images, image_sources=image_sources,
                            selected=selected, grouped=True, condition_candidates=condition_candidates)

    def _assess(self, requirement, evidence, images, *, image_sources, selected=None, grouped=False,
                condition_candidates=None):
        full = decompose_requirement(requirement)
        contract = full if selected is None else selected
        images = images or []
        if self.strategy == "condition_nli" and images:
            raise ModelFailure("Condition NLI supports text evidence only")
        sources = normalize_image_sources(len(images), image_sources, evidence)
        catalog = build_citation_catalog(evidence)
        if any(c["role"] != "evidence" for c in catalog):
            raise ModelFailure("MLX assessments require supplier-only evidence")
        selections, observations, records = [], [], []
        if not catalog:
            unknown = {"checks": [{"obligation_id": o["obligation_id"], "assessment": "unknown",
                "reasoning": "No supplier evidence was supplied for this obligation.", "evidence": []}
                for o in contract["obligations"]]}
            selections, observations = canonical_grounded_selections(json.dumps(unknown),
                {"contract": contract, "catalog": catalog, "evidence": evidence, "images": [], "image_receipts": []})
        else:
            from PIL import Image
            from bidtrace_core.local.runtime import extract_answer
            try:
                page_images = [Image.open(io.BytesIO(raw)).convert("RGB") for raw in images]
            except Exception as exc:
                raise ModelFailure("Invalid local assessment image") from exc
            slices = [{**contract, "obligations": contract["obligations"][start:start + self.batch_limit]}
                      for start in range(0, len(contract["obligations"]), self.batch_limit)]
            if self.strategy == "condition_nli":
                from .condition_assessment import expand_condition_response
                if not isinstance(condition_candidates, dict):
                    raise ModelFailure("Condition NLI requires condition-specific supplier candidates")
                expected_ids = {o["obligation_id"] for o in contract["obligations"]}
                if set(condition_candidates) != expected_ids:
                    raise ModelFailure("Condition candidates must cover exactly the selected obligation IDs")
                prepared = self._condition_cases(contract, evidence, condition_candidates)
            else:
                if condition_candidates is not None:
                    raise ModelFailure("Condition candidates require the condition_nli strategy")
                prepared = [prepare_assessment(requirement, part, catalog, evidence, page_images, sources)
                            for part in slices]
            with (nullcontext() if self.strategy == 'condition_nli' else self._lock):
                with self._lock:
                    self._active_requests += 1
                    self._busy = True
                try:
                    for index, case in enumerate(prepared):
                        if self.strategy == "condition_nli" and not any(case["condition_candidates"].values()):
                            empty = json.dumps([{"id": str(number), "verdict": "N",
                                "reasoning": "No supplier candidates were supplied for this obligation."}
                                for number, _ in enumerate(case["contract"]["obligations"], 1)])
                            expanded = expand_condition_response(empty, case)
                            checks, grounded = canonical_grounded_selections(json.dumps(expanded), case,
                                                                             max_quote_chars=800)
                            selections.extend(checks)
                            observations.extend(grounded)
                            continue
                        start = time.perf_counter()
                        context = inference_context.get()
                        record = {"id": len(self._calls) + 1, "runtime_id": self.runtime_id,
                            "run_id": context.get("run_id"), "stage": context.get("stage"),
                            "model": self.model_id, "provider": "mlx", "revision": self.model_revision,
                            "method": "assess_group" if grouped else "assess" if selected is None else "assess_batch", "kind": "reasoning",
                            "status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "load_seconds": 0,
                            "input_tokens": None, "output_tokens": None,
                            "obligation_ids": [o["obligation_id"] for o in case["contract"]["obligations"]],
                            "images": len(page_images), "hosted_inference": False,
                            "mlx_memory_limit_bytes": self.memory_limit_bytes,
                            "prompt_sha256": hashlib.sha256(json.dumps(case["messages"], ensure_ascii=False).encode()).hexdigest(),
                            "user_prompt_chars": sum(len(m["content"]) for m in case["messages"] if m["role"] == "user"),
                            "condition_prompt_target_chars": CONDITION_PROMPT_TARGET_CHARS if self.strategy == "condition_nli" else None,
                            "thinking_budget": self.thinking_budget, "max_new_tokens": self.max_new_tokens,
                            "decoding_strategy": "condition_nli" if self.strategy == "condition_nli" else "mlx_grounded_exact_spans_v1"}
                        with self._lock:
                            record['id'] = len(self._calls) + 1
                            self._calls.append(record)
                        try:
                            if self.strategy == 'condition_nli':
                                raw, stats = self._generation_queue.submit((case['messages'], case['images'],
                                    self.max_new_tokens, {'response_schema': case['response_schema'],
                                    'enable_thinking': True, 'thinking_budget': self.thinking_budget,
                                    '_run_id': context.get('run_id')}))
                                record.update({key: stats[key] for key in ('native_batch_id', 'native_batch_size',
                                    'native_batch_seconds', 'load_seconds') if key in stats})
                            else:
                                load_start = time.perf_counter()
                                try:
                                    self._load()
                                finally:
                                    record['load_seconds'] = round(time.perf_counter() - load_start, 3)
                                self._configure_source_prefix(case)
                                record['prefix_configuration'] = (dict(self._prefix_configuration)
                                    if self.prefix_cache and not page_images else {})
                                raw, stats = self.backend.generate(case['messages'], case['images'],
                                    self.max_new_tokens, response_schema=build_grounding_schema(case),
                                    enable_thinking=True, thinking_budget=self.thinking_budget)
                            record.update(input_tokens=stats.get("prompt_tokens"), output_tokens=stats.get("generation_tokens"),
                                total_tokens=stats.get("total_tokens"), cached_tokens=stats.get("cached_tokens"),
                                reasoning_tokens=stats.get("reasoning_tokens"), finish_reason=stats.get("finish_reason"),
                                prefix_cache=stats.get("prefix_cache"))
                            answer, _ = extract_answer(raw, thinking_prefilled=bool(stats.get("thinking_prefilled")))
                            if self.strategy == "condition_nli":
                                expanded = expand_condition_response(answer, case)
                                checks, grounded = canonical_grounded_selections(json.dumps(expanded), case,
                                                                                 max_quote_chars=800)
                            else:
                                checks, grounded = canonical_grounded_selections(answer, case)
                            selections.extend(checks)
                            observations.extend(grounded)
                            record["status"] = "completed"
                        except Exception as exc:
                            record.update(status="failed", error_type=type(exc).__name__)
                            # Retain code locations for local runtime diagnosis
                            # without exception messages, prompts or source text.
                            record['error_stack'] = [{'file': Path(frame.filename).name,
                                'line': frame.lineno, 'function': frame.name}
                                for frame in traceback.extract_tb(exc.__traceback__)]
                            if isinstance(exc, ModelFailure):
                                raise
                            # A failed native generation/cache setup must not leave
                            # a potentially invalid model cache for the next request.
                            self.backend, self._prefix_configured = None, False
                            self._prefix_source_sha256, self._prefix_configuration = None, {}
                            gc.collect()
                            raise ModelFailure(f"Local MLX assessment failed: {type(exc).__name__}") from exc
                        finally:
                            record["seconds"] = round(stats.get('seconds', time.perf_counter() - start), 3) if record['status'] == 'completed' else round(time.perf_counter() - start, 3)
                            records.append(dict(record))
                    if self.backend is not None:
                        try:
                            self._memory = self.backend.memory()
                        except Exception as exc:
                            self._memory = {"available": False, "error_type": type(exc).__name__}
                finally:
                    with self._lock:
                        self._active_requests -= 1
                        self._busy = self._active_requests > 0
        result = validate_obligation_selection(json.dumps({"checks": selections}), contract, catalog, evidence)
        result.update(model=self.model_id, provider="mlx", hosted_inference=False,
            grounding_observations=observations,
            assessment_strategy=("condition_nli" if self.strategy == "condition_nli"
                else "mlx_grounded_exact_spans_v1") if catalog else "no_supplier_evidence_v1")
        if selected is not None:
            scope = "obligation_group" if grouped else "obligation_batch"
            result.update(assessment_scope=scope, **{scope: {
                "obligation_ids": [o["obligation_id"] for o in contract["obligations"]],
                "total_obligations": len(full["obligations"])}})
        if records:
            result["model_stats"] = {**records[-1], "calls": records,
                "input_tokens": sum(r["input_tokens"] for r in records) if all(type(r["input_tokens"]) is int for r in records) else None,
                "output_tokens": sum(r["output_tokens"] for r in records) if all(type(r["output_tokens"]) is int for r in records) else None,
                "total_tokens": sum(r["total_tokens"] for r in records) if all(type(r["total_tokens"]) is int for r in records) else None,
                "seconds": round(sum(r["seconds"] for r in records), 3)}
        return result


runtime = MLXRuntime()
app = FastAPI(title="BidTrace local MLX assessment")
app.middleware("http")(localhost_only)


@app.get("/health")
def health():
    return runtime.status()


def _invoke(body, *, obligation_ids=None, grouped=False):
    kwargs = {"image_sources": [s.model_dump() for s in body.image_sources]
              if body.image_sources is not None else None}
    if hasattr(body, "condition_candidates"):
        kwargs["condition_candidates"] = body.condition_candidates
    method = runtime.assess
    if obligation_ids is not None:
        method, kwargs["obligation_ids"] = runtime.assess_group if grouped else runtime.assess_batch, obligation_ids
    try:
        return method(body.requirement, body.evidence, decode_images(body.images), **kwargs)
    except ModelFailure as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/assess")
def assess(body: AssessmentRequest):
    return _invoke(body)


@app.post("/assess/batch")
def assess_batch(body: AssessmentBatchRequest):
    return _invoke(body, obligation_ids=body.obligation_ids)


class AssessmentGroupRequest(AssessmentRequest):
    model_config = ConfigDict(extra="forbid")
    obligation_ids: list[str] = Field(min_length=1, max_length=12)
    condition_candidates: dict[str, list[str]] | None = None

    @model_validator(mode="after")
    def canonical_ids(self):
        try:
            select_local_obligations(self.requirement, self.obligation_ids, 12)
        except ModelFailure as exc:
            raise ValueError(str(exc)) from exc
        return self


@app.post("/assess/group")
def assess_group(body: AssessmentGroupRequest):
    return _invoke(body, obligation_ids=body.obligation_ids, grouped=True)


@app.post("/unload")
def unload():
    runtime.unload()
    return {"unloaded": True}
