# SPDX-License-Identifier: MPL-2.0
"""Local MLX assessment and cached CPU-only BGE retrieval; no hosted calls."""
from __future__ import annotations

import os
import base64
import json
from pathlib import Path
import re
import threading
import time
import uuid

from .models import ModelClient, ModelFailure

EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL_ID = EMBEDDING_MODEL_ID + ":cosine"
EMBEDDING_CACHE = Path(os.getenv("BIDTRACE_EMBEDDING_CACHE", str(Path(__file__).resolve().parents[1] / "data/model-cache")))


class LocalTextRuntime:
    """One CPU ONNX model and RAM vector cache per worker process."""
    def __init__(self, cache_dir=EMBEDDING_CACHE, *, model_factory=None):
        self.cache_dir, self.model_factory = Path(cache_dir), model_factory
        self.model, self._vectors = None, {}
        self._lock, self.calls, self.runtime_id = threading.Lock(), [], uuid.uuid4().hex

    def weights_ready(self):
        cached = self.cache_dir / "models--qdrant--bge-small-en-v1.5-onnx-q"
        try:
            revision = (cached / "refs/main").read_text().strip()
            if not re.fullmatch(r"[a-f0-9]{40}", revision):
                return False
            snapshot = cached / "snapshots" / revision
            return all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0
                       for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                                    "special_tokens_map.json", "model_optimized.onnx"))
        except OSError:
            return False

    def _load(self):
        if self.model is None:
            if self.model_factory is None:
                from fastembed import TextEmbedding
                factory = TextEmbedding
            else:
                factory = self.model_factory
            self.model = factory(EMBEDDING_MODEL_ID, cache_dir=str(self.cache_dir), threads=2,
                providers=["CPUExecutionProvider"], cuda=False, local_files_only=True)

    def _embed(self, texts, is_query=False):
        self._load()
        keys = [(is_query, text) for text in texts]
        missing = list(dict.fromkeys(key for key in keys if key not in self._vectors))
        if missing:
            method = self.model.query_embed if is_query else self.model.passage_embed
            vectors = list(method([key[1] for key in missing]))
            if len(vectors) != len(missing):
                raise ModelFailure("Local embedding did not cover every source passage")
            import numpy as np
            for key, vector in zip(missing, vectors):
                vector = np.asarray(vector, dtype=np.float32)
                if vector.shape != (384,) or not np.isfinite(vector).all():
                    raise ModelFailure("Invalid local BGE embedding")
                self._vectors[key] = vector.tolist()
        return [list(self._vectors[key]) for key in keys]

    def invoke(self, method, *args, run_id=None, stage=None):
        with self._lock:
            started = time.perf_counter()
            record = {"id": len(self.calls) + 1, "runtime_id": self.runtime_id,
                "run_id": run_id, "stage": stage, "method": method, "provider": "mlx",
                "model": EMBEDDING_MODEL_ID if method == "embed" else RERANKER_MODEL_ID,
                "device": "cpu", "hosted_inference": False, "status": "running",
                "input_tokens": None, "output_tokens": 0}
            try:
                if method == "embed":
                    result = self._embed(*args)
                else:
                    import numpy as np
                    query, texts = args
                    if not texts:
                        record["status"] = "completed"
                        return []
                    vectors = np.asarray(self._embed(texts), dtype=np.float32)
                    q = np.asarray(self._embed([query], True)[0], dtype=np.float32)
                    cosine = (vectors @ q) / np.maximum(np.linalg.norm(vectors, axis=1) * np.linalg.norm(q), 1e-9)
                    result = np.clip(cosine, 0, 1).tolist()
                record["status"] = "completed"
                return result
            except Exception as exc:
                record.update(status="failed", error_type=type(exc).__name__)
                if isinstance(exc, ModelFailure):
                    raise
                raise ModelFailure(f"Offline CPU retrieval failed: {type(exc).__name__}") from exc
            finally:
                record["seconds"] = round(time.perf_counter() - started, 3)
                self.calls.append(record)


text_runtime = LocalTextRuntime()


class MLXClient(ModelClient):
    def __init__(self, url="http://127.0.0.1:8295", run_id=None, *, retrieval_runtime=None,
                 assessment_strategy="condition_nli"):
        super().__init__(url, run_id)
        if assessment_strategy not in {"condition_nli", "grounded"}:
            raise ValueError("Unknown local assessment strategy")
        self.assessment_strategy = assessment_strategy
        self.retrieval = retrieval_runtime or text_runtime
        self._condition_retrieval = None
        self._condition_init_lock = threading.Lock()

    def _retrieve(self, method, *args):
        return self.retrieval.invoke(method, *args, run_id=self.run_id, stage=self.stage)

    def embed(self, texts, is_query=False):
        return self._retrieve("embed", texts, is_query)

    def rerank(self, query, texts):
        return self._retrieve("rerank", query, texts)

    def visual_embed(self, images):
        raise ModelFailure("The MLX provider supports text and hybrid profiles; visual retrieval is unavailable")

    def visual_query(self, text):
        raise ModelFailure("The MLX provider supports text and hybrid profiles; visual retrieval is unavailable")

    def assess_group(self, requirement, evidence, images=None, *, image_sources=None, obligation_ids):
        payload = {"requirement": requirement, "evidence": evidence, "obligation_ids": obligation_ids,
                   "images": [base64.b64encode(raw).decode() for raw in (images or [])]}
        if self.assessment_strategy == "condition_nli":
            if images:
                raise ModelFailure("Condition NLI supports text evidence only")
            from .condition_evidence import ConditionEvidenceRetriever, select_condition_premise
            from bidtrace_core.local.batching import select_local_obligations
            contract = select_local_obligations(requirement, obligation_ids, 12)
            with self._condition_init_lock:
                if self._condition_retrieval is None:
                    self._condition_retrieval = ConditionEvidenceRetriever(self.retrieval)
            ranking = self._condition_retrieval.retrieve(contract, evidence)
            payload["condition_candidates"] = {row['obligation_id']: select_condition_premise(row, catalog=ranking['catalog'])
                                               for row in ranking['conditions']}
            catalog_sources = {source['citation_id']:source for source in ranking['catalog']}
            for row in ranking['conditions']:
                chosen = payload['condition_candidates'][row['obligation_id']]
                row['selected_citation_ids'] = chosen
                row['selected_evidence'] = [catalog_sources[cid] for cid in chosen]
            from .local_nli_gate import gate
            rejected, receipt = gate.screen(contract, ranking, run_id=self.run_id, stage=self.stage)
            remaining = [oid for oid in obligation_ids if oid not in rejected]
            if rejected:
                from .models import validate_obligation_selection
                payload['obligation_ids'] = remaining
                payload['condition_candidates'] = {oid: payload['condition_candidates'][oid] for oid in remaining}
                result = self._post('/assess/group', payload) if remaining else {'obligation_checks': [], 'grounding_observations': []}
                # The server already validated source scope and bounded facts.
                # A rejected sibling must not restore the other check's raw
                # model verdict or discard its source-grounding decision.
                checks = {c['obligation_id']: {'obligation_id': c['obligation_id'],
                          'assessment': {'gap': 'contradicted'}.get(c['assessment'], c['assessment']),
                          'rationale': c['rationale'], 'citation_ids': c['citation_ids']} for c in result['obligation_checks']}
                observations = list(result.get('grounding_observations', []))
                for condition in ranking['conditions']:
                    oid = condition['obligation_id']
                    if oid in rejected:
                        checks[oid] = {'obligation_id': oid, 'assessment': 'unknown', 'rationale': rejected[oid][:240],
                                      'citation_ids': condition['selected_citation_ids'][:2]}
                        observations.append({'obligation_id': oid, 'assessment': 'unknown', 'reasoning': rejected[oid][:240],
                            'evidence': [{'mode':'text','citation_id': c['citation_id'],'quote': c['quote']}
                                         for c in condition['selected_evidence'][:2]]})
                validated = validate_obligation_selection(json.dumps({'checks': [checks[oid] for oid in obligation_ids]}),
                    contract, ranking['catalog'], evidence)
                server_checks = {c['obligation_id']:c for c in result['obligation_checks']}
                for check in validated['obligation_checks']:
                    server = server_checks.get(check['obligation_id'])
                    if server:
                        for key in ('model_assessment','model_rationale','decision_rule'):
                            if key in server:
                                check[key] = server[key]
                        check['validation_issues'] = list(dict.fromkeys(check['validation_issues'] + server.get('validation_issues',[])))
                validated.update(grounding_observations=sorted(observations, key=lambda o: obligation_ids.index(o['obligation_id'])),
                    provider='mlx', hosted_inference=False, model=self.model_id if hasattr(self,'model_id') else None,
                    model_stats={**result.get('model_stats', {}), 'calls': [receipt, *result.get('model_stats', {}).get('calls', [])]})
                return validated
            payload['_prescreen_receipt'] = receipt
        if image_sources is not None:
            payload["image_sources"] = image_sources
        receipt = payload.pop('_prescreen_receipt', None)
        result = self._post("/assess/group", payload)
        if receipt:
            result.setdefault('model_stats', {}).setdefault('calls', []).insert(0, receipt)
        return result

    def status(self):
        snapshot = super().status()
        from .local_nli_gate import gate
        with gate.lock:
            prescreen_calls = [{key: value for key, value in c.items() if key != 'pair_receipts'} for c in gate.calls]
        return {**snapshot,
                "models": [{"id": model, "kind": kind, "weights_ready": self.retrieval.weights_ready(),
                            "successful_invocations": sum(c["status"] == "completed" and c["model"] == model
                                                          for c in self.retrieval.calls)}
                           for model, kind in [(EMBEDDING_MODEL_ID, "embedding"), (RERANKER_MODEL_ID, "reranker")]]
                          + snapshot.get("models", []),
                "invocations": [dict(c) for c in self.retrieval.calls] + prescreen_calls + snapshot.get("invocations", []),
                "retrieval_service": "local-cpu-fastembed", "supported_profiles": ["text", "hybrid"]}
