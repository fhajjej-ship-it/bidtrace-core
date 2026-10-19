# SPDX-License-Identifier: MPL-2.0
"""Tenant-scoped Qdrant retrieval with isolated local and Gemini vector indexes."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
import re
import time
from urllib.parse import urlparse
import uuid
from .models import MODEL_IDS, ModelClient

TEXT_COLLECTION = "bidtrace_v2_qwen3_text"
VISUAL_COLLECTION = "bidtrace_v2_colqwen_pages"
PROFILES = {"text", "hybrid", "multimodal"}

class RetrievalFailure(RuntimeError):
    pass


def tokens(text):
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def lexical_rank(query: str, passages: list[dict], limit=20) -> list[tuple[str, float]]:
    """BM25 over the already-authorized source pages only."""
    if not passages:
        return []
    queries = tokens(query)
    bags = [Counter(tokens(p["text"])) for p in passages]
    lengths = [sum(b.values()) for b in bags]
    average = max(sum(lengths) / len(lengths), 1)
    counts = Counter(term for bag in bags for term in bag)
    ranked = []
    for passage, bag, length in zip(passages, bags, lengths):
        score = 0.0
        for term in queries:
            freq = bag[term]
            if freq:
                idf = math.log(1 + (len(bags) - counts[term] + .5) / (counts[term] + .5))
                score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / average))
        if score > 0:
            ranked.append((passage["point_id"], score))
    return sorted(ranked, key=lambda x: (-x[1], x[0]))[:limit]


def fuse_rankings(rankings: list[list[tuple[str, float]]], k=60) -> list[tuple[str, float]]:
    totals = Counter()
    for ranking in rankings:
        for rank, (key, _) in enumerate(ranking, start=1):
            totals[key] += 1 / (k + rank)
    return sorted(totals.items(), key=lambda x: (-x[1], x[0]))


def chunk_page(text: str, size=1800, overlap=200):
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def bounded_supplier_context(documents, *, max_pages=20, max_characters=40000):
    """Keep every readable supplier page when the complete pack fits the local budget.

    This avoids a broad buyer section hiding relevant supplier details behind a
    top-k cutoff. Larger or partly unreadable packs retain the retrieval path.
    No expected answers, reference quotations, or buyer pages select evidence.
    """
    suppliers = [doc for doc in documents if doc.get('role') == 'evidence' and doc.get('active', True)]
    pages, characters, identities = [], 0, set()
    for doc in sorted(suppliers, key=lambda item: item['id']):
        if not doc.get('pages'):
            return None
        for page in sorted(doc['pages'], key=lambda item: item['page']):
            text = page.get('text')
            if not isinstance(text, str) or not text.strip():
                return None
            identity = (doc['id'], page['page'])
            if identity in identities:
                raise RetrievalFailure('Supplier source pages must be uniquely identified.')
            identities.add(identity)
            characters += len(text)
            pages.append({'document_id': doc['id'], 'title': doc['title'], 'page': page['page'],
                          'text': text, 'role': 'evidence', 'version': doc.get('version'),
                          'method': 'bounded_complete_supplier_text'})
            if len(pages) > max_pages or characters > max_characters:
                return None
    return pages or None


_embedded_clients = {}


class QdrantRetriever:
    def __init__(self, url: str, model_url: str, workspace_id="local", run_id=None, *, config=None, models=None):
        from qdrant_client import QdrantClient
        if urlparse(url).hostname not in {"127.0.0.1", "localhost", "::1", "qdrant"}:
            raise ValueError("The vector database must remain local")
        if not workspace_id:
            raise ValueError("workspace_id is mandatory")
        self.workspace_id = workspace_id
        if config is not None and config.storage_backend == "embedded":
            path = str((config.data_dir / "vectors").resolve())
            if path not in _embedded_clients:
                _embedded_clients[path] = QdrantClient(path=path, force_disable_check_same_thread=True)
            self.client = _embedded_clients[path]
        else:
            self.client = QdrantClient(url=url, timeout=120)
        self.config = config
        if models is not None:
            self.models = models
        elif config is not None:
            from .providers import create_model_client
            self.models = create_model_client(config, run_id=run_id)
        else:
            self.models = ModelClient(model_url, run_id=run_id)
        self._query_embeddings = {}
        self._visual_queries = {}
        self.check_cancel = None
        self.report_progress = None

    def _gemini(self):
        return getattr(getattr(self, "config", None), "inference_provider", "local") == "gemini"

    def _modern_dense(self):
        return getattr(getattr(self, "config", None), "inference_provider", "local") in {"gemini", "mlx", "codex"}

    def _model_ids(self):
        if self._modern_dense():
            from .providers import model_ids
            return model_ids(self.config)
        return MODEL_IDS

    def _index_identity(self):
        from .providers import provider_identity
        identity = provider_identity(self.config)
        return {"provider": identity["provider"], "model": self._model_ids()["embedding"],
            "dimensions": identity["embedding_dimensions"], "recipe": identity["embedding_recipe"],
            "visual_strategy": identity["visual_strategy"], "index_version": 1}

    def _collection_name(self, visual=False):
        if not self._modern_dense():
            return VISUAL_COLLECTION if visual else TEXT_COLLECTION
        identity = self._index_identity()
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        model = re.sub(r"[^a-z0-9]+", "_", identity["model"].lower()).strip("_")[:40]
        kind = "pages" if visual else "text"
        return f"bidtrace_v2_{identity['provider']}_{kind}_{model}_{identity['dimensions']}_v1_{fingerprint}"

    @property
    def text_collection(self):
        return self._collection_name()

    @property
    def visual_collection(self):
        return self._collection_name(visual=True)

    def _dimensions(self, visual=False):
        if self._modern_dense():
            return self._index_identity()["dimensions"]
        return 128 if visual else 1024

    def _model_call(self, operation, *args, **kwargs):
        self._check()
        if self._gemini():
            # The workflow assigns its cancellation callback after construction.
            self.models.check_cancel = self._check
        # Record a fully completed batch before the next cancellation boundary.
        # Gemini checks cancellation between its individual HTTP requests.
        return getattr(self.models, operation)(*args, **kwargs)

    def _dense_vector(self, vector, *, visual=False):
        size = self._dimensions(visual)
        if (not isinstance(vector, (list, tuple)) or len(vector) != size
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) for value in vector)):
            raise RetrievalFailure(f"Embedding vector must contain {size} finite numeric dimensions")
        return list(vector)

    def _visual_vector(self, vector):
        if not isinstance(vector, (list, tuple)) or not vector:
            raise RetrievalFailure("Visual embedding service returned no vectors")
        if self._gemini():
            if len(vector) != 1:
                raise RetrievalFailure("Gemini page embedding must contain exactly one dense vector")
            return self._dense_vector(vector[0], visual=True)
        return [self._dense_vector(patch, visual=True) for patch in vector]

    def _retrieval_metadata(self, visual=False, rerank=False):
        ids = self._model_ids()
        return {"provider": getattr(getattr(self, "config", None), "inference_provider", "local"),
            "text_model": ids["embedding"], "reranker_model": ids["reranker"] if rerank else None,
            "visual_model": ids["visual"] if visual else None,
            "visual_comparator": ("COSINE" if self._gemini() else "MAX_SIM") if visual else None,
            "visual_strategy": ("dense_cosine" if self._gemini() else "colqwen_maxsim") if visual else None,
            "embedding_dimensions": self._dimensions(), "text_collection": self.text_collection,
            "visual_collection": self.visual_collection if visual else None}

    def _check(self):
        if getattr(self, 'check_cancel', None):
            self.check_cancel()

    def _progress(self, completed, total, unit):
        if getattr(self, 'report_progress', None):
            self.report_progress(completed, total, unit)

    def prepare_queries(self, queries, profile):
        """Optional worker optimization: batch lightweight stages before repeated reranking."""
        if profile not in PROFILES:
            raise ValueError("Unknown retrieval profile")
        unique = list(dict.fromkeys(queries))
        missing = [q for q in unique if q not in self._query_embeddings]
        total = len(unique) * (2 if profile == "multimodal" else 1)
        completed = len(unique) - len(missing)
        if profile == "multimodal":
            completed += sum(q in self._visual_queries for q in unique)
        self._progress(completed, total, "query encodings")
        # The inference API accepts at most 128 texts per request. A full tender can
        # have up to 300 candidate clauses, so preparation must respect that bound.
        for offset in range(0, len(missing), 128):
            batch = missing[offset:offset + 128]
            self._check()
            vectors = self._model_call("embed", batch, is_query=True)
            if len(vectors) != len(batch):
                raise RetrievalFailure("Embedding service returned incomplete query vectors")
            self._query_embeddings.update(zip(batch, [self._dense_vector(v) for v in vectors]))
            completed += len(batch)
            self._progress(completed, total, "query encodings")
        if profile == "multimodal" and self._gemini():
            missing_visual = [query for query in unique if query not in self._visual_queries]
            for offset in range(0, len(missing_visual), 128):
                batch = missing_visual[offset:offset + 128]
                self._check()
                vectors = self._model_call("visual_queries", batch)
                if len(vectors) != len(batch):
                    raise RetrievalFailure("Embedding service returned incomplete visual query vectors")
                self._visual_queries.update(zip(batch, [self._visual_vector(v) for v in vectors]))
                completed += len(batch)
                self._progress(completed, total, "query encodings")
        elif profile == "multimodal":
            for query in unique:
                if query not in self._visual_queries:
                    self._check()
                    self._visual_queries[query] = self._visual_vector(self._model_call("visual_query", query))
                    completed += 1
                    self._progress(completed, total, "query encodings")
        self._check()
        return {"text_queries": len(unique), "visual_queries": len(unique) if profile == "multimodal" else 0}

    def health(self):
        self.client.get_collections()
        return True

    def _collection_configs(self, visual=False):
        from qdrant_client import models as q
        configs = [(self.text_collection, q.VectorParams(size=self._dimensions(), distance=q.Distance.COSINE))]
        if visual:
            config = q.VectorParams(size=self._dimensions(visual=True), distance=q.Distance.COSINE)
            if not self._gemini():
                config.multivector_config = q.MultiVectorConfig(comparator=q.MultiVectorComparator.MAX_SIM)
                config.hnsw_config = q.HnswConfigDiff(m=0)
            configs.append((self.visual_collection, config))
        return configs

    def _validate_collection(self, name, expected):
        actual = self.client.get_collection(name).config.params.vectors
        if (isinstance(actual, dict) or actual is None or actual.size != expected.size
                or actual.distance != expected.distance
                or actual.multivector_config != expected.multivector_config):
            raise RetrievalFailure(f"Incompatible vector collection {name}: dimensions, distance or multivector configuration differ")

    def _ensure(self, visual=False):
        from qdrant_client import models as q
        for name, config in self._collection_configs(visual):
            self._check()
            if not self.client.collection_exists(name):
                self.client.create_collection(name, vectors_config=config)
                for field in ("workspace_id", "tender_id", "document_id", "version"):
                    self.client.create_payload_index(name, field_name=field, field_schema=q.PayloadSchemaType.KEYWORD)
            else:
                self._validate_collection(name, config)

    def _scope(self, tender_id, document_ids, versions=None):
        from qdrant_client import models as q
        if not tender_id or not document_ids:
            raise ValueError("Tender and document allowlist are mandatory")
        must = [q.FieldCondition(key="workspace_id", match=q.MatchValue(value=self.workspace_id)),
            q.FieldCondition(key="tender_id", match=q.MatchValue(value=tender_id)),
            q.FieldCondition(key="document_id", match=q.MatchAny(any=list(document_ids)))]
        if versions is not None:
            if not versions:
                raise ValueError("Document/version allowlist is mandatory")
            must.append(q.Filter(should=[q.Filter(must=[
                q.FieldCondition(key="document_id", match=q.MatchValue(value=doc_id)),
                q.FieldCondition(key="version", match=q.MatchValue(value=str(version)))])
                for doc_id, version in versions]))
        return q.Filter(must=must)

    def _scroll(self, collection, query_filter):
        result, offset = [], None
        while True:
            self._check()
            points, offset = self.client.scroll(collection, scroll_filter=query_filter, limit=256,
                offset=offset, with_payload=True, with_vectors=False)
            result.extend({"point_id": str(p.id), **p.payload} for p in points)
            if offset is None:
                return result

    def index(self, tender_id, documents, object_store, profile):
        from qdrant_client import models as q
        if profile not in PROFILES:
            raise ValueError("Unknown retrieval profile")
        started = time.perf_counter()
        self._search_scopes = {}
        self._check()
        self._ensure(profile == "multimodal")
        passages, visual_pages = [], []
        for doc in documents:
            if doc.get("tender_id", tender_id) != tender_id:
                raise RetrievalFailure("Document belongs to a different tender")
            if not doc.get("id") or doc.get("version") is None:
                raise RetrievalFailure("Document id and immutable version are required")
            pages = doc.get("pages") or [{"page": 1, "text": doc.get("text", "")}]
            for page in pages:
                payload = {"workspace_id": self.workspace_id, "tender_id": tender_id,
                    "document_id": doc["id"], "version": str(doc["version"]), "page": page["page"],
                    "title": doc["title"], "role": doc.get("role", "evidence"),
                    "source_url": doc.get("source_url"), "logical_id": doc.get("logical_id"),
                    "sha256": doc.get("sha256", doc.get("raw_sha256")), "image_key": page.get("image_key")}
                for chunk, text in enumerate(chunk_page(page.get("text", ""))):
                    key = str(uuid.uuid5(uuid.NAMESPACE_URL,
                        f'{self.workspace_id}/{tender_id}/{doc["id"]}/{doc["version"]}/{page["page"]}/text/{chunk}'))
                    passages.append({"point_id": key, **payload, "text": text, "chunk": chunk})
                if profile == "multimodal":
                    if not page.get("image_key"):
                        raise RetrievalFailure(f'Page {page["page"]} of {doc["title"]} has no rendered image; multimodal indexing cannot downgrade')
                    key = str(uuid.uuid5(uuid.NAMESPACE_URL,
                        f'{self.workspace_id}/{tender_id}/{doc["id"]}/{doc["version"]}/{page["page"]}/image'))
                    visual_pages.append({"point_id": key, **payload, "text": page.get("text", "")})
        pending_text = self._missing_points(self.text_collection, passages) if self._modern_dense() else passages
        pending_pages = self._missing_points(self.visual_collection, visual_pages, visual=True) if self._gemini() else visual_pages
        reused_text, reused_pages = len(passages) - len(pending_text), len(visual_pages) - len(pending_pages)
        completed, total = reused_text + reused_pages, len(passages) + len(visual_pages)
        self._progress(completed, total, "index items")
        for offset in range(0, len(pending_text), 16):
            batch = pending_text[offset:offset + 16]
            self._check()
            vectors = self._model_call("embed", [p["text"] for p in batch])
            if len(vectors) != len(batch):
                raise RetrievalFailure("Embedding service returned incomplete vectors")
            vectors = [self._dense_vector(v) for v in vectors]
            self.client.upsert(self.text_collection, [q.PointStruct(id=p["point_id"], vector=v, payload=p)
                for p, v in zip(batch, vectors)], wait=True)
            completed += len(batch)
            self._progress(completed, total, "index items")
        # All text is embedded first, then all images: no interleaved heavyweight loads.
        page_batch_size = getattr(self.config, "gemini_max_concurrency", 8) if self._gemini() else 1
        for offset in range(0, len(pending_pages), page_batch_size):
            batch = pending_pages[offset:offset + page_batch_size]
            self._check()
            raws = [object_store.get(page["image_key"]) for page in batch]
            self._check()
            vectors = self._model_call("visual_embed", raws)
            if len(vectors) != len(batch):
                raise RetrievalFailure("Visual embedding service returned incomplete page vectors")
            vectors = [self._visual_vector(vector) for vector in vectors]
            self.client.upsert(self.visual_collection, [q.PointStruct(id=page["point_id"], vector=vector, payload=page)
                for page, vector in zip(batch, vectors)], wait=True)
            completed += len(batch)
            self._progress(completed, total, "index items")
        self._check()
        return {"text_passages": len(passages), "visual_pages": len(visual_pages),
            "reused_text_passages": reused_text, "reused_visual_pages": reused_pages,
            "documents": len(documents), "profile": profile, "seconds": round(time.perf_counter() - started, 3),
            **self._retrieval_metadata(visual=bool(visual_pages)),
            "text_model": self._model_ids()["embedding"] if passages else None}

    def _missing_points(self, collection, points, *, visual=False):
        """Reuse only exact immutable payloads with valid vectors in this model's collection."""
        missing = []
        for offset in range(0, len(points), 128):
            self._check()
            batch = points[offset:offset + 128]
            existing = {str(point.id): point for point in self.client.retrieve(collection,
                ids=[point["point_id"] for point in batch], with_payload=True, with_vectors=True)}
            for expected in batch:
                point = existing.get(expected["point_id"])
                if point is not None and point.payload == expected:
                    try:
                        self._dense_vector(point.vector, visual=visual)
                        continue
                    except RetrievalFailure:
                        pass
                missing.append(expected)
        return missing

    def _search_scope_key(self, tender_id, document_ids, profile):
        return (self.workspace_id, tender_id, tuple(document_ids), profile,
                self.text_collection, self.visual_collection if profile == "multimodal" else None)

    def prepare_search_scope(self, tender_id, document_ids, profile):
        """Freeze this run's authorized source scope before concurrent read-only searches."""
        query_filter, scoped = self._load_search_scope(tender_id, document_ids, profile)
        scopes = getattr(self, "_search_scopes", {})
        scopes[self._search_scope_key(tender_id, document_ids, profile)] = (query_filter, tuple(scoped))
        self._search_scopes = scopes
        return {"passages": len(scoped)}

    def _load_search_scope(self, tender_id, document_ids, profile):
        self._check()
        if profile not in PROFILES:
            raise ValueError("Unknown retrieval profile")
        if not document_ids:
            return None, []
        if not self.client.collection_exists(self.text_collection):
            raise RetrievalFailure("Text index has not been created")
        for name, expected in self._collection_configs(profile == "multimodal"):
            if not self.client.collection_exists(name):
                raise RetrievalFailure("Multimodal index missing; cannot downgrade")
            self._validate_collection(name, expected)
        # Resolve immutable document snapshot versions inside the tenant/tender boundary.
        initial_scope = self._scope(tender_id, document_ids)
        scoped = self._scroll(self.text_collection, initial_scope)
        visual_scoped = []
        if profile == "multimodal":
            if not self.client.collection_exists(self.visual_collection):
                raise RetrievalFailure("Multimodal index missing; cannot downgrade")
            visual_scoped = self._scroll(self.visual_collection, initial_scope)
        versions = {}
        for passage in scoped + visual_scoped:
            versions.setdefault(passage["document_id"], set()).add(passage["version"])
        if any(len(v) != 1 for v in versions.values()):
            raise RetrievalFailure("Ambiguous document versions; document IDs must identify immutable snapshots")
        if not scoped and not visual_scoped:
            return None, []
        pairs = [(key, next(iter(value))) for key, value in versions.items()]
        query_filter = self._scope(tender_id, document_ids, pairs)
        # Lexical candidates also traverse the exact final version filter.
        scoped = self._scroll(self.text_collection, query_filter)
        return query_filter, scoped

    def search(self, tender_id, query, document_ids, profile, limit=5):
        self._check()
        prepared = getattr(self, "_search_scopes", {}).get(self._search_scope_key(tender_id, document_ids, profile))
        query_filter, scoped = prepared if prepared is not None else self._load_search_scope(tender_id, document_ids, profile)
        if query_filter is None:
            return []
        by_id = {p["point_id"]: dict(p) for p in scoped}
        count = max(limit * 3, 10)
        vector = self._query_embeddings.get(query)
        if vector is None:
            self._check()
            vectors = self._model_call("embed", [query], is_query=True)
            if len(vectors) != 1:
                raise RetrievalFailure("Embedding service returned incomplete query vectors")
            vector = self._dense_vector(vectors[0])
            self._query_embeddings[query] = vector
        semantic = self.client.query_points(self.text_collection, query=vector, query_filter=query_filter,
            limit=count, with_payload=True).points
        rankings = [[(str(p.id), float(p.score)) for p in semantic]]
        for point in semantic:
            by_id[str(point.id)] = {"point_id": str(point.id), **point.payload}
        if profile in {"hybrid", "multimodal"}:
            rankings.append(lexical_rank(query, scoped, count))
        if profile == "multimodal":
            if not self.client.collection_exists(self.visual_collection):
                raise RetrievalFailure("Multimodal index missing; cannot downgrade")
            visual_vector = self._visual_queries.get(query)
            if visual_vector is None:
                self._check()
                visual_vector = self._visual_vector(self._model_call("visual_query", query))
                self._visual_queries[query] = visual_vector
            visual = self.client.query_points(self.visual_collection, query=visual_vector,
                query_filter=query_filter, limit=count, with_payload=True).points
            if not visual:
                raise RetrievalFailure("No authorized visual pages indexed for selected documents")
            visual_ranking = []
            for point in visual:
                # Fuse image-page scores into text chunks on that exact same source page.
                matches = [p["point_id"] for p in scoped if p["document_id"] == point.payload["document_id"]
                    and p["version"] == point.payload["version"] and p["page"] == point.payload["page"]]
                if not matches:
                    by_id[str(point.id)] = {"point_id": str(point.id), **point.payload}
                    matches = [str(point.id)]
                visual_ranking.extend((key, float(point.score)) for key in matches)
            rankings.append(visual_ranking)
        ranked = fuse_rankings(rankings) if profile != "text" else rankings[0]
        candidates = [{**by_id[key], "retrieval_score": score} for key, score in ranked[:count]]
        mlx = getattr(getattr(self, "config", None), "inference_provider", "local") == "mlx"
        method = "gemini_embedding" if self._gemini() else "bge_embedding_cpu" if mlx else "qwen3_embedding"
        if profile in {"hybrid", "multimodal"}:
            self._check()
            reranked = self._model_call("rerank", query, [p["text"] for p in candidates])
            if len(reranked) != len(candidates):
                raise RetrievalFailure("Reranker returned incomplete scores")
            for passage, score in zip(candidates, reranked):
                passage["score"] = score
            candidates.sort(key=lambda p: (-p["score"], -p["retrieval_score"], p["point_id"]))
            method += "+bm25+rrf+" + ("gemini_relevance_scoring" if self._gemini() else "bge_cosine_cpu" if mlx else "qwen3_reranker")
            if profile == "multimodal":
                method += "+gemini_page_cosine" if self._gemini() else "+colqwen2_maxsim"
        else:
            for passage in candidates:
                passage["score"] = passage["retrieval_score"]
        metadata = self._retrieval_metadata(visual=profile == "multimodal", rerank=profile != "text")
        return [{**p, "method": method, **metadata} for p in candidates[:limit]]
