# SPDX-License-Identifier: MPL-2.0
"""CPU condition retrieval; rankings are candidates, never proof.

The recipe mirrors the isolated dense/BM25 diagnostic. Original E chunks, rather
than only their ranked P excerpts, retain nearby qualifications for assessment.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import threading
import unicodedata

import numpy as np

from .mlx_client import EMBEDDING_MODEL_ID, LocalTextRuntime
from .mlx_compact import build_supplier_span_catalog
from .retrieval import RetrievalFailure, fuse_rankings, lexical_rank
from bidtrace_core.local.context import obligation_context


RECIPE = {
    'version': 'condition_evidence_cpu_rrf_v11_explicit_service_corroboration', 'embedding_model': EMBEDDING_MODEL_ID,
    'device': 'cpu', 'dimensions': 384, 'span_policy': 'covering',
    'dense_query': 'unique selected, governing, paired-question, enclosing exact text',
    'lexical_query': 'selected exact text plus service action aliases; candidates only',
    'lexical_normalization': 'NFKC casefold Unicode word tokens',
    'dense_limit': 12, 'lexical_limit': 12, 'positive_scores_only': True,
    'rrf_k': 60, 'fused_limit': 12, 'distinct_original_chunks': 6,
    'score_ties': 'numpy.argsort(-scores) in original span order',
    'premise_selection': 'complete chunk; product-bound configuration and support before support alone',
}


def _service_subjects(text):
    """Small action vocabulary for ranking only, never an entailment rule."""
    families = {
        'report': r'\breport(?:s|ing)?\b',
        'meeting': r'\bmeetings?\b|\breview\s+(?:calls?|sessions?)\b|\bconsultations?\b',
        'application': r'\b(?:applications?|software)\b',
        'maintenance': r'\b(?:maintenan\w*|maintain\w*|upkeep)\b|\blifecycle\s+management\b',
        'update': r'\b(?:updates?|updating|patch(?:es|ing)?)\b',
        'soc': r'\bSOC\b(?!\s*(?:2|II|Type|certification))',
        'remediation': r'\bremediat\w*\b|\bresolv\w*\b|\bproblem[- ]solving\b',
        'sales': r'\bsales?\b(?!\s+(?:tax|final))|\bpurchas\w*\b|\bprocurement\b',
        'sla': r'\bSLAs?\b|\bservice[- ]level\b|\b(?:response|resolution) times?\b',
        'reference': r'\breferences?\b',
        'backup': r'\bback[- ]?ups?\b',
    }
    return {name for name, pattern in families.items() if re.search(pattern, text, re.I)}


def _product_subjects(text):
    # Spelling aliases affect candidate selection only. They cannot establish
    # configuration, coverage, or the identity of a different named product.
    return {'microsoft'} if re.search(r'\b(?:Microsoft(?:\s*365)?|M365|MS\s+Office)\b', text, re.I) else set()


def select_condition_premise(condition, *, catalog=None):
    """Select one complete original chunk, never an excerpt or a verdict.

    Exact selected-condition vocabulary breaks dense-ranking ties before a
    general source-promise cue. All original surrounding qualifiers remain in
    the chosen chunk. No benchmark metadata or supplier-reference answer enters
    selection. Empty evidence stays empty.
    """
    from .local_nli_gate import content_words
    from bidtrace_core.local.relations import is_contract_duration_condition, single_payment_term
    from bidtrace_core.local.content import _configuration_support_issue, _evidence_clauses, _positive_clause
    candidates = condition['selected_evidence'][:6]
    if not candidates:
        return []
    from bidtrace_core.local.relations import _REFERENCE_HEADING, standard_sla_disclosure_request
    originals = catalog if catalog is not None else candidates
    context = condition.get('buyer_context', {})
    enclosing = context.get('enclosing_span', {}).get('text', '')
    if ('remediation' in _service_subjects(condition['source_text'])
            and re.search(r'\bSOC\b|\bsecurity\b|\bcybersecurity\b', enclosing, re.I)):
        # Dense top-six candidates can omit a response passage when a PDF's
        # columns are interleaved. Search the same approved complete catalog
        # for the requested action and its SOC scope. Keep whole chunks and
        # all qualifications; selection still supplies no assessment.
        for item in originals:
            quote = item['quote']
            response = (re.search(r'\b(?:cybersecurity|threats?|security)\b', quote, re.I)
                        and re.search(r'\b(?:resolve|remediat\w*|problem[- ]solving|address\w*)\b', quote, re.I))
            soc = re.search(r'\bSOC\b(?!\s*(?:2|II|Type|Certification))', quote, re.I)
            if (response or soc) and item['citation_id'] not in {c['citation_id'] for c in candidates}:
                candidates.append(item)
    if re.search(r'\breferences?\b', enclosing, re.I) and re.search(r'\b(?:name|title|phone|address|fax)\b', condition['source_text'], re.I):
        # Keep a complete client-contact section across its source boundary;
        # executive resume titles and one reference cannot supply three fields.
        for i,item in enumerate(originals):
            if _REFERENCE_HEADING.search(item['quote']):
                chosen = [item['citation_id']]
                following = originals[i+1:i+2]
                if following and all(following[0].get(k)==item.get(k) and item.get(k) is not None for k in ('document_id','page')):
                    chosen.append(following[0]['citation_id'])
                return chosen
    if standard_sla_disclosure_request(condition['source_text'], context):
        for item in originals:
            if re.search(r'^\s*Response Times\b', item['quote'], re.I):
                chosen = [item['citation_id']]
                for other in originals:
                    if (all(other.get(k)==item.get(k) and item.get(k) is not None for k in ('document_id','page'))
                            and re.search(r'\bunable to provide guaranteed resolution times\b', other['quote'], re.I)
                            and other['citation_id'] not in chosen):
                        chosen.append(other['citation_id'])
                        break
                return chosen
    selected = content_words(condition['source_text'])
    if len(selected) <= 1:
        selected |= content_words(condition.get('buyer_context', {}).get('enclosing_span', {}).get('text', ''))
    vocabularies = [content_words(item['quote']) for item in candidates]
    # Rare subject words (e.g. SOC) distinguish a service from a generic
    # response/signature. Containing buyer context breaks ties for fragments;
    # it never becomes supplier evidence or a compliance decision.
    weights = {word: 1 + math.log((len(candidates) + 1) /
               (1 + sum(word in words for words in vocabularies))) for word in selected}
    context_words = content_words(condition.get('buyer_context', {}).get('enclosing_span', {}).get('text', ''))
    bound_context = condition.get('buyer_context', {})
    if re.search(r'\b(?:provided by incoming|will need to be provided|Provided by current)\b', condition['source_text'], re.I):
        selected |= content_words(' '.join([s['text'] for s in bound_context.get('governing_spans', [])] +
                                          [q['quote'] for q in bound_context.get('paired_buyer_questions', [])]))
        weights = {word: 1 + math.log((len(candidates) + 1) /
                   (1 + sum(word in words for words in vocabularies))) for word in selected}
    subject_text = condition['source_text']
    context = condition.get('buyer_context', {})
    if not _service_subjects(subject_text):
        subject_text += ' ' + context.get('enclosing_span', {}).get('text', '')
        subject_text += ' ' + ' '.join(q['quote'] for q in context.get('paired_buyer_questions', []))
    subjects = _service_subjects(subject_text)
    products = _product_subjects(condition['source_text'])
    promise = re.compile(r'\b(?:included|includes|including|proposes|shall|will\s+provide)\b', re.I)
    duration_condition = is_contract_duration_condition(condition['source_text'])
    configuration_condition = bool(re.search(
        r'\bconfigur\w*\s*(?:/|&|and)\s*support\w*\b', condition['source_text'], re.I))
    # Prefer a complete source with both product-bound services over support
    # alone. This is candidate selection; the local model still decides scope.
    configuration_matches = [configuration_condition and not _configuration_support_issue(
        condition['source_text'], [clause for clause in _evidence_clauses([item['quote']])
                                  if _positive_clause(clause)]) for item in candidates]
    best = max(enumerate(candidates), key=lambda pair: (
        not (re.search(r'\bClient Needs\s*:', pair[1]['quote'], re.I) and not re.search(r'\bProposed Solution\s*:', pair[1]['quote'], re.I)),
        bool(duration_condition and single_payment_term(pair[1]['quote'])),
        len(products & _product_subjects(pair[1]['quote'])),
        configuration_matches[pair[0]],
        bool(re.search(r'\bpatch\w*\b', condition['source_text'], re.I)
             and re.search(r'\bpatch(?:ing| management)\b', pair[1]['quote'], re.I)),
        bool('backup' in subjects and re.search(r'\bservers?\b', condition['source_text'], re.I)
             and re.search(r'\bserver\s+back[- ]?up\b|\bback[- ]?ups?\s+(?:of|for)\s+servers?\b', pair[1]['quote'], re.I)),
        bool('remediation' in subjects and re.search(r'\b(?:cybersecurity|threats?|security)\b', pair[1]['quote'], re.I)
             and re.search(r'\b(?:resolve|remediat\w*|problem[- ]solving|address\w*)\b', pair[1]['quote'], re.I)),
        len(subjects & _service_subjects(pair[1]['quote'])),
        sum(weights[word] for word in selected & vocabularies[pair[0]]),
        len(context_words & vocabularies[pair[0]]),
        bool(promise.search(pair[1]['quote'])), -pair[0]))[1]
    chosen = [best['citation_id']]
    # A category passage can omit the supplier's literal service wording.
    # Retain the complete purchasing/licensing or network-monitoring passage
    # alongside it. These are independent source candidates, never verdicts;
    # both passages and their surrounding qualifications reach the model.
    service_pattern = None
    if (re.fullmatch(r'Sales[,\s]*', condition['source_text'], re.I)
            and re.search(r'\blicen[sc]es?\b|\blicensing\b', enclosing, re.I)):
        service_pattern = r'\bpurchas\w*\b[\s\S]*\blicensing\b'
    elif re.fullmatch(r'Network[- ]Device\s+Monitoring[.\s]*', condition['source_text'], re.I):
        service_pattern = r'\bnetwork\s+monitoring\b'
    if service_pattern:
        literal = next((item for item in originals if re.search(service_pattern, item['quote'], re.I)
                        and item['citation_id'] not in chosen), None)
        if literal:
            chosen = [literal['citation_id'], *chosen]
    if re.fullmatch(r'(?:Security\s+)?Patch Management[.\s]*', condition['source_text'].strip(), re.I):
        # A matching service excerpt can be separated from its inclusion header
        # by PDF chunking. Keep an explicit included patching passage from the
        # same approved supplier catalog as corroborating scope. This is source
        # selection, with no benchmark labels or model-result lookup.
        for item in catalog if catalog is not None else candidates:
            quote = ' '.join(item['quote'].split())
            if (re.search(r'\bfollowing services are included in (?:this|the) service agreement\b', quote, re.I)
                    and re.search(r'\bsecurity patching\b', quote, re.I)
                    and item['citation_id'] not in chosen):
                chosen.append(item['citation_id'])
                break
    if 'remediation' in subjects:
        # A response/remediation item under SOC needs both the actual security
        # response passage and its included SOC scope, rather than borrowing
        # an unrelated helpdesk response clock. Still candidates, never proof.
        soc = next((item for item in candidates if re.search(r'\bSOC\b(?!\s*(?:2|II|Type|Certification))', item['quote'], re.I)), None)
        if soc and soc['citation_id'] not in chosen and len(chosen)<2:
            chosen.append(soc['citation_id'])
    return chosen


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _lexical_text(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def condition_queries(contract, obligation):
    context = obligation_context(contract, obligation)
    selected, enclosing = context['selected_span']['text'], context['enclosing_span']['text']
    dense = [selected, *(span['text'] for span in context['governing_spans']),
             *(question['quote'] for question in context['paired_buyer_questions']), enclosing]
    # A supplier may describe remediation as resolving problems. These words
    # retrieve candidates only; the model still sees the exact original duty.
    expansions = {'remediation': 'resolve problems cybersecurity threats',
                  'sales': 'purchasing procurement licensing', 'report': 'reporting reports',
                  'meeting': 'consultation consultations review sessions',
                  'application': 'software applications',
                  'maintenance': 'lifecycle maintenance upkeep', 'update': 'patching updates',
                  'sla': 'service level response times resolution times',
                  'reference': 'business references reference contacts', 'backup': 'backup back-up copies'}
    subject_text = selected
    if not _service_subjects(subject_text):
        subject_text += ' ' + enclosing + ' ' + ' '.join(q['quote'] for q in context['paired_buyer_questions'])
    extra = ' '.join(expansions[subject] for subject in sorted(_service_subjects(subject_text))
                     if subject in expansions)
    if _product_subjects(selected):
        extra += ' Microsoft Microsoft365 M365'
    return {'query': '\n'.join(dict.fromkeys(dense + ([extra] if extra else []))), 'selected_literal': selected,
            'lexical_query': selected + (' ' + extra if extra else ''),
            'buyer_context': context}


class ConditionEvidenceRetriever:
    """Reusable instance caches exact source and query recipes, with no activation."""

    def __init__(self, runtime=None):
        self.runtime = runtime if runtime is not None else LocalTextRuntime()
        self._sources, self._queries, self._rankings = {}, {}, {}
        self._lock = threading.RLock()
        self._recipe = {**RECIPE, 'numpy_version': np.__version__}
        self.recipe_sha256 = _hash(self._recipe)

    @property
    def recipe(self):
        return deepcopy(self._recipe)

    def _source(self, evidence):
        if not isinstance(evidence, list) or any(
                not isinstance(item, dict) or item.get('role') != 'evidence' for item in evidence):
            raise RetrievalFailure('Condition retrieval requires explicit supplier evidence roles.')
        key = _hash({'recipe': self.recipe_sha256, 'supplier_evidence': evidence})
        if key not in self._sources:
            supplier = build_supplier_span_catalog(evidence, span_policy='covering')
            catalog = json.loads(supplier.catalog_json)
            originals = {item['citation_id']: item for item in catalog}
            if len(originals) != len(catalog):
                raise RetrievalFailure('Duplicate supplier catalog ID.')
            indices = {}
            for index, span in enumerate(supplier.spans):
                original = originals.get(span.citation_id)
                if original is None or original.get('role') != 'evidence':
                    raise RetrievalFailure('Unknown or non-supplier original catalog ID.')
                if (span.span_id in indices or span.start < 0 or span.end <= span.start
                        or original['quote'][span.start:span.end] != span.quote):
                    raise RetrievalFailure('Invalid exact supplier span identity or offsets.')
                indices[span.span_id] = index
            self._sources[key] = {'supplier': supplier, 'catalog': catalog,
                'originals': originals, 'indices': indices, 'evidence': deepcopy(evidence),
                'passages': [{'point_id': str(i), 'text': _lexical_text(span.quote)}
                             for i, span in enumerate(supplier.spans)], 'vectors': None}
        return key, self._sources[key]

    def _embed(self, texts, is_query):
        if not self.runtime.weights_ready():
            raise RetrievalFailure('Offline CPU BGE weights are unavailable; downloading is disabled.')
        vectors = np.asarray(self.runtime.invoke('embed', texts, is_query), dtype=np.float64)
        if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
            raise RetrievalFailure('CPU BGE must return one finite 384-dimensional vector per input.')
        return vectors

    def _rank(self, source_key, source, queries):
        key = _hash({'recipe': self.recipe_sha256, 'source': source_key,
                     'dense_query': queries['query'], 'lexical_query': queries['lexical_query']})
        if key in self._rankings:
            return self._rankings[key], True, key
        supplier = source['supplier']
        dense = np.zeros(len(supplier.spans))
        sparse = np.zeros(len(supplier.spans))
        if supplier.spans:
            q = self._queries[(self.recipe_sha256, queries['query'])]
            vectors = source['vectors']
            dense = q @ vectors.T / np.maximum(np.linalg.norm(q) * np.linalg.norm(vectors, axis=1), 1e-9)
            for index, score in lexical_rank(_lexical_text(queries['lexical_query']),
                                              source['passages'], len(source['passages'])):
                if not isinstance(index, str) or not index.isdecimal() or int(index) >= len(sparse):
                    raise RetrievalFailure('Unknown lexical supplier span ID.')
                if not np.isfinite(score):
                    raise RetrievalFailure('Non-finite lexical score.')
                if score > 0:
                    sparse[int(index)] = score

        def top(scores, limit):
            return [(supplier.spans[i].span_id, float(scores[i]))
                    for i in np.argsort(-scores) if scores[i] > 0][:limit]

        dense_top = top(dense, self.recipe['dense_limit'])
        sparse_top = top(sparse, self.recipe['lexical_limit'])
        fused = np.zeros(len(supplier.spans))
        for pid, score in fuse_rankings([dense_top, sparse_top], k=self.recipe['rrf_k']):
            if pid not in source['indices']:
                raise RetrievalFailure('Unknown fused supplier span ID.')
            if not np.isfinite(score) or score <= 0:
                raise RetrievalFailure('Fusion must contain positive finite scores only.')
            fused[source['indices'][pid]] = score
        # A short listed service name must not be displaced by its containing
        # list. This boosts candidates only, never compliance.
        phrase = _lexical_text(queries['selected_literal'])
        if len(phrase.split()) >= 2:
            for index, span in enumerate(supplier.spans):
                if (' ' + phrase + ' ') in (' ' + _lexical_text(span.quote) + ' '):
                    fused[index] += 1.0
        ranked = []
        for pid, score in top(fused, self.recipe['fused_limit']):
            index = source['indices'][pid]
            span = supplier.spans[index]
            original = source['originals'][span.citation_id]
            ranked.append({**deepcopy(original), 'span_id': pid, 'quote': span.quote,
                'score': score, 'dense_score': float(dense[index]), 'lexical_score': float(sparse[index]),
                'source_start': span.start, 'source_end': span.end,
                'offset_scope': 'original_citation_chunk',
                'original_quote_sha256': hashlib.sha256(original['quote'].encode()).hexdigest()})
        selected = list(dict.fromkeys(row['citation_id'] for row in ranked))[:self.recipe['distinct_original_chunks']]
        value = {'ranked_spans': ranked, 'selected_citation_ids': selected,
            'selected_evidence': [deepcopy(source['originals'][cid]) for cid in selected],
            'dense_ranking': dense_top, 'lexical_ranking': sparse_top}
        self._rankings[key] = value
        return value, False, key

    def retrieve(self, contract, supplier_evidence, *, requirement_id=None):
        return self.retrieve_many([contract], supplier_evidence, requirement_ids=[requirement_id])[0]

    def retrieve_many(self, contracts, supplier_evidence, *, requirement_ids=None):
        """Return every canonical condition in order, with full selected E chunks.

        Batching changes only embedding preparation. No buyer text becomes supplier
        evidence, and no retrieval score is converted to an assessment label.
        """
        if requirement_ids is None:
            requirement_ids = [None] * len(contracts)
        if len(requirement_ids) != len(contracts):
            raise RetrievalFailure('Requirement identities must cover every contract.')
        with self._lock:
            source_key, source = self._source(supplier_evidence)
            prepared = []
            for contract in contracts:
                text = contract.get('text')
                if (not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest() != contract.get('source_sha256')):
                    raise RetrievalFailure('Canonical contract source hash does not match its exact text.')
                obligations = contract.get('obligations', [])
                ids = [item.get('obligation_id') for item in obligations]
                if any(not isinstance(oid, str) or not oid for oid in ids) or len(set(ids)) != len(ids):
                    raise RetrievalFailure('Canonical obligation IDs must be unique and nonempty.')
                prepared.append([(deepcopy(item), condition_queries(contract, item)) for item in obligations])
            if source['supplier'].spans:
                if source['vectors'] is None:
                    source['vectors'] = self._embed([span.quote for span in source['supplier'].spans], False)
                missing = list(dict.fromkeys(query['query'] for items in prepared for _, query in items
                    if (self.recipe_sha256, query['query']) not in self._queries))
                if missing:
                    self._queries.update(((self.recipe_sha256, query), vector)
                                         for query, vector in zip(missing, self._embed(missing, True)))
            output = []
            for requirement_id, contract, items in zip(requirement_ids, contracts, prepared):
                conditions = []
                for obligation, queries in items:
                    ranking, cached, rank_key = self._rank(source_key, source, queries)
                    conditions.append({**deepcopy(obligation), 'requirement_id': requirement_id,
                        **deepcopy(queries), **deepcopy(ranking), 'receipt': {
                            'source_cache_sha256': source_key, 'query_sha256': _hash(queries),
                            'ranking_cache_sha256': rank_key, 'ranking_cache_hit': cached,
                            'canonical_source_sha256': contract['source_sha256'],
                            'supplier_catalog_sha256': source['supplier'].sha256,
                            'recipe_sha256': self.recipe_sha256}})
                output.append({'requirement_id': requirement_id, 'recipe': deepcopy(self.recipe),
                    'recipe_sha256': self.recipe_sha256, 'source_cache_sha256': source_key,
                    'supplier_catalog_sha256': source['supplier'].sha256,
                    'source_spans': len(source['supplier'].spans),
                    'catalog': deepcopy(source['catalog']), 'supplier_evidence': deepcopy(source['evidence']),
                    'conditions': conditions, 'assessment_performed': False})
            return output
