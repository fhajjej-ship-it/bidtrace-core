# SPDX-License-Identifier: MPL-2.0
"""Checkpointed roles: requirements analyst, evidence analyst and verifier."""
from __future__ import annotations

import copy
from hashlib import sha256
import json
import os
import time
from threading import RLock
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from bidtrace.engine import analyze_documents, requirement_source_context
from bidtrace.store import counts, digest, event, now
from bidtrace_core.contracts import MODELS, PIPELINE_VERSION, VALIDATION_REASONS, JobCancelled, LeaseLostError
from bidtrace_core.sources import selected_sources
from .identity import source_fingerprint, execution_signature
from . import models as model_contracts
from .models import ModelClient, ModelFailure, validate_obligation_proposal
from .providers import create_model_client, model_ids
from .obligations import decompose_requirement, OBLIGATIONS_PER_BATCH
from .retrieval import QdrantRetriever, bounded_supplier_context
from .clauses import analyze_structured_requirements
from .segments import bound_requirements
from .buyer_references import (enrich_buyer_references, reference_targets, validate_buyer_reference,
                               REVIEW_ISSUES as REFERENCE_REVIEW_ISSUES)
from .buyer_scope import flag_buyer_scope_conflicts
from .parallel import WorkAdmission, bounded_map


class ReviewState(TypedDict, total=False):
    tender_id: str
    profile: str
    documents: list[dict]
    requirements: list[dict]
    previous_requirements: list[dict]
    retrieved: dict
    steps: list[dict]
    model_stats: dict
    metrics: dict


def extract_requirements(documents):
    result = []
    supplier = [d for d in documents if d["role"] == "evidence"]
    for current in [d for d in documents if d["role"] == "current"]:
        if not current.get("text", "").strip():
            raise ValueError(f"Buyer document has no readable text: {current['title']}. Inspect its page images and OCR coverage.")
        previous = next((d for d in documents if d["role"] == "previous" and d["logical_id"] == current["logical_id"]), None)
        items = analyze_structured_requirements(previous, current, supplier)
        if items is None:
            items = analyze_documents(previous, current, supplier)
        items = bound_requirements(items)
        for req in items:
            req["source_key"] = req["key"]
            req["key"] = current["logical_id"] + ":" + req["key"]
            req["id"] = "req_" + digest(req["key"])[:20]
            req.update(logical_id=current["logical_id"], source_document_id=current["id"],
                       source_title=current["title"], proposal=None, retrieval=[], stale=False,
                       initial_assessment=previous is None)
        result.extend(items)
    if len(result) > 1000:
        raise ValueError("This document pack contains more than 1000 candidate clauses, exceeding the local worker budget. Split by tender lot; no clauses were silently omitted.")
    if not result:
        raise ValueError("No requirement clauses could be extracted. Inspect the document's text and page coverage.")
    return flag_buyer_scope_conflicts(enrich_buyer_references(result, documents))


def normalized(text):
    return " ".join(text.split())


def buyer_input_page_receipts(documents):
    """Readable buyer pages traversed by extraction and source-context parsing.

    Missing/blank page text is explicit and never earns processing coverage.
    These receipts do not establish complete requirement discovery or image/OCR
    interpretation. Supplier pages belong to the separate retrieval receipt.
    """
    pages, excluded = [], []
    for doc in documents:
        if doc.get("role") not in {"current", "previous"}:
            continue
        if doc.get("role") == "current" and doc.get("active") is False:
            excluded.append({"document_id": doc["id"], "role": doc["role"],
                             "page": None, "reason": "inactive_buyer_document"})
            continue
        if not doc.get("pages"):
            excluded.append({"document_id": doc["id"], "role": doc["role"],
                             "page": None, "reason": "no_page_records_text_only_fallback_unverified"})
        for page in doc.get("pages", []):
            number = page.get("page") if isinstance(page, dict) else None
            identity = {"document_id": doc["id"], "page": number, "role": doc["role"]}
            if type(number) is not int or number < 1:
                excluded.append({**identity, "reason": "source_page_identity_unverified"})
            elif not isinstance(page.get("text"), str) or not page["text"].strip():
                excluded.append({**identity, "reason": "unreadable_or_empty_page_text"})
            else:
                pages.append(identity)
    return {"input_pages": pages, "excluded_input_pages": excluded,
            "input_page_scope": "Readable selected buyer page text passed through extraction and source-context parsing; excludes blank/unreadable pages and does not establish semantic requirement recall."}


def verified_buyer_context(requirement, documents):
    """Resolve explicit Q/A context within the selected buyer document versions."""
    context = requirement.get("buyer_context", [])
    if not isinstance(context, list):
        raise ValueError("Buyer context must contain source citations.")
    lookup = {d["id"]: d for d in documents}
    result = []

    def verify_source(citation, *, cross_document=False):
        if not isinstance(citation, dict) or citation.get("role") not in {"current", "previous"}:
            raise ValueError("Buyer context must cite a selected buyer source.")
        doc = lookup.get(citation.get("document_id"))
        role = citation["role"]
        if (doc is None or doc.get("role") != role
                or (role == "current" and not doc.get("active", True))
                or (not cross_document and requirement.get("logical_id") and doc.get("logical_id") != requirement["logical_id"])
                or (not cross_document and role == "current" and requirement.get("source_document_id")
                    and doc["id"] != requirement["source_document_id"])):
            raise ValueError("Buyer context is outside this requirement's selected document versions.")
        page = next((p for p in doc["pages"] if p["page"] == citation.get("page")), None)
        quote = citation.get("quote")
        if not page or not isinstance(quote, str) or not quote.strip() or quote not in page["text"]:
            raise ValueError("Buyer context quotation does not exist on its cited source page.")

    for citation in context:
        if not isinstance(citation, dict) or citation.get("relation") not in {
                "paired_question", "referenced_question", "referenced_answer", "referenced_document"}:
            raise ValueError("Buyer context must be an explicitly paired or referenced buyer source.")
        cross_document = citation['relation'] == 'referenced_document'
        verify_source(citation, cross_document=cross_document)
        if citation["relation"] != "paired_question":
            origin = citation.get("reference_source")
            verify_source(origin)
            path = citation.get("reference_path")
            if ((not cross_document and origin["document_id"] != citation["document_id"]) or origin["role"] != citation["role"]
                    or origin.get("target") not in reference_targets(origin["quote"])
                    or not isinstance(path, list) or not path
                    or any(not isinstance(target, str) for target in path)
                    or citation.get("reference_target") not in path):
                raise ValueError("Buyer context reference origin or path is invalid.")
            validate_buyer_reference(requirement, documents, citation)
        result.append(copy.deepcopy(citation))

    conflicts = requirement.get("source_conflicts", [])
    if not isinstance(conflicts, list):
        raise ValueError("Buyer context conflict records must be a list.")
    for conflict in conflicts:
        if (not isinstance(conflict, dict) or conflict.get("kind") != "candidate_inventory_discrepancy"
                or not isinstance(conflict.get("observations"), list) or len(conflict["observations"]) < 2):
            raise ValueError("Buyer context conflict record is invalid.")
        for observation in conflict["observations"]:
            if not isinstance(observation, dict) or observation.get("role") != "current":
                raise ValueError("Buyer context conflict must use current buyer sources.")
            verify_source(observation, cross_document=True)
            citation = {**copy.deepcopy(observation), "relation": "scope_conflict",
                        "conflict_asset": conflict.get("asset")}
            if citation not in result:
                result.append(citation)
    return result


def requirement_input(requirement, documents):
    source = {"text": requirement.get("after"), "title": requirement.get("title", "")}
    context = [c for c in verified_buyer_context(requirement, documents) if c["role"] == "current"]
    if context:
        source["buyer_context"] = context
    issues = requirement.get("buyer_context_issues", [])
    if (not isinstance(issues, list) or any(not isinstance(issue, str) or issue not in
            {"paired_question_missing", "paired_question_ambiguous", "buyer_scope_conflict", *REFERENCE_REVIEW_ISSUES}
            for issue in issues)):
        raise ValueError("Buyer context has invalid pairing review reasons.")
    if issues:
        source["buyer_context_issues"] = list(dict.fromkeys(issues))
    if requirement.get("source_conflicts") and "buyer_scope_conflict" not in source.get("buyer_context_issues", []):
        source.setdefault("buyer_context_issues", []).append("buyer_scope_conflict")
    return source


def retrieval_query(requirement, documents):
    source = requirement_input(requirement, documents)
    # Only verified paired/referenced scope travels with the answer. Whole pages
    # can contain unrelated requirements, and old versions do not define this query.
    return "\n".join([*(c["quote"] for c in source.get("buyer_context", [])), source["text"]])


def validate_citations(requirement, documents, *, obligation_ids=None):
    lookup = {d["id"]: d for d in documents}
    citations = list(requirement.get("citations", [])) + verified_buyer_context(requirement, documents)
    if requirement.get("proposal"):
        proposal = requirement["proposal"]
        source_input = requirement_input(requirement, documents)
        if obligation_ids is None:
            validate_obligation_proposal(proposal, source_input, requirement.get("retrieval", []))
        else:
            model_contracts.validate_obligation_batch_proposal(proposal, source_input, requirement.get("retrieval", []), obligation_ids)
        citations += proposal.get("citations", [])
        for check in proposal.get("obligation_checks", []):
            citations += check.get("citations", [])
        if 'grounding_observations' in proposal:
            observations = proposal['grounding_observations']
            routes = proposal.get('source_routes', [])
            checked_ids = [c['obligation_id'] for c in proposal['obligation_checks']]
            if routes:
                from .facts import route_contract
                expected_routes = [route for route in route_contract(decompose_requirement(source_input))['routes']
                                   if route['obligation_id'] in checked_ids]
                if routes != expected_routes:
                    raise ValueError('Source routing must match the exact current buyer source.')
                routed = {route['obligation_id'] for route in routes}
                if any(c['assessment'] != 'unknown' or c['model_assessment'] != 'unknown'
                       or c['citation_ids'] for c in proposal['obligation_checks'] if c['obligation_id'] in routed):
                    raise ValueError('Source review routes cannot establish supplier compliance.')
            else:
                routed = set()
            expected = [identifier for identifier in checked_ids if identifier not in routed]
            if (not isinstance(observations, list)
                    or [o.get('obligation_id') for o in observations] != expected):
                raise ValueError('Grounding observations must cover each model-assessed condition once.')
            catalog = {c['citation_id']: c for c in model_contracts.build_citation_catalog(requirement.get('retrieval', []))}
            for observation in observations:
                for span in observation.get('evidence', []):
                    original = catalog.get(span.get('citation_id'))
                    quote = span.get('quote')
                    if (not original or span.get('mode') != 'text' or not isinstance(quote, str)
                            or not quote.strip() or quote not in original['quote']):
                        raise ValueError('Grounded supplier observation must match its original exact text source.')
                    citations.append({**original, 'quote': quote})
    for cite in citations:
        doc = lookup.get(cite.get("document_id"))
        if doc is None:
            raise ValueError("A finding references a source outside this tender's selected document versions.")
        page = next((p for p in doc["pages"] if p["page"] == cite.get("page")), None)
        quote = normalized(cite.get("quote", ""))
        if not page or not quote or quote not in normalized(page["text"]):
            raise ValueError("A finding's quotation does not exist on its cited source page.")
        cite.update(title=doc["title"], version=doc["version"], source_url=doc.get("source_url"))
    return len(citations)


def obligation_batch_key(binding, obligation_ids):
    return digest({"binding": binding, "obligation_ids": obligation_ids, "cache_version": "durable_obligation_batches_v1"})


def assessment_inputs(req, documents, passages, profile, objects, run_id, execution):
    """Exact source-mapped input identity shared by the worker and bounded repair tool."""
    source_input = requirement_input(req, documents)
    docs = {doc['id']: doc for doc in documents}
    page_sources = {}
    if profile == 'multimodal':
        for passage in passages:
            if passage.get('image_key'):
                source = docs.get(passage['document_id'])
                page = next((p for p in (source or {}).get('pages', []) if p['page'] == passage['page']), None)
                if (not source or source.get('role') != 'evidence' or not page
                        or page.get('image_key') != passage['image_key']):
                    raise ValueError('Retrieved supplier image does not match its selected source page.')
                page_sources.setdefault(passage['image_key'], {'role': 'evidence', 'document_id': passage['document_id'],
                                                               'title': source['title'], 'page': passage['page']})
    page_keys = list(page_sources)[:4]
    image_sources = [page_sources[key] for key in page_keys]
    evidence = [{key: p[key] for key in ('document_id', 'page', 'title', 'text', 'role') if key in p} for p in passages]
    images = [objects.get(key) for key in page_keys]
    image_hashes = [sha256(image).hexdigest() for image in images]
    binding = digest({'run_id': run_id, 'requirement_id': req['id'], 'after': req['after'], 'title': req['title'],
                      'requirement_input': source_input,
                      'before': req.get('before'), 'citations': req.get('citations', []), 'evidence': evidence,
                      'image_keys': page_keys, 'image_sources': image_sources, 'image_sha256': image_hashes,
                      'execution': execution, 'profile': profile,
                      'sources': sorted((d['id'], d.get('version'), d.get('raw_sha256')) for d in documents)})
    return {'requirement': source_input, 'evidence': evidence, 'images': images,
            'image_sources': image_sources, 'image_sha256': image_hashes, 'binding': binding}


def assess_obligation_batches(models, requirement, evidence, images, *, image_sources, run_id,
                              requirement_id, binding, cache, check, save, validate=None, progress=None,
                              generation_model=None):
    """Checkpoint validated canonical batches; an incomplete requirement has no aggregate."""
    contract = decompose_requirement(requirement)
    identifiers = [o['obligation_id'] for o in contract['obligations']]
    groups = [identifiers[start:start + OBLIGATIONS_PER_BATCH] for start in range(0, len(identifiers), OBLIGATIONS_PER_BATCH)]
    # These common source issues already force every canonical check to unknown
    # in the validator. Inference cannot repair a missing or ambiguous buyer scope.
    if contract.get('buyer_context_issues'):
        check()
        raw = {'checks': [{'obligation_id': identifier, 'assessment': 'unknown',
                           'rationale': 'Buyer source scope requires review before supplier assessment.',
                           'citation_ids': []} for identifier in identifiers]}
        proposal = model_contracts.validate_obligation_selection(json.dumps(raw), contract,
                    model_contracts.build_citation_catalog(evidence), evidence)
        proposal.update(assessment_strategy='buyer_scope_review_required', model=None)
        if progress:
            progress(len(groups), len(groups), 0)
        check()
        return {'proposal': proposal, 'completed_batches': len(groups), 'reused_batches': 0,
                'last_model_stats': {}}
    proposals, reused, last_stats = [], 0, {}
    grouped_provider = callable(getattr(models, 'assess_group', None))
    if grouped_provider:
        from .facts import route_contract
        routes = {route['obligation_id']: route for route in route_contract(contract)['routes']}
    else:
        routes = {}
    prefetched = {}
    for group_index, ids in enumerate(groups):
        check()
        key = obligation_batch_key(binding, ids)
        entry = cache.get(key)
        proposal = None
        if (isinstance(entry, dict) and entry.get('run_id') == run_id and entry.get('requirement_id') == requirement_id
                and entry.get('binding') == binding and entry.get('obligation_ids') == ids):
            try:
                proposal = copy.deepcopy(entry['proposal'])
                model_contracts.validate_obligation_batch_proposal(proposal, requirement, evidence, ids)
                expected_routes = [routes[identifier] for identifier in ids if identifier in routes]
                if proposal.get('source_routes', []) != expected_routes:
                    raise ValueError('Cached source routes do not match the current routing contract.')
                if any(c['assessment'] != 'unknown' or c['model_assessment'] != 'unknown'
                       or c['citation_ids'] for c in proposal['obligation_checks'] if c['obligation_id'] in routes):
                    raise ValueError('Cached source review route contains a supplier verdict.')
                if validate:
                    validate(proposal, ids)
                reused += 1
            except (ModelFailure, ValueError, KeyError, TypeError, AttributeError):
                proposal = None
        if proposal is None:
            check()
            if grouped_provider and tuple(ids) not in prefetched:
                # MLX decodes several canonical triples together, then each
                # triple retains the existing verified durable checkpoint.
                combined = groups[group_index:group_index + 4]
                combined_ids = [identifier for group in combined for identifier in group]
                model_ids = [identifier for identifier in combined_ids if identifier not in routes]
                generated = {'obligation_checks': [], 'grounding_observations': []}
                if model_ids:
                    generated = copy.deepcopy(models.assess_group(requirement, evidence, images,
                        image_sources=image_sources, obligation_ids=model_ids))
                    selected_contract = {**contract, 'obligations': [o for o in contract['obligations']
                                                                   if o['obligation_id'] in model_ids]}
                    model_contracts._validate_proposal_for_contract(generated, selected_contract, evidence)
                selections = {c['obligation_id']: {'obligation_id': c['obligation_id'],
                    'assessment': c['model_assessment'], 'rationale': c['model_rationale'],
                    'citation_ids': c['citation_ids']} for c in generated['obligation_checks']}
                selections.update({identifier: {'obligation_id': identifier, 'assessment': 'unknown',
                    'rationale': routes[identifier]['reason'], 'citation_ids': []}
                    for identifier in combined_ids if identifier in routes})
                catalog = model_contracts.build_citation_catalog(evidence)
                for group in combined:
                    selected = model_contracts.select_obligation_batch(requirement, group)
                    split = model_contracts.validate_obligation_selection(
                        json.dumps({'checks': [selections[identifier] for identifier in group]}), selected, catalog, evidence)
                    split.update(assessment_scope='obligation_batch', obligation_batch={
                        'obligation_ids': group, 'total_obligations': len(identifiers)},
                        model_stats=generated.get('model_stats', {}))
                    if 'grounding_observations' in generated:
                        split['grounding_observations'] = copy.deepcopy([
                            observation for observation in generated['grounding_observations']
                            if observation['obligation_id'] in group])
                    if any(identifier in routes for identifier in group):
                        split['source_routes'] = copy.deepcopy([routes[identifier] for identifier in group if identifier in routes])
                    prefetched[tuple(group)] = split
            proposal = prefetched.pop(tuple(ids)) if tuple(ids) in prefetched else copy.deepcopy(
                models.assess_batch(requirement, evidence, images, image_sources=image_sources, obligation_ids=ids))
            last_stats = {key: copy.deepcopy(value) for key, value in proposal.pop('model_stats', {}).items()
                          if key in _INVOCATION_FIELDS}
            proposal.pop('raw_response', None)
            try:
                model_contracts.validate_obligation_batch_proposal(proposal, requirement, evidence, ids)
                if validate:
                    validate(proposal, ids)
            except Exception:
                save(None, None, last_stats)
                raise
            entry = {'run_id': run_id, 'requirement_id': requirement_id, 'binding': binding,
                     'obligation_ids': ids, 'proposal': copy.deepcopy(proposal),
                     'model_stats': last_stats, 'verified_at': now()}
            save(key, entry, last_stats)
            cache[key] = copy.deepcopy(entry)
        else:
            last_stats = copy.deepcopy(entry.get('model_stats', {}))
        proposals.append(proposal)
        if progress:
            progress(len(proposals), len(groups), reused)
        check()
    complete = model_contracts.assemble_obligation_proposals(proposals, requirement, evidence)
    if any('grounding_observations' in proposal for proposal in proposals):
        complete['grounding_observations'] = copy.deepcopy([
            observation for proposal in proposals for observation in proposal.get('grounding_observations', [])])
    if routes:
        complete['source_routes'] = copy.deepcopy([routes[identifier] for identifier in identifiers if identifier in routes])
        complete['assessment_strategy'] = ('source_role_review_required' if len(routes) == len(identifiers)
                                           else 'mlx_grounded_with_source_role_review_v1')
    complete['model'] = (None if len(routes) == len(identifiers) else
                         generation_model or last_stats.get('model') or MODELS['generation'])
    return {'proposal': complete, 'completed_batches': len(groups), 'reused_batches': reused,
            'last_model_stats': last_stats}


def review_fingerprint(req, documents, profile, config=None):
    evidence = [d for d in documents if d["role"] == "evidence"]
    conflict_citations = [c for record in req.get("source_conflicts", []) for c in record.get("observations", [])]
    citations = req.get("citations", []) + req.get("buyer_context", []) + conflict_citations + (req.get("proposal") or {}).get("citations", [])
    referenced = {c["document_id"] for c in citations} | {req.get("source_document_id")}
    # Exact source versions and model settings are review dependencies, not just the assessment label.
    # Raw model observations are retained for audit, but only the validated
    # conclusion shown to the reviewer belongs to this dependency contract.
    proposal = req.get('proposal')
    if isinstance(proposal, dict):
        proposal = {key: value for key, value in proposal.items() if key != 'grounding_observations'}
    return digest({"key": req["key"], "before": req.get("before"), "after": req.get("after"),
                   "assessment": req["assessment"], "proposal": proposal,
                   "buyer_context": req.get("buyer_context", []),
                   "buyer_context_issues": req.get("buyer_context_issues", []),
                   "source_conflicts": req.get("source_conflicts", []),
                   "citations": sorted((c["document_id"], c["page"], normalized(c["quote"])) for c in citations),
                   "sources": sorted((d["id"], d["version"], d["raw_sha256"]) for d in documents if d["id"] in referenced),
                   "evidence": sorted((d["id"], d["raw_sha256"]) for d in evidence),
                   "profile": profile, "execution": req.get("execution_config") or execution_signature(config)})


def apply_reviews(requirements, old_requirements, documents, profile, config=None):
    old = {r["key"]: r for r in old_requirements}
    invalidated = 0
    for req in requirements:
        req["fingerprint"] = review_fingerprint(req, documents, profile, config)
        prior = old.get(req["key"])
        if prior and prior.get("fingerprint") == req["fingerprint"]:
            for key in ("reviewed", "review_decision", "review_note"):
                if key in prior:
                    req[key] = prior[key]
            if prior.get("reviewed"):
                req["status"] = req["assessment"]
            elif prior.get("status") == "needs_review":
                req["status"] = "needs_review"
        elif prior and (prior.get("reviewed") or prior.get("status") == "needs_review"):
            req.update(status="needs_review", reviewed=False)
            invalidated += int(bool(prior.get("reviewed")))
    return invalidated


_INVOCATION_FIELDS = {
    "id", "runtime_id", "run_id", "stage", "model", "method", "status", "started_at",
    "input_tokens", "output_tokens", "seconds", "load_seconds", "device", "dtype", "revision",
    "items", "images", "image_roles", "citation_strategy", "citation_catalog_items",
    "is_query", "max_image_pixels", "max_text_tokens", "max_new_tokens",
    "vector_shapes", "adapter_tensors_verified", "adapter_key_mapping",
    "assessment_contract", "obligation_count", "obligation_checked_count", "obligation_coverage_complete",
    "generation_batches", "generation_batches_completed",
    "schema_retry_attempts", "schema_validation_failures", "schema_errors",
    "decoding_strategy", "decoding_library", "decoding_library_version",
    "assessment_strategy", "provider", "hosted_inference", "model_version", "response_id", "kind",
    "operation", "attempt", "http_status", "retry_scheduled", "finish_reason", "total_tokens",
    "thought_tokens", "reasoning_tokens", "cached_tokens", "prefix_cache",
    "embedding_dimensions", "max_output_tokens",
    "total_obligation_count", "obligation_ids", "validation_code",
    "native_batch_id", "native_batch_size", "native_batch_seconds", "gate_version", "rejected_count",
}


def _safe_invocations(records, run_id):
    """Keep only correlated execution metadata, never prompts, responses or request bodies."""
    result = []
    for record in records or []:
        if (not isinstance(record, dict) or record.get("run_id") != run_id
                or not record.get("runtime_id") or record.get("id") is None):
            continue
        clean = {key: copy.deepcopy(value) for key, value in record.items() if key in _INVOCATION_FIELDS}
        validation_code = clean.pop("validation_code", None)
        if isinstance(validation_code, str) and validation_code in VALIDATION_REASONS:
            clean["validation_code"] = validation_code
        # Error messages can contain user source text. Preserve only the exception class.
        error_type = record.get("error_type") or str(record.get("error", "")).partition(":")[0]
        if error_type:
            clean["error_type"] = error_type if error_type.isidentifier() else "ModelFailure"
        result.append(clean)
    return result


def _model_snapshot(models, run_id):
    try:
        snapshot = models.status()
        if not isinstance(snapshot.get("invocations"), list):
            raise ValueError("Model health did not include invocation records")
        return _safe_invocations(snapshot["invocations"], run_id), None
    except Exception as exc:
        # Telemetry availability must not turn a successful stage into a failure,
        # or replace the original inference failure while recording its outcome.
        return [], type(exc).__name__


def _invocation_totals(calls):
    completed = [call for call in calls if call.get("status") == "completed"]
    def known(value):
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    def total_for_call(call):
        if known(call.get("total_tokens")):
            return call["total_tokens"]
        if (call.get("provider") != "gemini" and known(call.get("input_tokens"))
                and known(call.get("output_tokens"))):
            return call["input_tokens"] + call["output_tokens"]
        return None
    inputs = [call.get("input_tokens") for call in calls]
    outputs = [call.get("output_tokens") for call in calls]
    totals = [total_for_call(call) for call in calls]
    def complete_sum(values):
        return sum(values) if all(known(value) for value in values) else None
    reported = {"input": sum(known(value) for value in inputs), "output": sum(known(value) for value in outputs),
                "total": sum(known(value) for value in totals)}
    coverage = ("not_started" if not calls else "complete" if all(n == len(calls) for n in reported.values())
                else "partial" if any(reported.values()) else "unavailable")
    return {"recorded_invocations": len(calls), "completed_invocations": len(completed),
            "failed_invocations": sum(call.get("status") == "failed" for call in calls),
            "input_tokens": complete_sum(inputs), "output_tokens": complete_sum(outputs), "total_tokens": complete_sum(totals),
            "reported_input_tokens": sum(value for value in inputs if known(value)),
            "reported_output_tokens": sum(value for value in outputs if known(value)),
            "reported_total_tokens": sum(value for value in totals if known(value)),
            "token_usage": {"status": coverage, "reported_invocations": reported},
            "duration_ms": round(sum(call.get("seconds") or 0 for call in calls) * 1000),
            "models_completed": sorted({call["model"] for call in completed if call.get("model")}),
            "models_attempted": sorted({call["model"] for call in calls if call.get("model")})}


def _merge_model_telemetry(run, state, update, records, run_id, stage_name, health_error):
    """Merge durable history and checkpoint state, retaining earlier retry attempts."""
    stats_sources = [value.get("model_stats") or {} for value in (run, state, update)]
    stats = {key: value for source in stats_sources for key, value in source.items()}
    merged = {}
    for source in [*(value.get("calls", []) for value in stats_sources), records]:
        for call in _safe_invocations(source, run_id):
            merged[(call["runtime_id"], call["id"])] = call
    calls = sorted(merged.values(), key=lambda c: (c.get("started_at", ""), c["runtime_id"], c["id"]))
    missed = max((source.get("telemetry", {}).get("unavailable_snapshots", 0) for source in stats_sources), default=0)
    telemetry = {"status": "unavailable" if health_error else "available", "last_stage": stage_name,
                 "last_collected_at": now(), "unavailable_snapshots": missed + int(bool(health_error))}
    if health_error:
        telemetry["error_type"] = health_error
    stats.update(calls=calls, telemetry=telemetry)
    metrics = {key: value for value in (run, state, update) for key, value in (value.get("metrics") or {}).items()}
    metrics["model_inference"] = {**_invocation_totals(calls), "telemetry_status": telemetry["status"]}
    run.update(model_stats=stats, metrics=metrics)
    return stats, metrics


def execute(config, store, objects, job, worker_id, cancelled=None):
    resources = []
    try:
        return _execute(config, store, objects, job, worker_id, cancelled, resources)
    finally:
        # All parallel work drains inside its stage before this resource boundary.
        # Cleanup cannot replace an inference, cancellation or lease exception.
        for resource in reversed(resources):
            close = getattr(resource, "close", None)
            if close:
                try:
                    close()
                except Exception:
                    pass


def _execute(config, store, objects, job, worker_id, cancelled, resources):
    tender_id, run_id = job["tender_id"], job["run_id"]
    profile = job["payload"]["profile"]
    workspace_id = job["workspace_id"]
    detail = store.get(tender_id, workspace_id)
    if source_fingerprint(detail, profile, config) != job["payload"]["fingerprint"]:
        raise ValueError("Documents or inference configuration changed after this analysis was queued.")
    cloud = getattr(config, "inference_provider", "local") in {"gemini", "codex"}
    configured_models = model_ids(config)
    if cloud or getattr(config, "inference_provider", "local") == "mlx":
        models = create_model_client(config, run_id=run_id)
        resources.append(models)
        retriever = QdrantRetriever(config.qdrant_url, config.model_url, workspace_id, run_id=run_id,
                                    config=config, models=models)
    else:
        models = ModelClient(config.model_url, run_id=run_id)
        retriever = QdrantRetriever(config.qdrant_url, config.model_url, workspace_id, run_id=run_id)

    state_lock = RLock()
    admission = None
    progress_written = {}

    def lease():
        if admission is not None:
            admission.raise_if_stopped()
        if cancelled and cancelled.is_set():
            raise LeaseLostError("Worker lease lost; this attempt cannot publish results.")
        with state_lock:
            store.heartbeat(job["id"], worker_id)

    def map_work(fn, items):
        nonlocal admission
        local_batching = (getattr(config, 'inference_provider', 'local') == 'mlx'
                          and getattr(config, 'mlx_assessment_strategy', 'condition_nli') == 'condition_nli')
        if not cloud and not local_batching:
            return [fn(item) for item in items]
        admission = WorkAdmission()
        try:
            concurrency = (config.codex_concurrency if config.inference_provider == "codex"
                           else 12 if local_batching else getattr(config, "gemini_max_concurrency", 8))
            return bounded_map(fn, items, max_workers=concurrency,
                               check=lease, admission=admission)
        finally:
            admission = None

    def mutate(callback, **options):
        with state_lock:
            return store.mutate_job(job["id"], worker_id, callback, workspace_id,
                                    expected_run_id=run_id, expected_tender_id=tender_id, **options)

    def mutate_run(callback, **options):
        with state_lock:
            return store.mutate_run(job["id"], worker_id, callback, workspace_id,
                                    expected_run_id=run_id, expected_tender_id=tender_id, **options)

    def work_progress(completed, total, unit):
        with state_lock:
            key = (models.stage, unit)
            previous, written_at = progress_written.get(key, (0, 0.0))
            completed = max(previous, completed)
            moment = time.monotonic()
            if completed not in (0, total) and moment - written_at < .25:
                return  # UI-only updates; call and cache receipts never take this path.
            def save(run):
                run["work_progress"] = {"stage": models.stage, "completed": completed, "total": total, "unit": unit}
                return run
            # Completed work is safe to record while stopping; publication stays fenced.
            mutate_run(save, allow_stopping=True)
            # A slow commit must not consume the quiet interval and cause every
            # subsequent item to issue another UI-only write.
            progress_written[key] = (completed, time.monotonic())

    retriever.check_cancel = lease
    retriever.report_progress = work_progress
    if cloud:
        models.check_cancel = lease
        def record_model_call(record):
            def save(run):
                _merge_model_telemetry(run, {}, {}, [record], run_id, models.stage, None)
                return run
            mutate_run(save, allow_stopping=True)
        models.on_model_call = record_model_call

    def stage(name, title, fn):
        def wrapped(state):
            lease()
            models.stage = retriever.models.stage = name
            start = time.perf_counter()
            entry = {"name": name, "title": title, "status": "running", "started_at": now(), "completed_at": None}

            def start_step(d):
                run = next(r for r in d["runs"] if r["id"] == run_id)
                run.update(status="running", started_at=run.get("started_at") or now())
                run["steps"] = [s for s in run.get("steps", []) if s["name"] != name] + [entry]
                run["work_progress"] = {"stage": name, "completed": 0, "total": None, "unit": "items"}
                d["project"]["analysis_status"] = "running"
                return d
            if name == 'requirements':
                # Materialize the running project once. Later stage starts only
                # change run checkpoints, so avoid rewriting source/history.
                mutate(start_step)
            else:
                def start_run_step(run):
                    run['steps'] = [s for s in run.get('steps', []) if s['name'] != name] + [entry]
                    run['work_progress'] = {'stage': name, 'completed': 0, 'total': None, 'unit': 'items'}
                    return run
                mutate_run(start_run_step)
            try:
                update = fn(state)
                lease()
                entry.update(status="completed", completed_at=now(), duration_ms=round((time.perf_counter() - start) * 1000))
                if update.get("stage_metrics"):
                    entry["metrics"] = update.pop("stage_metrics")
                records, health_error = _model_snapshot(models, run_id)
                update["steps"] = [s for s in state.get("steps", []) if s["name"] != name] + [entry]

                def done_step(run):
                    update["model_stats"], update["metrics"] = _merge_model_telemetry(
                        run, state, update, records, run_id, name, health_error)
                    run["steps"] = update["steps"]
                    return run
                mutate_run(done_step)
                return update
            except Exception as exc:
                entry.update(status="cancelled" if isinstance(exc, JobCancelled) else "failed",
                             completed_at=now(), duration_ms=round((time.perf_counter() - start) * 1000))
                if not cancelled or not cancelled.is_set():
                    records, health_error = _model_snapshot(models, run_id)
                    def fail_step(run):
                        _merge_model_telemetry(run, state, {}, records, run_id, name, health_error)
                        run["steps"] = [s for s in run.get("steps", []) if s["name"] != name] + [entry]
                        return run
                    try:
                        mutate_run(fail_step, allow_stopping=True)
                    except LeaseLostError:
                        pass
                raise
        return wrapped

    def requirements_analyst(state):
        work_progress(0, None, "requirements")
        requirements = extract_requirements(state["documents"])
        source_context = [citation for doc in state["documents"] if doc["role"] in {"current", "previous"}
                          for citation in requirement_source_context(doc, doc["role"])]
        work_progress(len(requirements), len(requirements), "requirements")
        return {"requirements": requirements, "stage_metrics": {
            "requirements": len(requirements), "method": "document-scoped clause alignment and conservative atomic checks",
            **buyer_input_page_receipts(state["documents"]),
            "source_context": source_context,
            "coverage": "Candidate clause segmentation is heuristic; review against the original source. "
                        "Cited layout, inventory and buyer Q/A context is retained in source_context and the original documents; "
                        "it is not assessed as standalone supplier obligations. OCR and generated preview coverage are recorded per document."}}

    def index_sources(state):
        metrics = retriever.index(tender_id, state["documents"], objects, profile)
        return {"stage_metrics": metrics, "metrics": {**state.get("metrics", {}), "index": metrics}}

    def retrieve_evidence(state):
        evidence_ids = [d["id"] for d in state["documents"] if d["role"] == "evidence"]
        eligible = [r for r in state["requirements"] if r.get("after")]
        if not evidence_ids:
            work_progress(0, 0, "retrieval steps")
            return {"retrieved": {}, "stage_metrics": {"queries": 0, "reason": "No supplier evidence supplied; no capability conclusion can be established."}}
        complete_context = (bounded_supplier_context(state['documents'])
                            if getattr(config, 'inference_provider', 'local') in {'mlx', 'codex'} else None)
        if complete_context is not None:
            work_progress(0, len(complete_context), 'supplier pages')
            for index, _ in enumerate(complete_context, 1):
                lease()
                work_progress(index, len(complete_context), 'supplier pages')
            return {'retrieved': {r['id']: copy.deepcopy(complete_context) for r in eligible},
                    'stage_metrics': {'queries': 0, 'profile': profile,
                        'selection_policy': 'bounded_complete_supplier_text',
                        'supplier_pages': len(complete_context),
                        'supplier_text_characters': sum(len(p['text']) for p in complete_context),
                        'input_pages': [{k: p[k] for k in ('document_id', 'page', 'role')} for p in complete_context],
                        'scope': 'all readable pages of active supplier snapshots within the local context budget'}}
        queries = {r["id"]: retrieval_query(r, state["documents"]) for r in eligible}
        encoding_count = len(set(queries.values())) * (2 if profile == "multimodal" else 1)
        total = encoding_count + len(eligible)
        retriever.report_progress = lambda completed, _, unit: work_progress(completed, total, "retrieval steps")
        work_progress(0, total, "retrieval steps")
        retriever.prepare_queries(list(queries.values()), profile)
        if cloud and hasattr(retriever, "prepare_search_scope"):
            retriever.prepare_search_scope(tender_id, evidence_ids, profile)
        finished = 0
        def search(req):
            nonlocal finished
            lease()
            found = retriever.search(tender_id, queries[req["id"]], evidence_ids, profile, limit=4)
            with state_lock:
                finished += 1
                work_progress(encoding_count + finished, total, "retrieval steps")
            return req["id"], found
        passages = dict(map_work(search, eligible))
        return {"retrieved": passages, "stage_metrics": {"queries": len(eligible), "profile": profile,
                "retrieved_passages": sum(len(v) for v in passages.values()), "scope": "active supplier snapshots only"}}

    def evidence_analyst(state):
        requirements = copy.deepcopy(state["requirements"])
        completed, reused, tokens, calls = 0, 0, 0, []
        docs = {d["id"]: d for d in state["documents"]}
        execution = execution_signature(config)
        run = next(r for r in store.get(tender_id, workspace_id)["runs"] if r["id"] == run_id)
        cache = copy.deepcopy(run.get("assessment_cache", {}))
        batch_cache = copy.deepcopy(run.get('obligation_batch_cache', {}))
        batch_counts = {r['id']: (len(decompose_requirement(requirement_input(r, state['documents']))['obligations'])
                                  + OBLIGATIONS_PER_BATCH - 1) // OBLIGATIONS_PER_BATCH
                        for r in requirements if r.get('after') and state.get('retrieved', {}).get(r['id'])}
        batches_completed, batches_reused, batches_total = 0, 0, sum(batch_counts.values())
        total = sum(bool(r.get("after") and state.get("retrieved", {}).get(r["id"])) for r in requirements)
        work_progress(0, batches_total, "obligation batches")

        def progress(requirement_id, entry=None, *, batch_key=None, batch_entry=None, batch_stats=None):
            # Cache and invocation history are committed together; final findings
            # remain exclusively owned by the graph's whole-run verifier.
            def save(run):
                nonlocal tokens
                if entry is not None:
                    run.setdefault("assessment_cache", {})[requirement_id] = copy.deepcopy(entry)
                if batch_entry is not None:
                    run.setdefault('obligation_batch_cache', {})[batch_key] = copy.deepcopy(batch_entry)
                if batch_stats:
                    calls.extend(_safe_invocations([batch_stats], run_id))
                stats = run.setdefault("model_stats", {})
                merged = {(call["runtime_id"], call["id"]): call
                          for call in _safe_invocations([*stats.get("calls", []), *calls], run_id)}
                stats["calls"] = sorted(merged.values(), key=lambda c: (c.get("started_at", ""), c["runtime_id"], c["id"]))
                assessment_calls = [c for c in stats["calls"] if c.get("stage") == "assess" or c.get("method") in {"assess", "assess_batch"}]
                tokens = _invocation_totals(assessment_calls)["total_tokens"]
                stats.update(completed_assessments=completed, reused_assessments=reused, tokens=tokens)
                run["progress"] = {"assessed": completed, "total": total, "reused": reused}
                run["work_progress"] = {"stage": "assess", "completed": batches_completed,
                                        "total": batches_total, "unit": "obligation batches"}
                return run
            with state_lock:
                mutate_run(save, allow_stopping=True)

        def assess_requirement(req):
            nonlocal completed, reused, batches_completed, batches_reused
            passages = state.get("retrieved", {}).get(req["id"], [])
            req["retrieval"] = passages
            if not req.get("after") or not passages:
                return
            lease()
            inputs = assessment_inputs(req, state["documents"], passages, profile, objects, run_id, execution)
            evidence, images, image_sources, binding = (inputs[key] for key in ("evidence", "images", "image_sources", "binding"))
            entry = cache.get(req["id"])
            cached = None
            if isinstance(entry, dict) and entry.get("run_id") == run_id and entry.get("binding") == binding:
                candidate = copy.deepcopy(req)
                candidate["proposal"] = copy.deepcopy(entry.get("proposal"))
                try:
                    if not isinstance(candidate["proposal"], dict):
                        raise ValueError("Incomplete assessment cache entry")
                    validate_citations(candidate, state["documents"])
                    cached = candidate
                except (ValueError, KeyError, TypeError, AttributeError, ModelFailure):
                    pass  # Invalid cached quotations must be assessed again.
            if cached is not None:
                req["proposal"] = cached["proposal"]
                req["execution_config"] = copy.deepcopy(entry["execution_config"])
                with state_lock:
                    reused += 1
                    batches_completed += batch_counts[req['id']]
                    batches_reused += batch_counts[req['id']]
            else:
                lease()
                used_batch_path = (getattr(config, 'inference_provider', 'local') in {'mlx', 'codex'}
                                   or batch_counts[req['id']] > 1
                                   or bool(inputs['requirement'].get('buyer_context_issues')))
                if used_batch_path:
                    local_done, local_reused = 0, 0
                    def batch_progress(done, expected, reused_count):
                        nonlocal batches_completed, batches_reused, local_done, local_reused
                        with state_lock:
                            batches_completed += done - local_done
                            batches_reused += reused_count - local_reused
                            local_done, local_reused = done, reused_count
                            # This is a UI counter only. The preceding save has
                            # already committed this batch's validated checks and
                            # invocation receipts. Publish counters with the next
                            # checkpoint or the final requirement checkpoint.
                    def validate_batch(proposal, ids):
                        candidate = {**copy.deepcopy(req), 'proposal': proposal}
                        validate_citations(candidate, state['documents'], obligation_ids=ids)
                    result = assess_obligation_batches(models, inputs['requirement'], evidence, images,
                        image_sources=image_sources, run_id=run_id, requirement_id=req['id'], binding=binding,
                        cache={key: value for key, value in batch_cache.items()
                               if isinstance(value, dict) and value.get('requirement_id') == req['id']},
                        check=lease, validate=validate_batch, progress=batch_progress,
                        generation_model=configured_models['generation'],
                        save=lambda key, value, stats: progress(req['id'], batch_key=key, batch_entry=value, batch_stats=stats))
                    proposal, stats = result['proposal'], result['last_model_stats']
                else:
                    proposal = copy.deepcopy(models.assess(inputs['requirement'], evidence, images,
                                                           image_sources=image_sources))
                    raw_stats = proposal.pop("model_stats", {})
                    proposal.pop("raw_response", None)
                    stats = {key: copy.deepcopy(value) for key, value in raw_stats.items() if key in _INVOCATION_FIELDS}
                    with state_lock:
                        calls.extend(_safe_invocations([stats], run_id))
                candidate = copy.deepcopy(req)
                candidate["proposal"] = proposal
                try:
                    validate_citations(candidate, state["documents"])
                except Exception:
                    progress(req["id"])  # Record the actual invocation, but never cache an invalid proposal.
                    raise
                req["proposal"] = candidate["proposal"]
                req["execution_config"] = {**execution, "runtime": {
                    key: stats[key] for key in ("provider", "model", "device", "dtype", "max_image_pixels", "max_new_tokens")
                    if key in stats}}
                entry = {"run_id": run_id, "binding": binding, "proposal": copy.deepcopy(req["proposal"]),
                         "model_stats": stats, "execution_config": copy.deepcopy(req["execution_config"]), "verified_at": now()}
                if not used_batch_path:
                    with state_lock:
                        batches_completed += 1
            if req['proposal'].get('assessment_strategy') == 'buyer_scope_review_required':
                req["rationale"] += " Buyer source scope requires review. These checks remain unknown; no model assessment was performed."
            elif req['proposal'].get('assessment_strategy') == 'source_role_review_required':
                req["rationale"] += " These buyer-context or bid-process checks require human review. No supplier compliance was inferred."
            else:
                req["rationale"] += " A model proposal is available below. Its cited passages were checked, but its semantic conclusion requires human review."
            if not req.get("reviewed"):
                req["status"] = "needs_review"
            with state_lock:
                completed += 1
                progress(req["id"], entry if cached is None else None)
            lease()
        # Complete-requirement cloud calls vary greatly in size. Start the
        # largest jobs first so one long section cannot strand the other slots
        # at the end. Mutate the same records and retain source order in results.
        assessment_work = (sorted(requirements, key=lambda req: batch_counts.get(req['id'], 0), reverse=True)
                           if getattr(config, 'inference_provider', 'local') == 'codex' else requirements)
        map_work(assess_requirement, assessment_work)
        calls.sort(key=lambda c: (c.get("started_at", ""), c["runtime_id"], c["id"]))
        return {"requirements": requirements,
                "model_stats": {"completed_assessments": completed, "reused_assessments": reused, "tokens": tokens, "calls": calls},
                "stage_metrics": {"completed_assessments": completed, "reused_assessments": reused,
                                  "new_assessment_invocations": len(calls), "tokens": _invocation_totals(calls)["total_tokens"],
                                  "completed_obligation_batches": batches_completed, "reused_obligation_batches": batches_reused,
                                  "generation_model": configured_models["generation"] if any(
                                      r.get('proposal') and r['proposal'].get('assessment_strategy') not in {'buyer_scope_review_required', 'source_role_review_required'}
                                      for r in requirements) else None,
                                  "images_enabled": profile == "multimodal",
                                  "buyer_context": "selected requirement and verified buyer source context; full buyer pages remain reviewable"}}

    def verifier(state):
        requirements = copy.deepcopy(state["requirements"])
        citation_count = 0
        work_progress(0, len(requirements), "requirements")
        for number, req in enumerate(requirements, 1):
            lease()
            citation_count += validate_citations(req, state["documents"])
            work_progress(number, len(requirements), "requirements")
        invalidated = apply_reviews(requirements, state["previous_requirements"], state["documents"], profile, config)
        return {"requirements": requirements, "metrics": {**state.get("metrics", {}),
                "citations_checked": citation_count, "reviews_invalidated": invalidated,
                "requirements": len(requirements), "human_review_required": True},
                "stage_metrics": {"citations_checked": citation_count, "reviews_invalidated": invalidated,
                                  "verification": "source identity, immutable version, page and exact normalized quote"}}

    graph = StateGraph(ReviewState)
    for name, title, action in [
        ("requirements", "Requirements analyst · align document versions", requirements_analyst),
        ("index", "Index document text and page images", index_sources),
        ("retrieve", "Evidence analyst · retrieve and rerank", retrieve_evidence),
        ("assess", "Evidence analyst · cited model proposals", evidence_analyst),
        ("verify", "Verifier · source checks and review dependencies", verifier),
    ]:
        graph.add_node(name, stage(name, title, action))
    graph.add_edge(START, "requirements")
    for source, target in [("requirements", "index"), ("index", "retrieve"), ("retrieve", "assess"), ("assess", "verify")]:
        graph.add_edge(source, target)
    graph.add_edge("verify", END)
    initial = {"tender_id": tender_id, "profile": profile, "documents": selected_sources(detail),
               "previous_requirements": detail["requirements"], "steps": [], "retrieved": {}, "metrics": {}}
    graph_config = {"configurable": {"thread_id": workspace_id + ":" + run_id}, "recursion_limit": 20,
                    "run_name": "BidTrace document review", "tags": ["bidtrace", profile],
                    "metadata": {"tender_id": tender_id, "run_id": run_id, "pipeline": PIPELINE_VERSION}}
    # Hosted traces are opt-in; inherited unrelated LANGSMITH settings cannot leak local documents.
    from langsmith import tracing_context
    allow_hosted = os.getenv("BIDTRACE_ALLOW_HOSTED_TRACES") == "1"
    if config.storage_backend == "embedded":
        from langgraph.checkpoint.sqlite import SqliteSaver
        checkpoint_context = SqliteSaver.from_conn_string(str(config.data_dir / "checkpoints.sqlite"))
    else:
        from langgraph.checkpoint.postgres import PostgresSaver
        checkpoint_context = PostgresSaver.from_conn_string(config.database_url)
    with checkpoint_context as checkpointer:
        checkpointer.setup()
        runner = graph.compile(checkpointer=checkpointer)
        saved = runner.get_state(graph_config)
        with tracing_context(enabled=allow_hosted):
            result = runner.invoke(None if saved.values else initial, config=graph_config) if (not saved.values or saved.next) else saved.values
    lease()
    def finish(d):
        if source_fingerprint(d, profile, config) != job["payload"]["fingerprint"]:
            raise ValueError("Current source versions or inference configuration no longer match this run.")
        d["requirements"] = result["requirements"]
        run = next(r for r in d["runs"] if r["id"] == run_id)
        run.update(status="completed", completed_at=now(), error=None, steps=result["steps"],
                   model_stats=result["model_stats"], metrics=result["metrics"], requirements=result["requirements"])
        run["duration_ms"] = sum(s.get("duration_ms", 0) for s in result["steps"])
        d["project"].update(analysis_status="completed", counts=counts(result["requirements"]),
                            results_run_id=run_id, updated_at=now())
        event(d, "analysis", f"Completed {profile} review of {len(result['requirements'])} candidate requirements. Citations checked; model conclusions remain reviewable proposals.")
        return d
    completion = {"requirements": len(result["requirements"]), "citations_checked": result["metrics"]["citations_checked"]}
    mutate(finish, result=completion)
    config.data_dir.joinpath("traces").mkdir(parents=True, exist_ok=True)
    trace = {"run_id": run_id, "tender_id": tender_id, "profile": profile, "steps": result["steps"],
             "metrics": result["metrics"], "model_stats": result["model_stats"], "completed_at": now()}
    config.data_dir.joinpath("traces", run_id + ".json").write_text(json.dumps(trace, indent=2))
    return completion
