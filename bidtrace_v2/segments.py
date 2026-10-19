# SPDX-License-Identifier: MPL-2.0
"""Bound oversized requirement views using their exact cited buyer-page spans.

No model calls, source edits, quote rewriting, or silent truncation. Segmentation
invalidates the parent's atomic assessment, so resulting findings need review.
"""
from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
from hashlib import sha256
from itertools import zip_longest
import re


class RequirementSegmentationError(ValueError):
    pass


def _normal(text):
    return " ".join(text.split())


def _citation_key(citation):
    return (citation.get("document_id"), citation.get("page"), citation.get("role"), citation.get("quote"))


def _distinct(citations):
    result, seen = [], set()
    for citation in citations:
        key = _citation_key(citation)
        if key not in seen:
            result.append(citation)
            seen.add(key)
    return result


def _source_citations(item, role, view):
    if not view:
        return []
    citations = [c for c in item.get("citations", []) if c.get("role") == role]
    if not citations or any(not isinstance(c.get("quote"), str) or not c["quote"].strip()
                            or not c.get("document_id") or c.get("page") is None for c in citations):
        raise RequirementSegmentationError(f"Oversized requirement {item['key']} has no complete {role} source citations.")
    unique = _distinct(citations)
    # Repeated identical references are normally duplicate citation metadata. If
    # the parent explicitly contains both occurrences, preserve both: equal text
    # alone cannot prove that a repeated source occurrence should be removed.
    original_text = _normal("\n".join(c["quote"] for c in citations))
    unique_text = _normal("\n".join(c["quote"] for c in unique))
    view_text = _normal(view)
    if view_text != original_text or view_text == unique_text:
        citations = unique
    source_text = _normal("\n".join(c["quote"] for c in citations))
    structured = (item.get("extraction") or {}).get("method") == "official_spreadsheet_requirement_rows"
    # The table parser's display contains generated field labels and omits cell
    # addresses. Its complete physical row citations are the authoritative text.
    # Generic clauses must have an explicit complete mapping; never guess which
    # uncited words can be dropped just to make the query fit.
    if not structured and view_text not in source_text:
        raise RequirementSegmentationError(f"Oversized requirement {item['key']} has {role} wording not covered by its source quotes.")
    return citations


def _quote_spans(quote, max_chars):
    start = 0
    while start < len(quote):
        end = min(start + max_chars, len(quote))
        if end < len(quote):
            window = quote[start:end]
            minimum = max(1, len(window) // 2)
            # Prefer paragraph, then line, then word boundaries, retaining the
            # separator itself. Slices are contiguous and have zero overlap.
            for pattern in (r"\n[ \t\r]*\n", r"\n", r"\s+"):
                boundaries = [m.end() for m in re.finditer(pattern, window) if m.end() >= minimum]
                if boundaries:
                    end = start + boundaries[-1]
                    break
        yield start, end
        start = end


def _segments(citations, max_chars):
    segments = []
    for index, citation in enumerate(citations):
        quote = citation["quote"]
        for start, end in _quote_spans(quote, max_chars):
            text = quote[start:end]
            if not text.strip():
                # Whitespace-only tails belong to the preceding slice when it
                # fits; otherwise retain their exact text as a source span.
                if segments and segments[-1]["citation_index"] == index and len(segments[-1]["text"]) + len(text) <= max_chars:
                    previous = segments[-1]
                    previous["text"] += text
                    previous["citation"]["quote"] += text
                    previous["end"] = end
                    continue
            segments.append({"text": text, "citation": {**deepcopy(citation), "quote": text},
                "citation_index": index, "start": start, "end": end, "source_quote_chars": len(quote)})
    return segments


def _align(before, after):
    matcher = SequenceMatcher(a=[_normal(s["text"]) for s in before],
                              b=[_normal(s["text"]) for s in after], autojunk=False)
    for operation, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        for prior, current in zip_longest(before[old_start:old_end], after[new_start:new_end]):
            yield prior, current, "exact_normalized_match" if operation == "equal" else "position_within_changed_region"


def _span(segment):
    if segment is None:
        return None
    return {"document_id": segment["citation"]["document_id"], "page": segment["citation"]["page"],
            **{key: segment[key] for key in ("citation_index", "start", "end", "source_quote_chars")}}


def bound_requirements(items, max_chars=3000):
    """Return complete requirement views bounded on both sides, backed by source quotes.

    Short items retain their existing identity and assessments. Oversized items
    become deterministic ``:part:001`` findings. The coverage metadata counts
    source-quote characters (including whitespace), not generated join newlines
    or unique whole-document characters. Supplier citations remain parent context.
    """
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    result = []
    for original in items:
        item = deepcopy(original)
        if max(len(item.get("before") or ""), len(item.get("after") or "")) <= max_chars:
            result.append(item)
            continue
        before_cites = _source_citations(item, "previous", item.get("before"))
        after_cites = _source_citations(item, "current", item.get("after"))
        before, after = _segments(before_cites, max_chars), _segments(after_cites, max_chars)
        pairs = list(_align(before, after))
        evidence = [deepcopy(c) for c in _distinct([c for c in item.get("citations", []) if c.get("role") == "evidence"])]
        for citation in evidence:
            citation["segmentation_context"] = "parent_requirement"
        total_before = sum(len(c["quote"]) for c in before_cites)
        total_after = sum(len(c["quote"]) for c in after_cites)
        for index, (prior, current, alignment) in enumerate(pairs, start=1):
            suffix = f":part:{index:03d}"
            part = deepcopy(item)
            part["key"] = item["key"] + suffix
            part["source_key"] = (item.get("source_key") or item["key"]) + suffix
            part["id"] = "req_" + sha256(part["key"].encode()).hexdigest()[:20]
            part["title"] = f"{item['title']} · part {index}/{len(pairs)}"
            part["before"] = prior["text"] if prior else None
            part["after"] = current["text"] if current else None
            if not before_cites and item.get("change_type") == "unchanged":
                change = "unchanged"  # Initial assessment has no verified amendment.
            elif prior is None:
                change = "added"
            elif current is None:
                change = "removed"
            else:
                change = "unchanged" if _normal(prior["text"]) == _normal(current["text"]) else "modified"
            part.update(change_type=change, assessment="unknown", status="needs_review", reviewed=False,
                review_decision=None, review_note=None, proposal=None, retrieval=[])
            for key in ("fingerprint", "execution_config"):
                part.pop(key, None)
            part["citations"] = [*( [prior["citation"]] if prior else []),
                                 *( [current["citation"]] if current else []), *deepcopy(evidence)]
            note = ("Long clause split into exact page-quote spans; all cited buyer text is retained across its parts. "
                    "Changed-region alignment is positional and requires review against the complete parent clause. "
                    "The parent's atomic assessment does not establish this part's compliance; inherited supplier citations are parent context.")
            part["rationale"] = note
            part["segmentation"] = {"method": "cited_page_spans", "parent_key": item["key"],
                "parent_change_type": item.get("change_type"), "part": index, "parts": len(pairs),
                "max_chars": max_chars, "alignment": alignment, "baseline_reset": True, "note": note,
                "before_span": _span(prior), "after_span": _span(current),
                "coverage": {"before_quote_chars_total": total_before, "after_quote_chars_total": total_after,
                    "before_quote_chars_in_part": len(part["before"] or ""),
                    "after_quote_chars_in_part": len(part["after"] or "")}}
            result.append(part)
    keys = [item["key"] for item in result]
    if len(keys) != len(set(keys)):
        raise RequirementSegmentationError("Segment identities collide; source requirement keys must be unique.")
    return result
