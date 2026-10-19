# SPDX-License-Identifier: MPL-2.0
"""Conservative, source-bound buyer inventory discrepancy candidates.

This is a small recognizer for explicit count tables and current inventory
statements, not a procurement fact extractor. Matching an asset never establishes
document precedence or equal applicability. N/A stays a non-numeric observation.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import re


_ASSETS = {
    "printer": r"printers?",
    "network_attached_storage": r"(?:network\s+attached\s+storage(?:\s+devices?)?|NAS(?:\s+devices?)?)",
    "workstation": r"(?:managed\s+)?workstations?",
    "server": r"(?:managed\s+)?servers?",
    "wireless_access_point": r"(?:(?:wireless\s+)?access\s+points?|WAPs?)",
    "switch": r"(?:network\s+device\s*[-–]\s*)?switch(?:es)?",
    "firewall": r"firewalls?",
    "router": r"routers?",
    "cable_modem": r"cable\s+modems?(?:\s*\(internet\))?",
}
_ASSET = "(?:" + "|".join(_ASSETS.values()) + ")"
_WORDS = {word: i for i, word in enumerate(("zero", "one", "two", "three", "four", "five", "six",
    "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty"))}
_COUNT = r"(?:\d+|" + "|".join(_WORDS) + ")"
_VALUE = rf"(?:{_COUNT}|N/A)"
_TABLE_HEADER = re.compile(r"\bDevice\s+Type[ \t]+Count\b[^\n]*", re.I)
_TABLE_ROW = re.compile(rf"^[ \t]*(?P<asset>{_ASSET})\s+(?P<value>{_VALUE})\b(?P<tail>[^\n]*)$", re.I | re.M)
_SECTION = re.compile(rf"^[ \t]*(?:[a-z][.)][ \t]+)?(?P<asset>{_ASSET})[ \t]*$", re.I)
_PREFIX = r"(?:[ \t]*(?:[•▪▫*-]|\d+[.)])[ \t]*)?"
_NUMBER_FIRST = re.compile(rf"^{_PREFIX}(?P<value>{_COUNT})\s+(?P<asset>{_ASSET})\b(?P<tail>[^\n]*)$", re.I)
_NA = re.compile(rf"^{_PREFIX}(?P<value>N/A)[ \t]*[.]?$", re.I)
_CURRENT_STATEMENT = re.compile(
    rf"^[ \t]*(?:the\s+)?(?:buyer|client|authority|council|town|city|organization|organisation)"
    rf"[ \t]+(?:currently[ \t]+)?(?:has|owns|operates)[ \t]+(?P<value>{_COUNT})"
    rf"\s+(?P<asset>{_ASSET})\b(?P<tail>[^\n]*)$", re.I)
_NON_INVENTORY = re.compile(r"\b(?:(?:supplier|bidder|vendor|proposer)\b|proposed\s+equipment|future\s+equipment|"
    r"new\s+equipment\s+to\s+be\s+supplied|unit\s+(?:price|cost))\b", re.I)


def _asset(raw):
    return next((key for key, pattern in _ASSETS.items() if re.fullmatch(pattern, raw, re.I)), None)


def _scope(tail):
    """Keep explicit sites distinct; never equate a site count with a total."""
    value = " ".join(tail.strip(" .\t\r").lower().split())
    if re.search(r"[$€£]|\b(?:per|cost|price|must|shall|should|can|will|minimum|maximum)\b", value):
        return None
    if re.fullmatch(r"(?:across|at|in) all (?:locations|sites|buildings)|in total", value):
        return "all_locations"
    if re.match(r"(?:at|in|across)\s+", value):
        return "site:" + value
    # Unknown trailing qualifications are not equivalent to an unqualified count.
    return "unspecified" if not value else None


def _current_inventory_question(row):
    for citation in row.get("buyer_context", []):
        if (citation.get("role") != "current" or citation.get("relation") != "paired_question"
                or citation.get("document_id") != row.get("source_document_id")):
            continue
        question = citation.get("quote", "")
        if (re.search(r"\b(?:equipment|devices?|inventory|printers?|NAS|workstations?|servers?)\b", question, re.I)
                and re.search(r"\b(?:current(?:ly)?|existing|inventory|how many|make\s+and\s+model|under warranty)\b", question, re.I)):
            return True
    return False


def _candidate_spans(row, text):
    header = _TABLE_HEADER.search(text)
    if header and not _NON_INVENTORY.search(text[:header.end()]):
        for match in _TABLE_ROW.finditer(text, header.end()):
            scope = _scope(match["tail"])
            if scope is not None:
                yield match.start(), match.end(), match["asset"], match["value"], scope

    paired_inventory = _current_inventory_question(row)
    section = None
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        match = _SECTION.fullmatch(content)
        if match:
            section = (offset, _asset(match["asset"]))
        elif content.strip():
            direct = _CURRENT_STATEMENT.fullmatch(content)
            numbered = _NUMBER_FIRST.fullmatch(content) if paired_inventory else None
            na = _NA.fullmatch(content) if paired_inventory and section else None
            if direct or numbered:
                found = direct or numbered
                asset = _asset(found["asset"])
                scope = _scope(found["tail"])
                if scope is not None and (direct or section is None or section[1] == asset):
                    start = section[0] if numbered and section and section[1] == asset else offset
                    yield start, offset + len(content), found["asset"], found["value"], scope
            elif na:
                yield section[0], offset + len(content), section[1], na["value"], "unspecified"
            # An unrelated paragraph or next labelled subsection ends the asset
            # association. An empty line alone is ordinary PDF layout whitespace.
            if not (numbered or na):
                section = None
        offset += len(line)


def _observation(row, text, span):
    start, end, raw_asset, raw_value, scope = span
    quote = text[start:end]
    # No inferred page or fuzzy quotation: only a unique current source citation
    # covering the exact span can ground a candidate observation.
    citations = [c for c in row.get("citations", []) if c.get("role") == "current"
        and c.get("document_id") and c.get("page") is not None
        and (not row.get("source_document_id") or c["document_id"] == row["source_document_id"])
        and isinstance(c.get("quote"), str) and quote in c["quote"]]
    anchors = {(c["document_id"], c["page"], c.get("version")) for c in citations}
    if len(anchors) != 1 or not row.get("id"):
        return None
    citation = citations[0]
    asset = raw_asset if raw_asset in _ASSETS else _asset(raw_asset)
    non_numeric = raw_value.upper() == "N/A"
    value = None if non_numeric else (int(raw_value) if raw_value.isdigit() else _WORDS[raw_value.lower()])
    return {"asset": asset, "requirement_id": row["id"], "document_id": citation["document_id"],
        "title": citation.get("title", ""), "page": citation["page"], "role": "current",
        "version": citation.get("version"), "quote": quote, "source_start": start, "source_end": end,
        "source_sha256": sha256(text.encode()).hexdigest(), "value_kind": "not_applicable" if non_numeric else "count",
        "value": value, "raw_value": raw_value, "scope": scope}


def flag_buyer_scope_conflicts(requirements):
    """Deep-copy requirements and attach exact-source inventory discrepancy candidates.

    Only participating requirements receive ``source_conflicts`` and the review
    issue. Existing assessments and statuses are left for the workflow to handle.
    The helper does not resolve scope, precedence, or the meaning of N/A.
    """
    result = deepcopy(requirements)
    groups = {}
    for row in result:
        # Recompute our own derived data, preserving other issue/conflict kinds.
        old = row.get("source_conflicts", [])
        retained = [c for c in old if c.get("kind") != "candidate_inventory_discrepancy"]
        if retained:
            row["source_conflicts"] = retained
        else:
            row.pop("source_conflicts", None)
        if "buyer_context_issues" in row:
            row["buyer_context_issues"] = [i for i in row["buyer_context_issues"] if i != "buyer_scope_conflict"]
        text = row.get("after")
        if not isinstance(text, str) or not text.strip():
            continue
        seen = set()
        for span in _candidate_spans(row, text):
            observation = _observation(row, text, span)
            if observation is None:
                continue
            key = (observation["asset"], observation["document_id"], observation["page"],
                   observation["source_start"], observation["source_end"])
            if key in seen:
                continue
            seen.add(key)
            scope = observation["scope"]
            group = (observation["asset"], "total_candidate" if scope in {"all_locations", "unspecified"} else scope)
            groups.setdefault(group, []).append(observation)
    for (asset, _), observations in groups.items():
        if (len({o["document_id"] for o in observations}) < 2
                or len({(o["value_kind"], o["value"]) for o in observations}) < 2):
            continue
        conflict = {"kind": "candidate_inventory_discrepancy", "asset": asset,
                    "requires_review": True, "observations": observations}
        participating = {o["requirement_id"] for o in observations}
        for row in result:
            if row.get("id") in participating:
                row.setdefault("source_conflicts", []).append(deepcopy(conflict))
                issues = row.setdefault("buyer_context_issues", [])
                if "buyer_scope_conflict" not in issues:
                    issues.append("buyer_scope_conflict")
    return result
