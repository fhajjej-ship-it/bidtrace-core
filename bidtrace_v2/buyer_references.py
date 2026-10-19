# SPDX-License-Identifier: MPL-2.0
"""Resolve explicit buyer references from selected sources in one buyer pack.

This materializes cited context only. It cannot establish supplier compliance,
interpret an answer, or substitute Q/A answers from another document/version.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import re

from bidtrace.engine import _clauses, _label_match, _pages, _QA_LABEL, _qa_info


REVIEW_ISSUES = frozenset({
    "buyer_reference_missing", "buyer_reference_ambiguous", "buyer_reference_cycle",
    "buyer_reference_source_missing", "buyer_reference_source_ambiguous",
})
REFERENCE_RELATIONS = frozenset({"referenced_question", "referenced_answer", "referenced_document"})
_REFERENCE = re.compile(
    r"(?<![\w-])(?:Q[- ]?|Question\s+)(?P<number>\d+(?:\.\d+)*)"
    r"(?:[.:](?P<joined>[a-z]\w*)|[.:]\s+(?P<sub>[a-z])\b|"
    r"(?P<compact>[a-z]\w*)|\((?P<paren>[a-z]\w*)\))?"
    r"(?P<suffix_list>(?:/[a-z]\w*)*)"
    r"(?!\w)", re.I,
)
_NAMED_REFERENCE = re.compile(
    r"\b(?:see|refer\s+to)\s+(?:the\s+)?(?P<name>"
    r"(?:appendix|annex)\s+[a-z0-9]+(?:\.\d+)*\b(?:,[ \t]*(?-i:[A-Z][A-Z0-9 &/-]+))?|"
    r"section\s+\d+(?:\.\d+)*\b|"
    r"report\s+(?:below|above|attached)\b(?:[ \t]*\([^\n)]+\))?|"
    r'[“\"][^”\"\n]{1,150}[”\"]|'
    r"abbreviations(?:\s+and\s+acronyms)?\b|deadlines\s+and\s+important\s+dates\b)", re.I,
)


def _identity(number, suffix=""):
    return "q-" + number + ("." + suffix.lower() if suffix else "")


def _references(text):
    """Yield exact reference tokens, excluding a passage's own opening label."""
    own = _QA_LABEL.match(text.splitlines()[0] if text else "")
    for match in _REFERENCE.finditer(text):
        if own and match.start() < own.start("title"):
            continue
        suffix = match["joined"] or match["sub"] or match["compact"] or match["paren"] or ""
        yield _identity(match["number"], suffix), match
        if suffix:
            for extra in match["suffix_list"].split("/")[1:]:
                yield _identity(match["number"], extra), match


def reference_targets(text):
    """Canonical explicit question and named identities, without repeats."""
    return list(dict.fromkeys(target for target, _ in _all_references(text)))


def _all_references(text):
    yield from _references(text)
    for match in _NAMED_REFERENCE.finditer(text):
        name = match["name"].strip('“”"')
        if _REFERENCE.fullmatch(name):
            continue  # Quoting a Q reference does not create a second target.
        yield "named:" + " ".join(name.casefold().split()), match


def _pointer_only(text):
    """Only a bare pointer is blocked; a mixed service section keeps a detail."""
    own = _label_match(text.splitlines()[0] if text else "")
    if own:
        text = text[own.start("title"):]
    spans = sorted({(match.start(), match.end()) for _, match in _all_references(text)}, reverse=True)
    if not spans:
        return False
    for start, end in spans:
        text = text[:start] + " " + text[end:]
    text = re.sub(r"\b(?:please|see|also|refer\s+to|and|or|for\s+(?:additional|more)\s+(?:information|details))\b", "", text, flags=re.I)
    return not text.strip(' \t\r\n.,:;()[]*-“”"')


@dataclass
class _Entry:
    kind: str
    target: str
    citations: list[dict]
    heading: bool = False


def _index(document):
    """Reuse the engine's exact page spans, including contextual Q/A answers.

    The engine emits each excluded Q/A's citation segments consecutively. A
    continuation segment therefore inherits only its own opening Q/A label,
    never a neighboring page or question guessed from its number.
    """
    context = []
    clauses = _clauses(document, document["role"], _context_out=context)
    entries = []
    for clause in clauses:
        if info := _qa_info(clause):
            entries.append(_Entry(info[0], _identity(*info[1:]), clause.citations))
    active = None
    active_kind = None
    for citation in context:
        kind = citation["context_kind"]
        match = _QA_LABEL.match(citation["quote"].splitlines()[0])
        if match:
            qa_kind = match["qa_kind"].lower()
            # A buyer's mislabeled Q answer is not a proven A relationship.
            if kind == "buyer_current_state_answer" and qa_kind != "a":
                active = None
                continue
            active = _Entry(qa_kind, _identity(match["qa_number"], match["qa_sub"] or ""), [],
                            heading=kind == "buyer_question_heading")
            entries.append(active)
            active_kind = kind
        elif kind != active_kind:
            active = None
        if active is not None:
            active.citations.append({key: value for key, value in citation.items() if key != "context_kind"})
    index = defaultdict(lambda: {"q": [], "a": []})
    for entry in entries:
        index[entry.target][entry.kind].append(entry)
    return dict(index)


def _selected_document(requirement, citation, documents):
    role = citation.get("role")
    candidates = [doc for doc in documents if doc.get("id") == citation.get("document_id")
                  and doc.get("role") == role and role in {"current", "previous"}
                  and (not requirement.get("logical_id") or doc.get("logical_id") == requirement["logical_id"])
                  and (role != "current" or not requirement.get("source_document_id")
                       or doc.get("id") == requirement["source_document_id"])]
    if len(candidates) != 1:
        return None, "buyer_reference_source_ambiguous" if candidates else "buyer_reference_source_missing"
    doc = candidates[0]
    pages = [text for page, text in _pages(doc) if page == citation.get("page")]
    quote = citation.get("quote", "")
    if len(pages) != 1 or not quote or quote not in pages[0]:
        return None, "buyer_reference_source_missing"
    return doc, None


def _target_entries(index, target):
    exact = index.get(target, {"q": [], "a": []})
    # A parent identity denotes a family only with an explicit heading. A
    # missing/duplicate label or a parent answer plus subparts is ambiguous.
    children = sorted(key for key in index if re.fullmatch(re.escape(target) + r"\.[a-z]", key))
    if children:
        if len(exact["q"]) != 1 or not exact["q"][0].heading or exact["a"]:
            return [], "buyer_reference_ambiguous"
        result = list(exact["q"])
        for child in children:
            entries, issue = _target_entries(index, child)
            if issue:
                return [], issue
            result.extend(entries)
        return result, None
    if len(exact["q"]) > 1 or len(exact["a"]) > 1:
        return [], "buyer_reference_ambiguous"
    if len(exact["q"]) != 1 or len(exact["a"]) != 1 or exact["q"][0].heading:
        return [], "buyer_reference_missing"
    return [exact["q"][0], exact["a"][0]], None


def _origin(citation, target, match):
    return {key: deepcopy(citation[key]) for key in ("document_id", "title", "page", "quote", "role")
            if key in citation} | {"target": target, "reference_text": match.group(),
                                  "start": match.start(), "end": match.end()}


def _document_citations(doc, pages):
    return [{"document_id": doc["id"], "title": doc.get("title", ""), "page": page,
             "quote": text, "role": doc["role"]} for page, text in pages if text.strip()]


def _named_citations(doc, target, source, documents):
    """Resolve only unambiguous local appendix labels or contiguous report pages.

    Cross-document targets must be active current buyer files in the same tender.
    Same-page prose merely mentioning an appendix is not a document heading.
    Report titles may insert a descriptive word (Hardware Lifecycle Report) into
    a named report (Hardware Report); every named word must occur in the heading.
    """
    name = target.removeprefix("named:")
    appendix = re.fullmatch(r"(appendix|annex)\s+([a-z0-9]+(?:\.\d+)*)(?:,\s*(.+))?", name)
    if appendix:
        label = re.compile(r"\b" + re.escape(appendix[1]) + r"\s+" + re.escape(appendix[2]) + r"\b", re.I)
        title_words = set(re.findall(r"\w+", appendix[3] or ""))
        matches = []
        for candidate in documents:
            if (candidate.get("role") != "current" or candidate.get("active") is False
                    or doc.get("role") != "current" or not doc.get("tender_id")
                    or candidate.get("tender_id") != doc["tender_id"]):
                continue
            pages = _pages(candidate)
            if not pages:
                continue
            heading = "\n".join(pages[0][1].splitlines()[:4])
            # An explicit appendix label must occur in the actual first-page header.
            if not label.search(heading) or re.search(r"\b(?:see|refer\s+to)\s+(?:the\s+)?" + label.pattern, heading, re.I):
                continue
            if title_words and not title_words <= set(re.findall(r"\w+", heading.casefold())):
                continue
            matches.append(_document_citations(candidate, pages))
        if len(matches) == 1:
            return matches[0], None
        return [], "buyer_reference_ambiguous" if matches else "buyer_reference_missing"
    report = re.fullmatch(r"report\s+below(?:\s*\(([^)]+)\))?", name)
    if report:
        # Parenthetical titles may include an RFP/file prefix separated by ' - '.
        hint = (report[1] or "").split(" - ")[-1]
        words = set(re.findall(r"\w+", hint)) - {"report", "the", "a", "an"}
        heading_pattern = re.compile(r"^[ \t]*(?P<title>(?:[A-Z][A-Za-z0-9&/-]*[ \t]+){0,7}(?:Report|REPORT))\b")
        groups = []
        for page, text in _pages(doc):
            if page <= source.get("page", 0):
                continue
            headings = [m["title"] for line in text.splitlines()[:8] if (m := heading_pattern.match(line))]
            headings = [h for h in headings if words <= set(re.findall(r"\w+", h.casefold()))]
            if len(headings) != 1:
                continue
            heading = " ".join(headings[0].casefold().split())
            if groups and groups[-1]["heading"] == heading and groups[-1]["pages"][-1][0] + 1 == page:
                groups[-1]["pages"].append((page, text))
            else:
                groups.append({"heading": heading, "pages": [(page, text)]})
        if len(groups) == 1:
            return _document_citations(doc, groups[0]["pages"]), None
        return [], "buyer_reference_ambiguous" if groups else "buyer_reference_missing"
    return [], "buyer_reference_missing"


def _cycle_has_independent_answers(index, path, target):
    """A fully present contextual cycle is navigational, not a missing answer."""
    for node in path[path.index(target):]:
        entries, issue = _target_entries(index, node)
        if issue:
            return False
        answers = ["\n".join(c["quote"] for c in entry.citations) for entry in entries if entry.kind == "a"]
        for answer in answers:
            opening = _label_match(answer.splitlines()[0])
            content = answer[opening.start("title"):] if opening else answer
            if _pointer_only(answer) or not re.search(r"[A-Za-z0-9]", content):
                return False
        if not answers:
            return False
    return True


def enrich_buyer_references(requirements, documents):
    """Return deep-copied requirements with explicitly referenced source context.

    Generated citations carry ``reference_target`` (the attached Q/A identity),
    ``reference_path`` (root to target), and ``reference_source`` (exact passage
    and token that named it). Unresolved edges are listed in
    ``buyer_reference_issues`` with the same provenance. Current unresolved
    edges also add a REVIEW_ISSUES code to ``buyer_context_issues``; previous
    failures remain historical when a current requirement exists.

    Named appendix documents resolve only by a unique label/title in the active
    current buyer pack. Named report-below pointers resolve only to a unique
    contiguous report in later pages of the selected document. Other named
    pointers remain unresolved. No applicability or value precedence is inferred.

    Re-enrichment rebuilds only generated references/issues, so it is idempotent
    and cannot keep stale context after the selected source changes.
    """
    result = deepcopy(requirements)
    indexes = {}
    for requirement in result:
        context = [c for c in requirement.get("buyer_context", []) if c.get("relation") not in REFERENCE_RELATIONS]
        issues = [issue for issue in requirement.get("buyer_context_issues", []) if issue not in REVIEW_ISSUES]
        details = []
        generated = []

        def record_issue(issue, target, path, source, *, blocking=True):
            detail = {"issue": issue, "target": target, "reference_path": path, "reference_source": source}
            if detail not in details:
                details.append(detail)
            effective_role = "current" if requirement.get("after") is not None else "previous"
            if blocking and source.get("role") == effective_role and issue not in issues:
                issues.append(issue)

        def resolve(doc, target, path, source):
            if target.startswith("named:"):
                citations, issue = _named_citations(doc, target, source, documents)
                if issue:
                    field = "after" if source.get("role") == "current" else "before"
                    record_issue(issue, target, path + [target], source,
                                 blocking=_pointer_only(requirement.get(field) or ""))
                else:
                    for citation in citations:
                        attached = {**deepcopy(citation), "relation": "referenced_document",
                                    "reference_target": target, "reference_path": path + [target],
                                    "reference_source": deepcopy(source)}
                        if attached not in generated:
                            generated.append(attached)
                return
            key = (doc["id"], doc["role"])
            if key not in indexes:
                indexes[key] = _index(doc)
            if target in path:
                resolved = _cycle_has_independent_answers(indexes[key], path, target)
                record_issue("buyer_reference_cycle_resolved" if resolved else "buyer_reference_cycle",
                             target, path + [target], source, blocking=not resolved)
                return
            path = path + [target]
            entries, issue = _target_entries(indexes[key], target)
            if issue:
                record_issue(issue, target, path, source)
                return
            for entry in entries:
                entry_path = path if entry.target == target else path + [entry.target]
                for citation in entry.citations:
                    attached = {**deepcopy(citation), "relation": "referenced_question" if entry.kind == "q" else "referenced_answer",
                                "reference_target": entry.target, "reference_path": entry_path,
                                "reference_source": deepcopy(source)}
                    if attached not in generated:
                        generated.append(attached)
                    for nested, match in _all_references(citation["quote"]):
                        resolve(doc, nested, entry_path, _origin(citation, nested, match))

        for role, field in (("current", "after"), ("previous", "before")):
            text = requirement.get(field) or ""
            paired = [c for c in context if c.get("relation") == "paired_question" and c.get("role") == role]
            wanted = {target for target, _ in _all_references(text)}
            wanted.update(target for cite in paired for target, _ in _all_references(cite.get("quote", "")))
            if not wanted:
                continue
            own = _QA_LABEL.match(text.splitlines()[0] if text else "")
            path = [_identity(own["qa_number"], own["qa_sub"] or "")] if own else []
            origins = [c for c in requirement.get("citations", []) if c.get("role") == role] + paired
            found = set()
            for citation in origins:
                for target, match in _all_references(citation.get("quote", "")):
                    if target not in wanted:
                        continue
                    found.add(target)
                    source = _origin(citation, target, match)
                    doc, issue = _selected_document(requirement, citation, documents)
                    if issue:
                        record_issue(issue, target, path + [target], source,
                                     blocking=not target.startswith("named:") or _pointer_only(text))
                    else:
                        resolve(doc, target, path, source)
            for target in sorted(wanted - found):
                record_issue("buyer_reference_source_missing", target, path + [target],
                             {"role": role, "document_id": requirement.get("source_document_id"), "target": target},
                             blocking=not target.startswith("named:") or _pointer_only(text))
        requirement["buyer_context"] = context + generated
        requirement["buyer_context_issues"] = issues
        if details:
            requirement["buyer_reference_issues"] = details
        else:
            requirement.pop("buyer_reference_issues", None)
    return result


def validate_buyer_reference(requirement, documents, citation):
    """Raise ValueError unless this is a complete canonical resolved citation.

    Page membership alone does not prove Q/A identity or the reference chain.
    Rebuild from original requirement citations and paired questions, ignoring
    supplied generated context. Equality binds continuation spans, selected
    document identity/role, origin token offsets, and every step of the path.
    Only trusted display metadata (title, version, source_url) is excluded from
    comparison; the document ID still binds the selected version. No input is
    changed and no external resource is consulted.
    """
    if not isinstance(citation, dict) or citation.get("relation") not in REFERENCE_RELATIONS:
        raise ValueError("Buyer context reference is not a canonical resolved citation.")

    def identity(value):
        retained = {key: item for key, item in value.items() if key not in {"title", "version", "source_url"}}
        if isinstance(retained.get("reference_source"), dict):
            retained["reference_source"] = identity(retained["reference_source"])
        return retained

    expected = enrich_buyer_references([requirement], documents)[0]["buyer_context"]
    actual = identity(citation)
    if not any(candidate.get("relation") in REFERENCE_RELATIONS and identity(candidate) == actual for candidate in expected):
        raise ValueError("Buyer context reference does not match its canonical target, source, and path.")
