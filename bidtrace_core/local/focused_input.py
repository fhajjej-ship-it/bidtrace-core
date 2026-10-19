# SPDX-License-Identifier: MPL-2.0
"""Isolated readable buyer-input projection; not imported by the active service.

Only selected/enclosing/governing source text, verified question/reference text
and matching local acronym definitions are exposed. Canonical contracts remain
unchanged. Source IDs resolve directly to strings, never character arithmetic.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from bidtrace_core.local.context import obligation_context

_DEFINITION = re.compile(r"(?m)^[ \t]*(?P<term>[A-Z][A-Z0-9/]{1,15}(?:[ \t]+(?:Type[ \t]+\d+[ \t]+Certification))?)[ \t]+[–—-][ \t]+\S[^\n]*")
_OTHER_DEFINITION = re.compile(r"(?m)^[ \t]*[A-Za-z][A-Za-z0-9/ ]{0,50}[ \t]+[–—-][ \t]+\S")
_SEMANTIC = ("role", "relation", "reference_target", "asset", "conflict_asset", "value_kind", "value", "raw_value", "scope")
_BACKWARD = re.compile(r"\b(?:those|these|such(?!\s+as\b)|above|aforementioned|foregoing|preceding)\b", re.I)
_PHRASE_STOP = frozenset("and or but must shall will would should may can could is are was were be been has have had to on for from with that which remain include includes".split())


def _word_root(word):
    word = word.casefold()
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def backward_reference_context(contract, obligation):
    """Bind explicit local backward pointers, with source receipts off-prompt.

    A uniquely named preceding block or the immediately preceding form-field
    block is retained. Multiple matching blocks are never resolved by guessing:
    all candidates remain visible with an explicit unresolved reason.
    """
    context = obligation_context(contract, obligation)
    text = contract["text"]
    enclosing = context["enclosing_span"]
    pointers = list(_BACKWARD.finditer(enclosing["text"]))
    model, receipts = [], []
    for pointer in pointers:
        pointer_start = enclosing["source_start"] + pointer.start()
        pointer_end = enclosing["source_start"] + pointer.end()
        # Paragraphs retain their complete nested list. A pointer inside a
        # paragraph can also use exact earlier text from that same paragraph.
        boundaries = [0, *(m.end() for m in re.finditer(r"\n[ \t]*\n+", text[:pointer_start])), pointer_start]
        blocks = []
        for start, end in zip(boundaries, boundaries[1:]):
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            # Exclude the current enclosing clause from antecedent candidates;
            # it is already supplied verbatim beside the selected condition.
            end = min(end, enclosing["source_start"])
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start and re.search(r"\w", text[start:end]):
                blocks.append({"text": text[start:end], "source_start": start, "source_end": end})
        following = enclosing["text"][pointer.end():]
        words = []
        for match in re.finditer(r"[A-Za-z0-9]+", following):
            between = following[:match.start()] if not words else following[previous_end:match.start()]
            if re.search(r"[.!?:;]", between) or match.group().casefold() in _PHRASE_STOP:
                break
            words.append(match.group())
            previous_end = match.end()
            if len(words) == 6:
                break
        candidates = []
        already_supplied = False
        reason, state = "local_backward_referent_not_found", "unresolved"
        if pointer.group().casefold() in {"those", "these", "such", "aforementioned", "foregoing", "preceding"} and words:
            head = _word_root(words[-1])
            prefix = enclosing["text"][:pointer.start()].rstrip()
            if head in {_word_root(word) for word in re.findall(r"[A-Za-z0-9]+", prefix)}:
                candidates = [{"text": prefix, "source_start": enclosing["source_start"],
                               "source_end": enclosing["source_start"] + len(prefix)}]
                reason, state = "named_referent_in_supplied_enclosing_clause", "resolved"
                already_supplied = True
            else:
                candidates = [block for block in blocks if head in {_word_root(word) for word in re.findall(r"[A-Za-z0-9]+", block["text"])}]
            if len(candidates) == 1 and not already_supplied:
                reason, state = "unique_named_preceding_block", "resolved"
            elif len(candidates) > 1:
                reason = "ambiguous_named_preceding_blocks"
        elif pointer.group().casefold() == "above" and blocks:
            current_field = re.search(r"^[^\n:]{1,120}:[ \t_]*_{3,}", enclosing["text"])
            previous_field = re.search(r"^[^\n:]{1,120}:[^\n]*_{3,}", blocks[-1]["text"])
            if current_field and previous_field:
                candidates = [blocks[-1]]
                reason, state = "adjacent_preceding_form_block", "resolved"
            elif re.search(r"\bimmediately\s+above\b", enclosing["text"], re.I):
                candidates = [blocks[-1]]
                reason, state = "explicit_immediately_preceding_block", "resolved"
        if not candidates and blocks:
            # No unique semantic binding was established. Retain preceding
            # source blocks rather than silently dropping potentially necessary
            # context or inventing a referent. Only explicit-pointer inputs pay
            # this conservative fallback cost.
            candidates = blocks
            reason = "unresolved_local_backward_reference"
        entry = {"pointer": pointer.group(), "state": state, "reason": reason,
                 "preceding_source_blocks": [] if already_supplied else [block["text"] for block in candidates]}
        if entry not in model:
            model.append(entry)
            receipts.append({"obligation_id": obligation["obligation_id"], "source_sha256": contract["source_sha256"],
                             "pointer_span": {"text": text[pointer_start:pointer_end], "source_start": pointer_start, "source_end": pointer_end},
                             "antecedent_spans": candidates, "state": state, "reason": reason})
    return {"model_context": model, "source_receipts": receipts}


def local_definitions(documents):
    result = []
    for document in documents:
        if document.get("role") != "current" or not document.get("active", True):
            continue
        for page in document.get("pages", []):
            text = page.get("text", "")
            if not isinstance(text, str):
                continue
            matches = list(_DEFINITION.finditer(text))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                blank = re.search(r"\n[ \t]*\n", text[match.end():end])
                if blank:
                    end = match.end() + blank.start()
                other = _OTHER_DEFINITION.search(text, match.end(), end)
                if other:
                    end = other.start()
                while end > match.start() and text[end - 1].isspace():
                    end -= 1
                start = match.start()
                while start < end and text[start].isspace():
                    start += 1
                result.append({"term": match.group("term"), "quote": text[start:end],
                               "document_id": document["id"], "page": page["page"],
                               "source_start": start, "source_end": end, "role": "current"})
    return result


def term_used(term, text):
    aliases = [term, *term.split("/")] if "/" in term else [term]
    for alias in aliases:
        pattern = r"\s+".join(re.escape(part) for part in alias.split())
        if alias.isupper() and " " not in alias:
            pattern += "s?"
        if re.search(r"(?<!\w)" + pattern + r"(?!\w)", text, re.I):
            return True
    return False


def reference_view(entry):
    result = {key: deepcopy(entry[key]) for key in _SEMANTIC if key in entry}
    result["quote"] = entry["quote"]
    origin = entry.get("reference_source")
    if isinstance(origin, dict):
        text = origin.get("reference_text")
        if isinstance(text, str) and text and text in origin.get("quote", ""):
            # Keep the exact reference expression, not its unrelated whole
            # containing requirement copied inside provenance metadata.
            result["reference_origin"] = {"quote": text, "target": origin.get("target")}
        elif isinstance(origin.get("quote"), str):
            # Unknown origin boundaries retain the supplied text conservatively.
            result["reference_origin"] = {"quote": origin["quote"], "target": origin.get("target")}
    return result


def applicable_references(contract, context):
    """Exclude a reference only when its exact origin proves sibling scope.

    Unknown origins remain visible. Nested references inherit their proven
    parent's applicability; paired-question origins apply to their answer.
    The canonical context and its provenance are never changed.
    """
    entries = context["referenced_buyer_context"]
    spans = [context["selected_span"], context["enclosing_span"], *context["governing_spans"]]

    def applies(entry, visiting=()):
        if id(entry) in visiting:
            return True
        origin = entry.get("reference_source")
        if not isinstance(origin, dict):
            return True
        quote, pointer = origin.get("quote"), origin.get("reference_text")
        if not isinstance(quote, str) or not isinstance(pointer, str) or not pointer:
            return True
        start, end = origin.get("start"), origin.get("end")
        if not (type(start) is int and type(end) is int and 0 <= start < end <= len(quote)
                and quote[start:end] == pointer):
            if quote.count(pointer) != 1:
                return True
            start, end = quote.index(pointer), quote.index(pointer) + len(pointer)
        text = contract["text"]
        if text and quote.count(text) == 1:
            offset = quote.index(text)
            return any(offset + span["source_start"] < end and offset + span["source_end"] > start
                       for span in spans)
        if any(item["quote"] == quote for item in context["paired_buyer_questions"]):
            return True
        parents = [item for item in entries if item is not entry and item.get("quote") == quote]
        if parents:
            return any(applies(parent, (*visiting, id(entry))) for parent in parents)
        return True

    return [entry for entry in entries if applies(entry)]


def project_condition(contract, obligation, *, definitions=(), section_title=None):
    """Self-contained buyer context to place beside one supplier candidate.

    No catalog IDs or character indices need model resolution. The caller keeps
    the full canonical contract for source checks and result validation.
    """
    context = obligation_context(contract, obligation)
    paired = [reference_view(entry) for entry in context["paired_buyer_questions"]]
    referenced = [reference_view(entry) for entry in applicable_references(contract, context)]
    selected = context["selected_span"]["text"]
    enclosing = context["enclosing_span"]["text"]
    governing = [span["text"] for span in context["governing_spans"]]
    local_context = backward_reference_context(contract, obligation)["model_context"]
    searchable = "\n".join([selected, enclosing, *governing, *(entry["quote"] for entry in [*paired, *referenced]),
                            *(text for entry in local_context for text in entry["preceding_source_blocks"])])
    selected_definitions, seen = [], set()
    for definition in definitions:
        key = (definition["term"], definition["quote"])
        if key not in seen and term_used(definition["term"], searchable):
            selected_definitions.append({"term": definition["term"], "quote": definition["quote"]})
            seen.add(key)
    result = {"role": "buyer_context_only", "obligation_id": obligation["obligation_id"],
              "selected_condition": selected, "enclosing_clause": enclosing, "governing_clauses": governing,
              "paired_buyer_questions": paired, "referenced_buyer_context": referenced,
              "local_definitions": selected_definitions,
              "local_backward_context": local_context,
              "requirement_type": obligation["requirement_type"],
              "named_features": deepcopy(obligation["named_features"]),
              "scope_review_reasons": deepcopy(obligation["decomposition_issues"]),
              "buyer_context_issues": deepcopy(context["buyer_context_issues"])}
    # Generated titles can themselves be truncated neighboring paragraphs.
    # Retain only a supplied short heading that exactly matches the source line.
    first_line = contract["text"].splitlines()[0].strip() if contract["text"].splitlines() else ""
    if isinstance(section_title, str) and section_title == first_line and len(section_title) <= 120:
        result["section_heading"] = section_title
    return result


def verify_condition(contract, obligation, result):
    expected = obligation_context(contract, obligation)
    assert result["obligation_id"] == obligation["obligation_id"]
    assert result["selected_condition"] == expected["selected_span"]["text"]
    assert result["enclosing_clause"] == expected["enclosing_span"]["text"]
    assert result["governing_clauses"] == [span["text"] for span in expected["governing_spans"]]
    assert result["paired_buyer_questions"] == [reference_view(entry) for entry in expected["paired_buyer_questions"]]
    assert result["referenced_buyer_context"] == [reference_view(entry) for entry in applicable_references(contract, expected)]
    for key in ("requirement_type", "named_features"):
        assert result[key] == obligation[key]
    assert result["scope_review_reasons"] == obligation["decomposition_issues"]
    assert result["buyer_context_issues"] == expected["buyer_context_issues"]
    backward = backward_reference_context(contract, obligation)
    assert result["local_backward_context"] == backward["model_context"]
    for receipt in backward["source_receipts"]:
        for span in [receipt["pointer_span"], *receipt["antecedent_spans"]]:
            assert span["text"] == contract["text"][span["source_start"]:span["source_end"]]
    assert "requirement_text" not in result and "source_spans" not in result
    return True


def project_buyer_input(contract, *, definitions=()):
    texts, text_ids, contexts, context_ids, conditions, scoped_text = {}, {}, {}, {}, [], []

    def intern(text):
        if text not in text_ids:
            identifier = f"S{len(texts) + 1}"
            text_ids[text] = identifier
            texts[identifier] = text
        return text_ids[text]

    def encode(value):
        if isinstance(value, dict):
            return {("quote_id" if key == "quote" else key): intern(item) if key == "quote"
                    else encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        return deepcopy(value)

    for obligation in contract["obligations"]:
        context = obligation_context(contract, obligation)
        selected, enclosing = context["selected_span"]["text"], context["enclosing_span"]["text"]
        governing = [span["text"] for span in context["governing_spans"]]
        paired = [reference_view(entry) for entry in context["paired_buyer_questions"]]
        referenced = [reference_view(entry) for entry in context["referenced_buyer_context"]]
        buyer = {"paired_buyer_questions": paired, "referenced_buyer_context": referenced,
                 "buyer_context_issues": deepcopy(context["buyer_context_issues"])}
        key = json.dumps(buyer, ensure_ascii=False, sort_keys=True)
        if key not in context_ids:
            identifier = f"B{len(contexts) + 1}"
            context_ids[key] = identifier
            contexts[identifier] = encode(buyer)
        conditions.append({"obligation_id": obligation["obligation_id"],
                           "selected_span_id": intern(selected), "enclosing_span_id": intern(enclosing),
                           "governing_span_ids": [intern(text) for text in governing],
                           "buyer_context_id": context_ids[key],
                           "requirement_type": obligation["requirement_type"],
                           "named_features": deepcopy(obligation["named_features"]),
                           "scope_review_reasons": deepcopy(obligation["decomposition_issues"])})
        scoped_text.extend([selected, enclosing, *governing, *(entry["quote"] for entry in [*paired, *referenced])])
    # Select definitions from actual selected context only, before adding any
    # definitions. A definition cannot pull in an unrelated definition recursively.
    searchable = "\n".join(scoped_text)
    selected_definitions, seen_definitions = [], set()
    for definition in definitions:
        key = (definition["term"], definition["quote"])
        if key not in seen_definitions and term_used(definition["term"], searchable):
            selected_definitions.append({"term": definition["term"], "quote_id": intern(definition["quote"])})
            seen_definitions.add(key)
    return {"requirement_role": "buyer_context_only",
            "context_reference_instruction": "S IDs resolve directly to exact text in source_spans. B IDs resolve "
                "in buyer_context_catalog. Assess each selected span with its enclosing span, governing spans, "
                "paired questions and resolved references. Local definitions clarify terms only. All are buyer "
                "context, never supplier evidence. No missing term, supplier fact or precedence is inferred.",
            "source_spans": texts, "buyer_context_catalog": contexts,
            "local_definitions": selected_definitions, "obligation_batch": conditions}


def decode_references(value, payload):
    if isinstance(value, dict):
        return {("quote" if key == "quote_id" else key): payload["source_spans"][item] if key == "quote_id"
                else decode_references(item, payload) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_references(item, payload) for item in value]
    return deepcopy(value)


def verify_projection(contract, payload):
    assert len(payload["obligation_batch"]) == len(contract["obligations"])
    assert len(payload["source_spans"].values()) == len(set(payload["source_spans"].values()))
    for obligation, actual in zip(contract["obligations"], payload["obligation_batch"]):
        expected = obligation_context(contract, obligation)
        assert actual["obligation_id"] == obligation["obligation_id"]
        for span in ("selected_span", "enclosing_span"):
            assert payload["source_spans"][actual[span + "_id"]] == expected[span]["text"]
        assert [payload["source_spans"][key] for key in actual["governing_span_ids"]] == [s["text"] for s in expected["governing_spans"]]
        buyer = decode_references(payload["buyer_context_catalog"][actual["buyer_context_id"]], payload)
        for category in ("paired_buyer_questions", "referenced_buyer_context"):
            assert buyer[category] == [reference_view(entry) for entry in expected[category]]
        assert buyer["buyer_context_issues"] == expected["buyer_context_issues"]
        assert actual["requirement_type"] == obligation["requirement_type"]
        assert actual["named_features"] == obligation["named_features"]
        assert actual["scope_review_reasons"] == obligation["decomposition_issues"]
    assert "requirement" not in payload and "requirement_text" not in payload
    return len(contract["obligations"])
