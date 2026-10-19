# SPDX-License-Identifier: MPL-2.0
"""Compact exact buyer-source context for the isolated local evaluator."""
from __future__ import annotations

from copy import deepcopy
import re

from bidtrace_v2.obligations import _LIST_MARKER, _introduces_list_choice, _list_layout, _trim


_SINGLE_ENTITY_LIST = re.compile(
    r"\bfor\s+(?:one|1|(?:a\s+)?single)\s+(?!of\b)[\w-]+(?:\s+[\w-]+){0,3}\s*:\s*$", re.I)


def obligation_context(contract: dict, obligation: dict) -> dict:
    """Retain the containing item and governing intros, excluding sibling items.

    Paired questions retain their supplied provenance; this helper does not verify
    external documents, resolve referents, or infer supplier facts. Ambiguous item
    boundaries and alternatives retain the original source context.
    """
    text = contract["text"]
    selected_start, selected_end = obligation["source_start"], obligation["source_end"]
    if text[selected_start:selected_end] != obligation["source_text"]:
        raise ValueError("Selected obligation does not match its source span")

    def span(start, end):
        start, end = _trim(text, start, end)
        return {"text": text[start:end], "source_start": start, "source_end": end}

    governing = []

    def enclosing(start, end):
        siblings, list_end = _list_layout(text, start, end)
        if not siblings:
            return start, end
        if selected_start >= list_end:
            return enclosing(list_end, end)
        introduction = span(start, siblings[0].start())
        if _LIST_MARKER.search(introduction["text"]):
            if selected_end <= siblings[0].start():
                return enclosing(start, siblings[0].start())
            introduction = span(siblings[0].start(), siblings[0].start())
        if _introduces_list_choice(introduction["text"]):
            return start, end
        # Fields under an explicitly singular entity share that entity's scope.
        # Keep their exact source together; no entity name or condition is
        # inferred or rewritten, and the canonical checks remain independent.
        if _SINGLE_ENTITY_LIST.search(introduction["text"]):
            return start, end
        for index, marker in enumerate(siblings):
            item_end = siblings[index + 1].start() if index + 1 < len(siblings) else list_end
            if marker.end() <= selected_start and selected_end <= item_end:
                if introduction["text"]:
                    governing.append(introduction)
                return enclosing(marker.end(), item_end)
        return start, end

    start, end = enclosing(obligation["context_start"], obligation["context_end"])
    questions = [deepcopy(question) for question in contract.get("buyer_context", [])
                 if question.get("role") == "current" and question.get("relation") == "paired_question"]
    references = [deepcopy(item) for item in contract.get("buyer_context", [])
                  if item.get("role") == "current" and item.get("relation") != "paired_question"]
    return {"obligation_id": obligation["obligation_id"],
            "selected_span": span(selected_start, selected_end),
            "enclosing_span": span(start, end), "governing_spans": governing,
            "paired_buyer_questions": questions,
            "referenced_buyer_context": references,
            "source_sha256": contract["source_sha256"],
            "buyer_context_issues": list(contract.get("buyer_context_issues", []))}
