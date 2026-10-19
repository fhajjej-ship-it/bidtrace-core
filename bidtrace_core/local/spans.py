# SPDX-License-Identifier: MPL-2.0
"""Deterministic exact excerpt candidates; selection does not prove entailment."""
from __future__ import annotations

import re


def source_spans(text: str) -> list[str]:
    """Return bounded exact parents, lines, sentences and nearby context windows.

    Preserve original whitespace as well as trimmed variants. Parent spans retain
    available conditions; long parents also expose adjacent and maximal windows,
    rather than every possible combination. Nothing is normalized or paraphrased.
    """
    if not isinstance(text, str):
        raise TypeError("Source text must be a string")
    limit, result, seen = 600, [], set()

    def add(value):
        for candidate in (value, value.strip()):
            if candidate.strip() and len(candidate) <= limit and candidate not in seen:
                seen.add(candidate)
                result.append(candidate)

    def ranges(value, separator):
        start, parts = 0, []
        for match in re.finditer(separator, value):
            if value[start:match.start()].strip():
                parts.append((start, match.start()))
            start = match.end()
        if value[start:].strip():
            parts.append((start, len(value)))
        return parts

    def add_long_unit(value):
        # A single long line/sentence still needs selectable exact excerpts.
        # Prefer word boundaries; an overlong token falls back to a character cut.
        start = 0
        while start < len(value):
            end = min(start + limit, len(value))
            if end < len(value):
                boundaries = list(re.finditer(r"\s+", value[start:end]))
                if boundaries and boundaries[-1].end() > limit // 2:
                    end = start + boundaries[-1].end()
            add(value[start:end])
            start = end

    def add_units(parent, parts):
        for index, (start, end) in enumerate(parts):
            unit = parent[start:end]
            add(unit)
            if len(unit) > limit and "\n" not in unit and "\r" not in unit:
                add_long_unit(unit)
            if index + 1 < len(parts):
                add(parent[start:parts[index + 1][1]])
            following = index + 1
            while following < len(parts) and parts[following][1] - start <= limit:
                end = parts[following][1]
                following += 1
            add(parent[start:end])

    add(text)
    for start, end in ranges(text, r"(?:\r\n?|\n)[ \t]*(?:\r\n?|\n)"):
        paragraph = text[start:end]
        add(paragraph)
        add_units(paragraph, ranges(paragraph, r"\r\n?|\n"))
        add_units(paragraph, ranges(paragraph, r"(?<=[.!?])\s+"))
    return result
