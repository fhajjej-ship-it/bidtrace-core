# SPDX-License-Identifier: MPL-2.0
"""Deterministic source spans for a conservative, auditable assessment contract.

This is a lexical decomposition, not a claim to fully parse procurement language.
Conjunction fragments retain their complete sentence as context. Local alternatives
stay together; unresolved relative baselines and overflow remain reasons to abstain.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import json
import re

ASSESSMENT_CONTRACT = "source_obligations_v1"
DECOMPOSITION_VERSION = "deterministic_source_spans_v11_url_literals"
MAX_OBLIGATIONS = 128
OBLIGATIONS_PER_BATCH = 3
SCHEMA_RETRIES_PER_BATCH = 1
GENERATION_ATTEMPTS_PER_BATCH = SCHEMA_RETRIES_PER_BATCH + 1

_METADATA = re.compile(r"^\s*(Requirement type|Weighting)\s*:\s*(.*?)\s*$", re.I)
# Match the generic parser's labelled headings, without importing its runtime.
_TITLE_LABEL = re.compile(
    r"^\s*(?:#{1,6}\s+)?(?P<label>(?:[A-Z][A-Z0-9]{0,11}[-_ ]?)?"
    r"\d+(?:\.\d+)*)(?:[.)]\s+|\s*[|:—–]\s*|\s+-\s+|$)(?P<title>.*)$", re.I)
_NORMATIVE_TITLE = re.compile(
    r"\b(?:must|shall|should|will|requires?|required|needs?\s+to|has\s+to|have\s+to)\b"
    r"|\b(?:is|are|be)\s+(?:mandatory|optional|preferred)\b", re.I)
_EXAMPLE = re.compile(r"\b(?:e\.?g\.?|for example|such as)\b", re.I)
_LIST_MARKER = re.compile(r"^(?P<indent>[ \t]*)(?P<marker>[-*•▪▫●‣\uf0b7]|\d+[.)]|[ivxlcdm]+[.)]|[a-z][.)]|o)(?=[ \t])[ \t]+", re.M)
_LIST_CHOICE = re.compile(r"\b(?:(?:one|any)\s+of|either|following\s+alternatives)\b", re.I)
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_TIME = r"\d{1,2}:\d{2}(?:\s*[AP]\.?M\.?)?(?:\s+(?-i:[A-Z]{2,5})\b)?"
_DATE_LITERAL = re.compile(
    rf"\b(?:(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?"
    rf"{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}"
    rf"(?:,?\s+(?:(?:at|by|Time:)\s*)?{_TIME})?", re.I)
_ADDRESS_LITERAL = re.compile(
    r"(?:(?:addressed|sent|delivered|submitted)\s+to:\s*[^.!?;]{0,180}?)?"
    r"\b\d{1,6}\s+(?:[\w'-]+[ \t]+){1,6}"
    r"(?:Street|St\.?|Road|Rd\.?|Avenue|Ave\.?|Lane|Ln\.?|Drive|Dr\.?|Boulevard|Blvd\.?)"
    r"(?:,?\s+[A-Z][A-Za-z'-]*(?:[ \t]+[A-Z][A-Za-z'-]*){0,3}),\s*"
    r"[A-Z]{2},?\s*\d{5}(?:-\d{4})?\b")
_URL_LITERAL = re.compile(r"\b(?:https?://|www\.)[^\s]+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}/[^\s]*(?:\n[ \t]*[a-z0-9/_-]+(?:/|\.))?", re.I)
_ABBREVIATION = re.compile(r"\b(?:[ap]\.m\.|e\.g\.|i\.e\.|mr\.|mrs\.|ms\.|dr\.|st\.|rd\.|ave\.)$", re.I)
_FEATURE = re.compile(
    r"(?<!\w)(?:[A-Z]{2,}\s+\d+(?:[.-]\d+)*|[A-Z](?:[A-Za-z]*\d|\.\d)[A-Za-z0-9.-]*"
    r"|[A-Z]{2,}[A-Za-z0-9]*(?:[./-][A-Za-z0-9]+)*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)(?!\w)")


def is_hard_literal_anchor(feature):
    """Only numbered/versioned identifiers justify an exact-token override.

    Capitalization alone cannot distinguish a contextual acronym from a required
    capability. Bare candidates stay in the model's source context; their absence
    from a selected quote does not deterministically overturn its assessment.
    """
    return isinstance(feature, str) and re.search(r"\d", feature) is not None


def _type(value):
    value = str(value or "").strip().casefold()
    if value in {"mandatory", "required"}:
        return "mandatory"
    if value in {"evaluation", "optional", "preference", "preferred"}:
        return "preference"
    return "unspecified"


def summarize_type(types):
    types = set(types)
    if {"mandatory", "preference"} <= types:
        return "mixed"
    return next(iter(types)) if len(types) == 1 else "unspecified"


def _trim(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _first_line_metadata(line, title, has_body):
    """A generated display title is not evidence that source prose is metadata."""
    if not isinstance(title, str) or not title.strip():
        return None
    labelled = _TITLE_LABEL.fullmatch(line.rstrip("\r\n"))
    candidate = labelled["title"].strip() if labelled else line.strip()
    # Structured rows keep the identifier in their supplied title; generic rows
    # supply only the title text. Both must bind to this exact source line.
    if title.strip() not in {line.strip(), candidate}:
        return None
    # Q/A labels identify content, including non-normative current-state answers.
    # Other matched headings must look like short noun phrases, not sentences or
    # a wrapped first line. Ambiguity retains source text for review.
    answer_label = labelled and re.fullmatch(r"[QA][-_ ]?\d+", labelled["label"], re.I)
    substantive = (answer_label or _NORMATIVE_TITLE.search(candidate)
        or re.search(r"\b(?:is|are|was|were|would|could|can|has|have|does|do)\b|[,.;!?]", candidate, re.I)
        or len(candidate.split()) > 12)
    if has_body and not substantive:
        return len(line), "requirement_title"
    if labelled:
        # The identifier is metadata even when its line contains the obligation.
        # The remaining text starts at its exact original character position.
        return labelled.start("title"), "requirement_label"
    return None


def _strength(text, inherited="unspecified", declared="unspecified"):
    if re.search(r"\b(?:optional|preference|preferred|advantageous)\b|\bnot\s+(?:mandatory|required)\b", text, re.I):
        return "preference"
    # A row explicitly classed as an evaluation criterion is not transformed
    # into a minimum compliance condition merely by a modal in its description.
    if declared == "preference" and not re.search(r"\bmandatory\b", text, re.I):
        return "preference"
    if re.search(r"\b(?:must|shall|required|mandatory)\b", text, re.I):
        return "mandatory"
    if re.search(r"\bshould\b", text, re.I):
        return "mandatory" if inherited == "mandatory" else "preference"
    return None


def _outside_delimiters(text, position):
    prefix = text[:position]
    return (prefix.count('"') % 2 == 0 and prefix.count("“") == prefix.count("”")
            and prefix.count("(") == prefix.count(")"))


def _inside_literal(text, position):
    """Punctuation inside a cited fact is not a new supplier condition."""
    for pattern in (_DATE_LITERAL, _ADDRESS_LITERAL, _URL_LITERAL):
        for match in pattern.finditer(text):
            end = match.end()
            if pattern is _URL_LITERAL:
                # A sentence after a URL is still a separate duty.
                end -= len(match.group()) - len(match.group().rstrip('.,;:!?)'))
            if match.start() <= position < end:
                return True
    return False


def _introduces_list_choice(introduction):
    # Alternatives in earlier prose cannot turn unrelated subsequent bullets
    # into one permitted choice. Bind the cue to the actual list introduction.
    tail = re.split(r"[.!?]\s+|\n[ \t]*\n", introduction.strip().rstrip(".!?"))[-1]
    return bool(_LIST_CHOICE.search(tail))


def _list_layout(text, start, end):
    """Share PDF list boundaries with context extraction without rewriting text.

    Extraction can trim the first line's indent. Restore its structural indent
    only when repeated markers of that exact kind consistently establish it.
    Nested same-marker lists with multiple observed levels remain unchanged.
    """
    all_markers = list(_LIST_MARKER.finditer(text))
    indents = {marker.start(): len(marker["indent"].expandtabs(4)) for marker in all_markers}
    # A complete alphabetical sequence supplies stronger sibling evidence than
    # drifting PDF whitespace. Repeated/restarted alpha sequences are ambiguous
    # and retain their observed indentation. Roman child sequences stay separate.
    labels = [marker["marker"].rstrip(".)") for marker in all_markers]
    by_letter = {}
    for marker, label in zip(all_markers, labels):
        if marker["marker"].endswith((".", ")")) and len(label) == 1 and label.isalpha():
            by_letter.setdefault(label, []).append(marker)
    alphabetical = []
    ambiguous = False
    for label in 'abcdefghijklmnopqrstuvwxyz':
        if label not in by_letter:
            break
        candidates = by_letter[label]
        # Repeated i/v/x/etc. can be Roman children. A unique shallower marker
        # may be an alphabetic parent, but repeated alphabetic lists are not
        # repaired from whitespace guesses.
        minimum = min(indents[marker.start()] for marker in candidates)
        shallowest = [marker for marker in candidates if indents[marker.start()] == minimum]
        if len(candidates) > 1 and (label not in "ivxlcdm" or len(shallowest) != 1):
            ambiguous = True
            break
        alphabetical.append(shallowest[0])
    alpha_labels = {marker["marker"][0] for marker in alphabetical}
    # Roman-only child labels need not appear in the parent alphabet. They must
    # be indented beneath a preceding candidate parent; otherwise do not guess.
    for label, candidates in by_letter.items():
        if label in alpha_labels:
            continue
        for marker in candidates:
            parents = [parent for parent in alphabetical if parent.start() < marker.start()]
            if (label not in 'ivxlcdm' or not parents
                    or indents[marker.start()] <= indents[parents[-1].start()]):
                ambiguous = True
    ordered = sorted(alphabetical, key=lambda marker: marker.start())
    if alphabetical != ordered:
        ambiguous = True
    values = [ord(marker["marker"][0]) - ord("a") for marker in alphabetical]
    if not ambiguous and len(values) >= 2 and values == list(range(len(values))):
        sibling_indent = min(indents[marker.start()] for marker in alphabetical)
        indents.update({marker.start(): sibling_indent for marker in alphabetical})
    else:
        alphabetical = []

    # Explicit consecutive Roman labels identify siblings even when extracted
    # indentation drifts. Do not promote an alphabetic i/l/etc. into this run.
    alpha_positions = {marker.start() for marker in alphabetical}
    roman_sequence = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
                      "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx")
    index = 0
    while index < len(all_markers):
        run = []
        while (index + len(run) < len(all_markers) and len(run) < len(roman_sequence)
               and all_markers[index + len(run)].start() not in alpha_positions
               and labels[index + len(run)] == roman_sequence[len(run)]):
            run.append(all_markers[index + len(run)])
        if len(run) >= 2:
            sibling_indent = min(indents[marker.start()] for marker in run)
            indents.update({marker.start(): sibling_indent for marker in run})
        index += max(1, len(run))
    first_indent = None
    if all_markers and all_markers[0].start() == 0 and not all_markers[0]["indent"]:
        later = [len(marker["indent"].expandtabs(4)) for marker in all_markers[1:]
                 if marker["marker"] == all_markers[0]["marker"]]
        if len(later) >= 2 and len(set(later)) == 1 and later[0] >= 4:
            first_indent = later[0]
    matches = [marker for marker in all_markers if start <= marker.start() and marker.end() <= end]
    if not matches:
        return [], end
    def indent(marker):
        return first_indent if marker.start() == 0 and first_indent is not None else indents[marker.start()]
    minimum = min(indent(marker) for marker in matches)
    siblings = [marker for marker in matches if indent(marker) == minimum]
    # A less-indented prose paragraph ends this list even if a later paragraph
    # starts another list using the same glyph. It supplies its own introduction.
    list_end = end
    for gap in re.finditer(r"\n[ \t]*\n(?P<indent>[ \t]*)(?=\S)", text[siblings[0].end():end]):
        paragraph = siblings[0].end() + gap.start("indent")
        if (len(gap["indent"].expandtabs(4)) <= minimum
                and not _LIST_MARKER.match(text, paragraph)):
            list_end = siblings[0].end() + gap.start()
            break
    return [marker for marker in siblings if marker.start() < list_end], list_end


def _clause_cuts(sentence):
    """Split outer actions first so an inner alternative cannot swallow the list.

    Gerund-led clauses are a generic action-list boundary (creating, deleting,
    resetting). Alternative adjectives inside an action stay together. Every
    fragment still carries the complete sentence and is never rewritten.
    """
    # Blank form labels are one completion field, not comma-separated service
    # capabilities. Discourse prefixes qualify the following duty; they cannot
    # stand alone as a supplier commitment.
    if re.fullmatch(r"[\w ,/()-]+:\s*[_ .…]+", sentence):
        return [0, len(sentence)]
    prefix = re.match(r"As\s+(?:mentioned|noted|stated|specified|described)\s+in\s+[^,;.!?]+,\s*",
                      sentence, re.I)
    buyer_name = re.match(
        r"(?:(?:PURPOSE|BACKGROUND)(?:/BACKGROUND)?:\s*)?(?:The\s+)?(?:Town|City|Council)\s+of\s+"
        r"[\w' -]+,\s*[\w -]+\s*(?:\(the\s+[\"“][^\"”]+[\"”]\)\s*)?is\s+", sentence, re.I)
    attached_prefix_end = max(prefix.end() if prefix else 0, buyer_name.end() if buyer_name else 0)
    outer = [0]
    # A standard service, an offered extension and its pricing are distinct
    # duties. A trailing availability qualifier belongs to its own price duty,
    # not to every service earlier in the sentence. Preserve exact source spans
    # and enclosing context rather than projecting a compound Unknown later.
    for match in re.finditer(r"\bwith\s+an?\s+option\s+for\b|\band\s+(?=(?:the\s+)?cost\s+differential\b)", sentence, re.I):
        if _outside_delimiters(sentence, match.start()) and not _inside_literal(sentence, match.start()):
            outer.append(match.end() if match.group().lower().startswith('and') else match.start())
    # A colon makes the following enumeration explicit even when a conditional
    # governs the introduction. Keep that conditional in the sentence context.
    colon = re.search(r":\s+", sentence)
    if (colon and _outside_delimiters(sentence, colon.start())
            and re.search(r"\b(?:provide|include|submit|contain|following|required)\b", sentence[:colon.start()], re.I)):
        if _LIST_CHOICE.search(sentence[:colon.start()]):
            return [0, len(sentence)]
        outer.append(colon.end())
    for match in re.finditer(r";\s*|,\s*(?:and\s+)?(?=[A-Za-z]+ing\b)|\band\s+(?=[A-Za-z]+ing\b)", sentence, re.I):
        if _outside_delimiters(sentence, match.start()) and not _inside_literal(sentence, match.start()):
            outer.append(match.end())
    outer = sorted(set([*outer, len(sentence)]))
    cuts = [0]
    # Explicit parenthetical entity lists need independent checks. Retain
    # their action and qualifications in the enclosing source, and retain
    # every separator in a selected span for complete engineering coverage.
    for match in re.finditer(r"\(([^()]+,[^()]+)\)", sentence):
        if not re.search(r"\b(?:or|either|e\.?g\.?|such as)\b", match[1], re.I):
            cuts.extend(match.start(1) + comma.end() for comma in re.finditer(r",\s*", match[1]))
    # Slash-separated noun objects, unlike configuration/support actions or
    # and/or, can name independent requested entities.
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9-]+)/([A-Za-z][A-Za-z0-9-]+)\b", sentence):
        left, right = match[1], match[2]
        if (not _inside_literal(sentence, match.start(2))
                and any(c.islower() for c in left) and any(c.islower() for c in right)
                and left.casefold() not in {'and', 'configuration', 'configure', 'sales', 'support', 'recovery', 'monitoring'}
                and right.casefold() not in {'or', 'support', 'configuration', 'configure', 'restoration', 'management'}):
            cuts.append(match.start(2))
    for begin, end in zip(outer, outer[1:]):
        clause = sentence[begin:end]
        # A local disjunction, conditional or bracketed scope is evaluated as one
        # check. It does not suppress independent outer action clauses.
        scoped = re.search(r"\bor\b|\b(?:if|unless|except)\b", clause, re.I)
        if not scoped:
            # Two complete multiword service objects joined by & are distinct
            # duties (e.g. hardware monitoring & security reporting). Elided
            # shared objects such as application Updates & Maintenance stay
            # together. Exact offsets and the governing clause are retained.
            heads = r'(?:Management|Ordering|Protection|Filtering|Monitoring|Maintenance|Reports|Support|Services|Training|Parts)'
            match = re.fullmatch(rf'\s*\w[\w /-]*\s+{heads}\s*&\s*(?P<second>\w[\w /-]*\s+{heads})\s*[.]?\s*', clause, re.I)
            if match and _outside_delimiters(clause, match.start('second')):
                cuts.append(begin + match.start('second'))
            # Coordinated report types share the final noun but remain separate
            # duties. Retain the connective in the left exact source fragment;
            # the enclosing clause supplies "Reports" to its elided modifier.
            for match in re.finditer(r"\b[A-Za-z][\w-]*\s*&\s*(?P<second>[A-Za-z][\w-]*)\s+reports?\b", clause, re.I):
                if _outside_delimiters(clause, match.start('second')):
                    cuts.append(begin + match.start('second'))
            cuts.extend(begin + match.end() for match in
                        re.finditer(r"[,;](?:\s+(?:and|as well as)\b)?(?=\s)|\b(?:and|as well as)\s+", clause, re.I)
                        if _outside_delimiters(clause, match.start()) and not _inside_literal(clause, match.start()))
        cuts.append(end)
    # A dangling connective before the next explicit bullet qualifies the
    # preceding item; it is not an independently assessable source condition.
    return [cut for cut in sorted(set(cuts)) if cut in {0, len(sentence)} or
            (cut > attached_prefix_end and
             not re.fullmatch(r"\s*(?:and|as\s+well\s+as)[\s,;:]*", sentence[cut:], re.I))]


def _list_units(text, start, end, exclusions, inherited, declared, parent_start=None):
    """Recognize explicit list structure before interpreting sentence punctuation.

    List markers are recorded metadata. Child spans retain a contiguous context
    from the governing introduction through their complete item, including its
    conditions. Each sibling inherits from the parent, never an optional sibling.
    """
    siblings, list_end = _list_layout(text, start, end)
    if not siblings:
        yield start, end, parent_start, end, inherited, False
        return
    first = siblings[0].start()
    # Unindented prose after a blank line ends this explicit list. It must not
    # become another choice or inherit the preceding list's strength.
    introduction = text[start:first]
    # A leading list can precede a later, less indented section. It is a
    # separate list, never governing prose for that later section.
    if _LIST_MARKER.search(introduction):
        yield from _list_units(text, start, first, exclusions, inherited, declared, parent_start)
        start, introduction = first, ""
    # An independent bullet is not governed by every earlier paragraph on the
    # page. Bind only an actual list introduction; retain inherited ancestors
    # for nested/conditional lists. A closed unrelated duty before a blank line
    # must not turn a later policy description into a supplier obligation.
    breaks = list(re.finditer(r"\n[ \t]*\n", introduction.rstrip()))
    local_start = start + (breaks[-1].end() if breaks else 0)
    local_introduction = text[local_start:first].strip()
    scope_heading = re.search(r"^[ \t]*SCOPE OF (?:SERVICES|WORK)[ \t]*$", introduction, re.I | re.M)
    governs = bool(scope_heading or local_introduction and (
        local_introduction.endswith(':') or _introduces_list_choice(local_introduction)
        or re.search(r"\b(?:the\s+following|following\s+(?:services|requirements|items|information|documents|"
                     r"capabilities|features)|listed\s+below|specified\s+below)\b", local_introduction, re.I)
        or (_NORMATIVE_TITLE.search(local_introduction) and not re.search(r"[.!?]$", local_introduction))))
    unrelated_prose = bool(local_introduction and re.search(r"[.!?]$", local_introduction) and not governs)
    # Bare service lists and their section headings keep the existing shared
    # scope. Only a closed, non-governing prose paragraph is separated from
    # subsequent independent bullets.
    context_start = (parent_start if parent_start is not None else None if unrelated_prose else start)
    strength = (None if unrelated_prose else _strength(introduction, inherited, declared)) or inherited
    if _introduces_list_choice(introduction):
        yield start, list_end, context_start, list_end, strength, True
    else:
        if introduction.strip():
            yield start, first, parent_start, first, inherited, False
        for index, marker in enumerate(siblings):
            item_end = siblings[index + 1].start() if index + 1 < len(siblings) else list_end
            exclusions.append({"source_start": marker.start(), "source_end": marker.end(), "reason": "list_marker"})
            yield from _list_units(text, marker.end(), item_end, exclusions, strength, declared, context_start)
    if list_end < end:
        yield from _list_units(text, list_end, end, exclusions, declared, declared)


def decompose_requirement(requirement, max_obligations=MAX_OBLIGATIONS):
    # Cache parsing only, never retrieval, assessments or model results. The
    # complete input (including buyer context) binds the cache; return a private
    # copy because callers legitimately select different subsets of obligations.
    if not isinstance(requirement, (str, dict)):
        raise ValueError('Requirement must supply source text')
    if not isinstance(max_obligations, int) or isinstance(max_obligations, bool) or max_obligations < 1:
        raise ValueError('max_obligations must be a positive integer')
    encoded = json.dumps(requirement, ensure_ascii=False, sort_keys=True)
    return deepcopy(_cached_decomposition(encoded, max_obligations))


@lru_cache(maxsize=128)
def _cached_decomposition(encoded, max_obligations):
    return _decompose_requirement(json.loads(encoded), max_obligations)


def _decompose_requirement(requirement, max_obligations=MAX_OBLIGATIONS):
    if not isinstance(max_obligations, int) or isinstance(max_obligations, bool) or max_obligations < 1:
        raise ValueError("max_obligations must be a positive integer")
    if isinstance(requirement, str):
        text, title, declared = requirement, "", "unspecified"
        buyer_context = []
        buyer_context_issues = []
    elif isinstance(requirement, dict):
        text = requirement.get("text", requirement.get("after"))
        title, declared = requirement.get("title", ""), _type(requirement.get("requirement_type"))
        buyer_context = deepcopy(requirement.get("buyer_context") or [])
        if not isinstance(buyer_context, list) or any(not isinstance(item, dict) for item in buyer_context):
            raise ValueError("Buyer context must be a list of source citation objects")
        buyer_context_issues = requirement.get("buyer_context_issues") or []
        if not isinstance(buyer_context_issues, list) or any(not isinstance(issue, str) for issue in buyer_context_issues):
            raise ValueError("Buyer context issues must be a list of review reason strings")
        buyer_context_issues = list(dict.fromkeys(buyer_context_issues))
    else:
        raise ValueError("Requirement must supply source text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Requirement source text is empty or unavailable")

    blocks, exclusions = [], []
    offset, start = 0, None
    for line in text.splitlines(keepends=True):
        end = offset + len(line)
        metadata = _METADATA.fullmatch(line.strip())
        first_metadata = _first_line_metadata(line, title, bool(text[end:].strip())) if offset == 0 else None
        is_title = first_metadata is not None and first_metadata[1] == "requirement_title"
        if metadata or is_title:
            if start is not None:
                blocks.append((start, offset))
                start = None
            exclusions.append({"source_start": offset, "source_end": end,
                               "reason": "requirement_metadata" if metadata else "requirement_title"})
            if metadata and metadata[1].casefold() == "requirement type":
                declared = _type(metadata[2])
        elif first_metadata:
            prefix_end, reason = first_metadata
            exclusions.append({"source_start": offset, "source_end": offset + prefix_end, "reason": reason})
            start = offset + prefix_end
        elif start is None:
            start = offset
        offset = end
    if start is not None:
        blocks.append((start, len(text)))

    obligations = []
    units = [unit for block_start, block_end in blocks for unit in
             _list_units(text, block_start, block_end, exclusions, declared, declared)]
    for block_start, block_end, parent_start, parent_end, inherited, choice in units:
        block = text[block_start:block_end]
        boundaries = [block_start]
        for match in ([] if choice else re.finditer(r"[.!?](?=\s+(?:[A-Z]|\d))|\n\s*\n", block)):
            # Keep dotted abbreviations/examples attached to their following text.
            preceding = block[max(0, match.start() - 8):match.end()]
            if _ABBREVIATION.search(preceding) or _inside_literal(block, match.start()):
                continue
            boundaries.append(block_start + match.end())
        boundaries.append(block_end)
        for sentence_start, sentence_end in zip(boundaries, boundaries[1:]):
            sentence_start, sentence_end = _trim(text, sentence_start, sentence_end)
            sentence = text[sentence_start:sentence_end]
            if not sentence:
                continue
            issues = []
            delimiter_text = _LIST_MARKER.sub("", sentence) if choice else sentence
            if delimiter_text.count('"') % 2 or delimiter_text.count("(") != delimiter_text.count(")"):
                issues.append("unbalanced_source_delimiters")
            cuts = [sentence_start + cut for cut in _clause_cuts(sentence)] if not issues and not choice else [sentence_start, sentence_end]
            # Strength flows forward across fragments. A local explicit preference
            # overrides the sentence's mandatory subject, and vice versa.
            sentence_type = _strength(sentence[:cuts[1] - sentence_start], inherited, declared) or inherited
            example = _EXAMPLE.search(sentence)
            for begin, end in zip(cuts, cuts[1:]):
                begin, end = _trim(text, begin, end)
                quote = text[begin:end]
                if not quote:
                    continue
                sentence_type = _strength(quote, sentence_type, declared) or sentence_type
                local_issues = list(issues)
                # "Additional formats" needs a comparison baseline; a clear
                # prohibition on extra charges does not have that ambiguity.
                if re.search(r"\badditional\b", quote, re.I) and not re.search(
                    r"\b(?:without|no)\s+(?:an?\s+)?additional\s+(?:[\w-]+\s+){0,2}(?:charges?|fees?|costs?)\b", quote, re.I):
                    local_issues.append("relative_baseline_requires_review")
                features = []
                for match in _FEATURE.finditer(quote):
                    # Example names do not become mandatory named features.
                    if (choice or _LIST_CHOICE.search(quote)
                            or (example and begin + match.start() >= sentence_start + example.start())
                            or re.search(r"\bor\b", quote, re.I)):
                        continue
                    feature = match.group().rstrip(".")
                    if feature not in features:
                        features.append(feature)
                obligations.append({"source_text": quote, "source_start": begin, "source_end": end,
                    "context_start": sentence_start if parent_start is None else parent_start,
                    "context_end": sentence_end if parent_start is None else parent_end,
                    "requirement_type": sentence_type, "named_features": features,
                    "decomposition_issues": local_issues})
    if len(obligations) > max_obligations:
        # Preserve the entire tail as one visible span, but do not claim it is an
        # atomic or completely checked obligation when the bounded parser overflowed.
        tail = obligations[max_obligations - 1:]
        first, last = tail[0], tail[-1]
        obligations = obligations[:max_obligations - 1] + [{
            "source_text": text[first["source_start"]:last["source_end"]],
            "source_start": first["source_start"], "source_end": last["source_end"],
            "context_start": first["context_start"], "context_end": last["context_end"],
            "requirement_type": "unspecified", "named_features": [],
            "decomposition_issues": ["obligation_limit_requires_review"]}]
    if not obligations:
        obligations = [{"source_text": text, "source_start": 0, "source_end": len(text),
                        "context_start": 0, "context_end": len(text), "requirement_type": "unspecified",
                        "named_features": [], "decomposition_issues": ["no_assessable_body"]}]
    for index, obligation in enumerate(obligations, 1):
        obligation["obligation_id"] = f"O{index}"
        obligation["decomposition_issues"] = list(dict.fromkeys([*obligation["decomposition_issues"], *buyer_context_issues]))
    # No substantive source character may disappear. Separator words and
    # punctuation are retained in the left span, and metadata is explicitly named.
    intervals = [(o["source_start"], o["source_end"]) for o in obligations]
    intervals += [(e["source_start"], e["source_end"]) for e in exclusions]
    covered = [False] * len(text)
    for begin, end in intervals:
        covered[begin:end] = [True] * (end - begin)
    source_complete = all(done or char.isspace() for done, char in zip(covered, text))
    return {"version": DECOMPOSITION_VERSION, "text": text,
            "source_sha256": sha256(text.encode()).hexdigest(), "obligations": obligations,
            "buyer_context": buyer_context,
            "buyer_context_issues": buyer_context_issues,
            "buyer_context_sha256": sha256(json.dumps({"buyer_context": buyer_context, "buyer_context_issues": buyer_context_issues},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest(),
            "excluded_spans": exclusions, "source_text_complete": source_complete,
            "requirement_type": summarize_type(o["requirement_type"] for o in obligations)}
