# SPDX-License-Identifier: MPL-2.0
"""Official-ID requirement rows from the application's extracted spreadsheet pages.

This reader recognizes a Number/Description header and explicit IDs such as F-001.
Cell addresses locate fields; they never become requirement identities. Auxiliary
sheets remain available as sources but are not converted into requirement rows.
Unrecognized document/table layouts return None for the caller's existing parser.
No source documents or stored extraction results are changed.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import re

from bidtrace.engine import _admin, _baseline, _category, _chunks, _digest, _norm, _rank, _unique


# The extractor emits cells as "A3: value | B3: value", wrapping text and even
# whole rows across generated pages. A marker must begin a line or follow '|'.
_CELL = re.compile(r'(?:^|\|)[ \t\r\n]*(?P<column>[A-Z]{1,3})(?P<row>[1-9][0-9]{0,6}):[ \t]*', re.M)
_OFFICIAL_ID = re.compile(r'[A-Z][A-Z0-9]{0,15}-[0-9]{1,6}', re.I)
_ID_HEADERS = {'number', 'requirement no', 'requirement number', 'requirement id', 'kravnummer', 'krav nr'}
_DESCRIPTION_HEADERS = {'description of requirement', 'requirement description', 'requirement specification', 'kravbeskrivelse'}


class StructuredRequirementError(ValueError):
    """A recognized requirement table has ambiguous or incomplete rows."""


@dataclass
class _Row:
    official_id: str
    sheet: str
    row: int
    body: str
    group: str
    requirement_type: str
    weighting: str
    text: str
    citations: list[dict]
    description_cell: str

    @property
    def key(self):
        return 'label:' + self.official_id.casefold()


def _clean(value):
    return ' '.join(value.split())


def _header(value):
    return _clean(value).casefold().rstrip('.:')


def _sheet_pages(document):
    sheets = OrderedDict()
    for page in document.get('pages', []):
        if not isinstance(page, dict) or not isinstance(page.get('text'), str):
            continue
        # Requiring extractor-owned sheet metadata avoids interpreting PDF/TXT
        # prose that happens to mention a cell address as a spreadsheet table.
        sheet = page.get('sheet')
        if not isinstance(sheet, str) or not sheet.strip():
            continue
        sheets.setdefault(sheet.strip(), []).append(page)
    return sheets


def _citations(document, role, sheet, intervals, text, start, end):
    result = []
    for page, page_start, page_end in intervals:
        left, right = max(start, page_start), min(end, page_end)
        if left >= right:
            continue
        quote = text[left:right].strip()
        if not quote:
            continue
        # Quotes are slices of the original page text, including its original
        # cell markers and line wrapping; never quote recomposed field values.
        assert quote in page['text']
        result.append({'document_id': document['id'], 'title': document.get('title', 'Spreadsheet'),
                       'page': page['page'], 'quote': quote, 'role': role,
                       'source_url': document.get('source_url'), 'sheet': sheet})
    return result


def _read(document, role):
    recognized = False
    result = []
    identities = set()
    for sheet, pages in _sheet_pages(document).items():
        intervals = []
        parts = []
        offset = 0
        for page in pages:
            value = page['text']
            intervals.append((page, offset, offset + len(value)))
            parts.append(value)
            offset += len(value) + 1
        text = '\n'.join(parts)
        tokens = list(_CELL.finditer(text))
        cells = OrderedDict()
        for index, token in enumerate(tokens):
            row = int(token['row'])
            end = tokens[index + 1].start() if index + 1 < len(tokens) else len(text)
            # Exclude the separator preceding the next physical row/cell.
            cells.setdefault(row, []).append({'column': token['column'], 'start': token.start(), 'end': end,
                                              'value': text[token.end():end].strip()})
        header = None
        for row, items in cells.items():
            ids = [c['column'] for c in items if _header(c['value']) in _ID_HEADERS]
            descriptions = [c['column'] for c in items if _header(c['value']) in _DESCRIPTION_HEADERS]
            if len(ids) == 1 and len(descriptions) == 1:
                optional = {}
                for cell in items:
                    name = _header(cell['value'])
                    if re.fullmatch(r'group(?: v[0-9.]+)?', name):
                        optional['group'] = cell['column']
                    elif name.startswith('requirement type'):
                        optional['requirement_type'] = cell['column']
                    elif name in {'weighting', 'weight'}:
                        optional['weighting'] = cell['column']
                header = {'row': row, 'id': ids[0], 'description': descriptions[0], **optional}
                break
        if header is None:
            continue
        recognized = True
        for row, items in cells.items():
            if row <= header['row']:
                continue
            by_column = {c['column']: c for c in items}
            id_cell = by_column.get(header['id'])
            if id_cell is None or not id_cell['value'].strip():
                continue
            official_id = _clean(id_cell['value']).upper()
            if not _OFFICIAL_ID.fullmatch(official_id):
                # A populated ID+description row with another identifier format
                # is outside this parser's scope; do not omit it selectively.
                if by_column.get(header['description'], {}).get('value', '').strip():
                    return None
                continue
            if len(items) != len(by_column):
                raise StructuredRequirementError(f'Ambiguous repeated cell address in {sheet}, row {row}.')
            if official_id in identities:
                raise StructuredRequirementError(f'Official requirement ID {official_id} occurs more than once; resolve its source identity before comparing versions.')
            body_cell = by_column.get(header['description'])
            body = _clean(body_cell['value']) if body_cell else ''
            if not body:
                raise StructuredRequirementError(f'Official requirement {official_id} has no readable description in {sheet}, row {row}.')
            identities.add(official_id)
            metadata = {key: _clean(by_column.get(header.get(key), {}).get('value', ''))
                        for key in ('group', 'requirement_type', 'weighting')}
            display = [official_id + (' | ' + metadata['group'] if metadata['group'] else ''), body]
            if metadata['requirement_type']:
                display.append('Requirement type: ' + metadata['requirement_type'])
            if metadata['weighting']:
                display.append('Weighting: ' + metadata['weighting'])
            citations = _citations(document, role, sheet, intervals, text,
                                   min(c['start'] for c in items), max(c['end'] for c in items))
            result.append(_Row(official_id, sheet, row, body, metadata['group'], metadata['requirement_type'],
                               metadata['weighting'], '\n'.join(display), citations,
                               f'{header["description"]}{row}'))
    return result if recognized else None


def analyze_structured_requirements(previous, current, supplier) -> list[dict] | None:
    """Analyze recognized spreadsheet rows, or return None for legacy fallback.

    IDs (not row numbers) align revisions. Changed wording, grouping, weighting,
    or requirement type are changes; moving a row/page is not. Existing narrow
    atomic evidence checks operate on the complete description only. The whole
    row remains cited with one exact quote for each physical source page.
    """
    after_rows = _read(current, 'current')
    if after_rows is None:
        return None
    before_rows = _read(previous, 'previous') if previous is not None else []
    if before_rows is None:
        return None
    prior = {r.key: r for r in before_rows}
    current_keys = {r.key for r in after_rows}
    pairs = [(prior.get(r.key), r) for r in after_rows]
    pairs.extend((r, None) for r in before_rows if r.key not in current_keys)
    chunks = _chunks(supplier or [])
    requirements = []
    for before, after in pairs:
        row = after or before
        if before is None:
            change = 'unchanged' if previous is None else 'added'
        elif after is None:
            change = 'removed'
        else:
            change = 'unchanged' if _norm(before.text) == _norm(after.text) else 'modified'
        citations = [*(before.citations if before else []), *(after.citations if after else [])]
        category, priority = _category(row.body)
        if row.requirement_type.casefold() == 'mandatory' or row.weighting.casefold() in {'important', 'high'}:
            priority = 'high'
        if change == 'removed':
            assessment, rationale = 'unknown', 'This official requirement row was removed from the current specification. Confirm the amendment; no current supplier compliance conclusion is made for a removed requirement.'
        elif _admin(row.body):
            assessment, rationale = 'unknown', 'Administrative date or submission condition: confirm the instruction and completion with the bid owner. Supplier capability evidence does not establish completion.'
        else:
            ranked = _rank(row.body, chunks, None)
            assessment, rationale, supporting = _baseline(row.body, chunks, ranked)
            citations.extend(supporting)
        if previous is None:
            rationale = 'No previous version was supplied; this is an initial assessment, not a verified version comparison. ' + rationale
        requirements.append({
            'id': 'req_' + _digest(row.key), 'key': row.key,
            'title': row.official_id + (' | ' + row.group if row.group else ''),
            'category': category, 'priority': priority, 'assessment': assessment,
            'status': 'needs_review' if change != 'unchanged' else assessment,
            'change_type': change, 'before': before.text if before else None, 'after': after.text if after else None,
            'rationale': rationale, 'citations': _unique(citations), 'reviewed': False,
            'review_decision': None, 'review_note': None, 'official_id': row.official_id,
            'extraction': {'method': 'official_spreadsheet_requirement_rows', 'sheet': row.sheet,
                           'row': row.row, 'description_cell': row.description_cell,
                           'requirement_type': row.requirement_type, 'weighting': row.weighting,
                           'scope': 'Explicit official-ID requirement rows; glossary, instructions and support sheets remain source context.'},
        })
    return requirements
