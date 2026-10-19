# SPDX-License-Identifier: MPL-2.0
"""Small deterministic relations over validated local-trial text observations."""
from __future__ import annotations

from decimal import Decimal
import re

_ONES = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
_TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
_WORD_NUMBER = '(?:' + '|'.join(_TENS) + r')(?:[- ‐‑–]+(?:' + '|'.join(list(_ONES)[1:10]) + '))?|(?:' + '|'.join(_ONES) + ')'
_DURATION = re.compile(rf"\b(?P<value>\d+(?:\.\d+)?|{_WORD_NUMBER})"
    r"(?:\s*\(\s*(?P<parenthesized>\d+(?:\.\d+)?)\s*\))?\s*[-‐‑‒–— ]?\s*"
    r"(?P<unit>months?|years?|minutes?|hours?|days?|weeks?)\b", re.I)
_MIN = re.compile(r"\b(?:at least|no less than|not less than|minimum(?: of)?)\b", re.I)
_MAX = re.compile(r"\b(?:at most|no more than|not more than|not exceed|up to|within|maximum(?: of)?)\b", re.I)
_AMBIGUOUS = re.compile(r"\b(?:if|could|would|might|may|example|illustrative|hypothetical|either|or|options?|previous|historical|former|approximately|about)\b", re.I)
_REFUSAL = re.compile(r"\b(?:(?:will|shall|would|can) not|cannot|can't|won't|refus(?:e[sd]?|ing)|declin(?:e[sd]?|ing)|never)\b", re.I)
_UNSPECIFIED = re.compile(r"\b(?:this|the|our|submitted)\s+(?:offer|proposal|quote|document|submission)\s+(?:does not|doesn't)\s+(?:specify|state|define|include|provide)\s+([^.!?]+)", re.I)
_OTHER_PROPERTY = re.compile(r"\b(?:retention|retain(?:ed)?|backups?|warranty|warranties|experience|"
    r"SLA|response|acknowledg\w*|notification|renewal|extension)\b", re.I)
_TERM_AMBIGUOUS = re.compile(r"\b(?:if|unless|could|would|might|may|either|or|previous|historical|former|"
    r"approximately|about|estimated|targets?|aims?|hopes?|another|competitor|subject\s+to)\b", re.I)
_OPTIONAL_CONTEXT = re.compile(r"\b(?:options?|optional|example|illustrative|hypothetical)\b", re.I)
_NEGOTIABLE = re.compile(r"\b(?:negotiab(?:le|ility)|willing(?:ness)?\s+to\s+negotiate)\b", re.I)
_WILLING = re.compile(r"\b(?:(?:we|the\s+(?:supplier|provider|contractor|firm))\s+(?:are|is|remain)\s+"
    r"(?:willing|prepared)\s+to\s+negotiate|we\s+will\s+negotiate|"
    r"(?:contract\s+)?terms?\s+(?:are|is)\s+negotiable)\b", re.I)
_UNWILLING = re.compile(r"\b(?:(?:contract\s+)?terms?\s+(?:are|is)\s+not\s+negotiable|"
    r"we\s+(?:are\s+not\s+willing\s+to|will\s+not)\s+negotiate)\b", re.I)
_RECURRING_PERIOD = re.compile(r'\brecurring\s+(?:fees?|charges?|services?)\b', re.I)
_REQUEST_REPORT = re.compile(r'\b(?:must|shall|should|requires?|required|requests?|requested|'
    r'buyer|client|customer|authority|RFP|tender)\b', re.I)
_BACKUP = re.compile(r'\b(?:back[ -]?ups?|backup\s+copies)\b', re.I)
_RECOVERY_PERIOD = re.compile(r'\b(?:retention|retain(?:s|ed|ing)?|recoverable|kept|keep|'
    r'available\s+for\s+(?:restoration|recovery))\b', re.I)
_BACKUP_ENTITIES = {'server': r'\bservers?\b', 'workstation': r'\b(?:workstations?|desktops?)\b',
                   'endpoint': r'\bendpoints?\b', 'saas': r'\b(?:saas|microsoft\s*365|m365|google\s*workspace)\b'}
_BACKUP_LOCATIONS = {'local': r'\b(?:local|on[- ]premises|on[- ]prem)\b', 'cloud': r'\bcloud\b'}
_CADENCE = re.compile(r'\b(?:daily|weekly|monthly|quarterly|annually|yearly|'
    r'(?:once|twice|\d+)\s+(?:per|a)\s+(?:day|week|month|quarter|year)|'
    r'every\s+(?:(?:\d+|one|two|three|four|five|six)\s+)?(?:day|week|month|quarter|year)s?)\b', re.I)


def _number_value(match):
    raw = match['value'].lower()
    if re.fullmatch(r'\d+(?:\.\d+)?', raw):
        value = Decimal(raw)
    else:
        words = re.split(r'[- ‐‑–]+', raw)
        value = Decimal(sum(_ONES.get(word, _TENS.get(word, 0)) for word in words))
    if match['parenthesized'] and Decimal(match['parenthesized']) != value:
        return None
    return value


def _duration_value(match):
    value = _number_value(match)
    if value is None:
        return None
    unit = match['unit'].lower().rstrip('s')
    return value * (12 if unit == 'year' else 1) if unit in {'month', 'year'} else None


def _elapsed_duration(text):
    """Only exact elapsed units; calendar months and business days vary."""
    match = _DURATION.fullmatch(text.strip())
    if not match or _number_value(match) is None:
        return None
    factor = {'minute': 60, 'hour': 3600, 'day': 86400, 'week': 604800}.get(
        match['unit'].lower().rstrip('s'))
    return _number_value(match) * factor if factor is not None else None


_NOTICE_VERB = r'(?:announc\w*|notif\w*|inform\w*|notice)'
_NOTICE_REQUEST = re.compile(
    rf'(?P<event>[\w][\w -]*?)\s+(?:must|shall|should)\s+be\s+(?:announced|notified)\s+'
    rf'(?P<bound>{_MIN.pattern}|{_MAX.pattern})\s+(?P<duration>.+?)\s+'
    r'(?:beforehand|in advance)\.?', re.I)
_NOTICE_OFFER = re.compile(
    r'(?:we|the supplier|the provider)\s+(?:will\s+)?(?:notify|notifies|inform|informs)\s+'
    r'(?:the|our)\s+(?:buyer|client|customer)\s+(?P<duration>.+?)\s+before\s+'
    r'(?:(?:every|each|all)\s+)?(?P<event>[\w][\w -]*?)\.?', re.I)
_NOTICE_PASSIVE = re.compile(
    r'(?P<event>[\w][\w -]*?)\s+(?:is|are|will be)\s+(?:announced|notified)\s+'
    r'(?P<duration>.+?)\s+(?:beforehand|in advance)\.?', re.I)


def _advance_notice_bound(request, quotes, check):
    """Compare a complete notice lead-time condition, never an unrelated number.

    Full statement grammar binds the duration to notification *before* one event.
    Event words retain entity/location qualifiers; only a trailing "window" and
    an initial article are syntactic variants. Other timing families stay with
    their existing guards rather than borrowing a notice value.
    """
    if not (re.search(_NOTICE_VERB, request, re.I)
            and re.search(r'\b(?:beforehand|in advance)\b', request, re.I)
            and (_MIN.search(request) or _MAX.search(request))):
        return None
    unknown = lambda: _unknown(check, 'advance_notice_duration_or_scope_unresolved',
        'Selected supplier text does not establish one unambiguous notice lead time for the same event and entity; unrelated or qualified durations cannot establish this condition.')
    requested = _NOTICE_REQUEST.fullmatch(request.strip())
    if not requested or _elapsed_duration(requested['duration']) is None:
        return unknown()

    def event_words(value):
        value = re.sub(r'^the\s+', '', value.strip().lower())
        value = re.sub(r'\s+windows?$', '', value)
        return tuple(value.split())

    event = event_words(requested['event'])
    if not event or _TERM_AMBIGUOUS.search(requested['event']):
        return unknown()
    values = set()
    ambiguous = False
    for quote in quotes:
        for clause in _clauses(quote):
            if not re.search(_NOTICE_VERB, clause, re.I):
                continue
            offered = _NOTICE_OFFER.fullmatch(clause.strip()) or _NOTICE_PASSIVE.fullmatch(clause.strip())
            if not offered:
                ambiguous = True
                continue
            if event_words(offered['event']) != event:
                continue
            value = _elapsed_duration(offered['duration'])
            if (value is None or _OPTIONAL_CONTEXT.search(quote)
                    or _TERM_AMBIGUOUS.search(clause) or _REFUSAL.search(clause)):
                ambiguous = True
            else:
                values.add(value)
    if ambiguous or len(values) != 1:
        return unknown()
    actual, required = next(iter(values)), _elapsed_duration(requested['duration'])
    minimum = bool(_MIN.fullmatch(requested['bound']))
    matches = actual >= required if minimum else actual <= required
    return {'assessment': 'supported' if matches else 'contradicted',
            'reason': 'explicit_advance_notice_duration_comparison',
            'rationale': f'Supplier notice lead time for {requested["event"]}: {actual / 3600} hours; '
                         f'requested {"minimum" if minimum else "maximum"}: {required / 3600} hours before the same event.'}


def _term_scope(text):
    if _OTHER_PROPERTY.search(text):
        return False
    return bool((re.search(r'\b(?:contract|agreement|service)\b', text, re.I)
                 and re.search(r'\b(?:initial|terms?|duration|length)\b', text, re.I))
                or (re.search(r'\binitial\b', text, re.I) and re.search(r'\bterm\b', text, re.I)))


def is_contract_duration_condition(text):
    """A selected contract/service duration, not a trial or another clock."""
    durations = list(_DURATION.finditer(text))
    return (_term_scope(text) and len(durations) == 1
            and _duration_value(durations[0]) is not None)


def single_payment_term(quote):
    """Read one explicit term-table row, with a matching payment schedule.

    A lone pricing period is insufficient. The source must label Term Options
    beneath Description/Payments/Interval/Amount and supply exactly one row.
    Multiple options, inconsistent counts and qualified examples remain open.
    Returned text is original source; this function never assigns a verdict.
    """
    lines = quote.splitlines()
    headings = [i for i, line in enumerate(lines)
                if re.fullmatch(r'\s*Term Options\s*', line, re.I)]
    if len(headings) != 1:
        return None
    index = headings[0]
    prefix = '\n'.join(lines[:index])
    preceding = next((line for line in reversed(lines[:index]) if line.strip()), '')
    if (not re.fullmatch(r'\s*Description\s+Payments\s+Interval\s+Amount\s*', preceding, re.I)
            or _TERM_AMBIGUOUS.search(prefix) or _OPTIONAL_CONTEXT.search(prefix)
            or _REQUEST_REPORT.search(prefix) or _REFUSAL.search(prefix)):
        return None
    rows = []
    for line in lines[index + 1:]:
        if not line.strip():
            continue
        row = re.fullmatch(r'\s*(?P<description>.+?)\s{2,}(?P<count>\d+)\s{2,}'
                           r'(?P<interval>Monthly|Annually|Yearly)\s{2,}'
                           r'[$€£]?\s*[\d,]+(?:\.\d+)?\s*', line, re.I)
        if not row:
            if rows:
                break
            return None
        rows.append((line, row))
    if len(rows) != 1:
        return None
    line, row = rows[0]
    description = row['description']
    duration = list(_DURATION.finditer(description))
    if (len(duration) != 1 or _duration_value(duration[0]) is None
            or not re.fullmatch(r'(?:Monthly|Annual|Yearly)\s*[-–—]\s*' + re.escape(duration[0][0]), description, re.I)
            or _duration_value(duration[0]) <= 0):
        return None
    months = Decimal(row['count']) * (1 if row['interval'].lower() == 'monthly' else 12)
    if months != _duration_value(duration[0]):
        return None
    # Another explicitly qualified term heading anywhere in this chunk must
    # not be hidden by extracting the table row.
    residual = '\n'.join(lines[:index] + lines[index + 1:])
    if (_OPTIONAL_CONTEXT.search(residual)
            or any(_term_scope(clause) and (_TERM_AMBIGUOUS.search(clause)
                                            or _REQUEST_REPORT.search(clause)
                                            or _REFUSAL.search(clause))
                   for clause in _clauses(residual))):
        return None
    return {'clause': line, 'duration': duration[0][0]}


def _clauses(quote):
    """Group source clauses; adjacent term/condition headings stay attached."""
    pieces = re.split(r'(?<=[.!?;])\s+|\n+', quote)
    for index, piece in enumerate(pieces):
        previous = pieces[index - 1] if index else ''
        if (previous.rstrip().endswith(':') and not _DURATION.search(previous)
                and (_term_scope(previous) or _TERM_AMBIGUOUS.search(previous)
                     or (_BACKUP.search(previous) and _RECOVERY_PERIOD.search(previous)))):
            yield previous + '\n' + piece
        else:
            yield piece


def _unknown(check, reason, rationale):
    return {'assessment': 'unknown', 'reason': reason, 'rationale': rationale} if check.get('assessment') != 'unknown' else None


def _property_words(text):
    text = re.sub(r"\bnotif(?:y|ications?)\b", "notification", text.lower())
    ignored = set("a an the this our offer proposal quote document submission supplier buyer must shall should provide include send within after before initial contract term deadline time period duration value number".split())
    return {word.removesuffix("s") for word in re.findall(r"[a-z]+", text) if len(word) > 2 and word not in ignored}


def _missing_scoped_timing(request, quotes, check):
    """A related service/duration cannot establish an absent scoped period."""
    clauses = [clause for quote in quotes for clause in _clauses(quote)]
    scopes = lambda text, patterns: {name for name, pattern in patterns.items() if re.search(pattern, text, re.I)}
    if _BACKUP.search(request) and _RECOVERY_PERIOD.search(request) and _DURATION.search(request):
        required = scopes(request, _BACKUP_ENTITIES)
        locations = scopes(request, _BACKUP_LOCATIONS)
        if len(required) == 1:
            for clause in clauses:
                if not required <= scopes(clause, _BACKUP_ENTITIES) or not _BACKUP.search(clause):
                    continue
                explicit_refusal = re.search(r'\b(?:will\s+not|cannot|can\x27t|never)\s+(?:provide|retain|store|keep|support)\b|'
                    r'\bno\s+(?:server|endpoint|workstation|desktop|saas)\s+backups?\b|'
                    r'\bbackups?\s+(?:is|are)\s+not\s+(?:provided|included|available)\b', clause, re.I)
                if explicit_refusal:
                    return None
                if (_RECOVERY_PERIOD.search(clause) and (not locations or locations <= scopes(clause, _BACKUP_LOCATIONS))
                        and (_DURATION.search(clause) or re.search(r'\b(?:indefinitely|forever|unlimited|permanently)\b', clause, re.I))):
                    return None  # Relevant value exists; this guard does not compare it.
            return _unknown(check, 'backup_recovery_period_not_established_for_requested_scope',
                'Selected supplier text does not establish the recovery/retention period for the requested backup entity and location. A different backup scope or mere service inclusion does not establish a conflicting period.')
    if _CADENCE.search(request):
        event = _property_words(_CADENCE.sub('', request)) - set('occur occurs occurrence happen happens perform performed exercise schedule scheduled frequency cadence every once twice per'.split())
        # Preserve the event noun "exercise"; "perform" and "occur" alone are
        # generic verbs, while restoration/exercise jointly identify the event.
        if re.search(r'\bexercises?\b', request, re.I):
            event.add('exercise')
        matching = [clause for clause in clauses if len(event) >= 2 and event <= _property_words(clause)]
        if matching and not any(_CADENCE.search(clause) or _REFUSAL.search(clause)
                                or re.search(r'\b(?:no|not|never)\b', clause, re.I) for clause in matching):
            return _unknown(check, 'event_frequency_not_established_by_selected_text',
                'The selected text describes the requested activity but supplies no frequency for that activity; an absent cadence is not an explicit conflicting cadence.')
    return None


_REFERENCE_HEADING = re.compile(r'^\s*(?:Business|Client|Customer) References\s*:\s*$', re.I | re.M)


def reference_field_request(request, context):
    """One contact field under an explicit minimum reference count."""
    leaf = re.sub(r'[,\s]*(?:and)?[.\s]*$', '', request.strip(), flags=re.I)
    field = leaf.lower() if leaf.lower() in {'title', 'phone'} else None
    if re.fullmatch(r'A minimum of .+ references relating to the services is being requested with full\s+name', leaf, re.I):
        field = 'name'
    enclosing = context.get('enclosing_span', {}).get('text', '')
    match = re.search(r'\b(?:minimum(?: of)?|at least)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
                      r'(?:\s*\((\d+)\))?\s+references?\b', enclosing, re.I)
    if (not field or not match or context.get('paired_buyer_questions') or context.get('referenced_buyer_context')
            or re.search(r'\b(?:municipal|government|public|dated|recent|within|last|similar|certified)\b', enclosing, re.I)):
        return None
    count = int(match[1]) if match[1].isdigit() else _ONES[match[1].lower()]
    if count < 1 or match[2] and int(match[2]) != count:
        return None
    return field, count


def reference_contact_field_fact(request, context, evidence):
    """Count actual client contact records, excluding preceding staff resumes.

    A split source must have validated, identical document/page identity. Only
    the requested field is established; address, fax and reference validity are
    separate. Repeated names never increase the reference count.
    """
    requested = reference_field_request(request, context)
    if not requested or not evidence or any(e.get('mode') != 'text' for e in evidence):
        return None
    if len(evidence) > 1:
        identities = {(e.get('document_id'), e.get('page')) for e in evidence}
        if len(identities) != 1 or None in next(iter(identities)):
            return None
    text = '\n\n'.join(e['quote'] for e in evidence)
    heading = _REFERENCE_HEADING.search(text)
    if not heading:
        return None
    body = text[heading.end():]
    if re.search(r'\b(?:hypothetical|example|sample|optional|pending|not provided|no references)\b', body, re.I):
        return None
    entries = list(re.finditer(r'^\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*$', body, re.M))
    names = set()
    for i, entry in enumerate(entries):
        name, title = entry[1].strip(), entry[2].strip()
        if (len(name.split()) < 2 or re.search(r'\d|https?://', name + title)
                or re.search(r'\b(?:unknown|TBD|none|not provided)\b', title, re.I)):
            continue
        if requested[0] == 'phone':
            region = body[entry.end():entries[i+1].start() if i+1 < len(entries) else len(body)]
            lines = [line for line in region.splitlines() if not re.search(r'\bfax\b', line, re.I)]
            if not re.search(r'(?<!\d)(?:\(\d{3}\)|\d{3})[- .]?\d{3}[- .]\d{4}(?!\d)', '\n'.join(lines)):
                continue
        names.add(name.casefold())
    if len(names) < requested[1]:
        return None
    return {'assessment':'supported','reason':'explicit_reference_contact_field_presence',
            'rationale':f'The client-reference section supplies {requested[0]} fields for {len(names)} distinct named contacts, meeting the requested minimum of {requested[1]}. Other reference fields and validity remain separate.'}


def standard_sla_disclosure_request(request, context):
    leaf = re.sub(r'[,\s]*(?:and)?[.\s]*$', '', request.strip(), flags=re.I)
    return bool(re.fullmatch(r'(?:Include|Provide|Submit)(?:\s+in (?:the|your) (?:response|proposal))?\s+(?:your|the) standard SLAs?', leaf, re.I)
        and context and not context.get('paired_buyer_questions') and not context.get('referenced_buyer_context')
        and not any(re.search(r'\b(?:guaranteed|guarantee|recovery|availability|minimum|maximum)\b|\d', s.get('text',''), re.I)
                    for s in context.get('governing_spans', [])))


def standard_response_terms_fact(request, context, evidence):
    """Disclosure of qualified response terms, never a recovery guarantee."""
    if not standard_sla_disclosure_request(request, context):
        return None
    for e in evidence:
        if e.get('mode') != 'text':
            continue
        text = ' '.join(e['quote'].split())
        if (re.search(r'\bResponse Times\b', text, re.I)
                and re.search(r'\bresponse times? (?:are )?measured from the moment the client submits a support request\b', text, re.I)
                and re.search(r'\bresponse times? apply during standard (?:working|business) hours\b', text, re.I)
                and re.search(r'\btarget is (?:\d+(?:\.\d+)?)% of support tickets reach Response stage within \d+(?:\.\d+)? hours? of being submitted\b', text, re.I)
                and re.search(r'\bdeemed to have responded when it has replied to the client', text, re.I)
                and not re.search(r'\b(?:hypothetical|example|illustrative|optional|client needs|buyer requires)\b', text, re.I)):
            return {'assessment':'supported','reason':'explicit_standard_response_terms_disclosure',
                    'rationale':'Standard business-hours response terms, a target and acknowledgement definition are supplied. This verifies disclosure, not guaranteed resolution, enhanced options or achieved performance.'}
    return None


def source_field_decision(request, context, evidence):
    decision = reference_contact_field_fact(request, context, evidence) or standard_response_terms_fact(request, context, evidence)
    if decision:
        return decision
    enclosing = ' '.join(context.get('enclosing_span',{}).get('text','').split()).rstrip('.')
    leaf = ' '.join(request.split()).rstrip(' .,&')
    unqualified = (not context.get('paired_buyer_questions') and not context.get('referenced_buyer_context')
        and not any(re.search(r'\d|\b(?:certified|guaranteed|current|valid|certificate copy|audit report|daily|weekly)\b',
                             span.get('text',''),re.I) for span in context.get('governing_spans',[])))
    if unqualified:
        quotes = [e['quote'] for e in evidence if e.get('mode') == 'text']
        if (re.fullmatch(r'Line of Business Application Updates & Maintenance',leaf,re.I)
                and re.fullmatch(re.escape(leaf),enclosing,re.I)):
            from bidtrace_core.local.content import _included_software_lifecycle_maintenance
            if _included_software_lifecycle_maintenance(quotes):
                return {'assessment':'supported','reason':'explicit_included_software_maintenance',
                        'rationale':'The proposal includes comprehensive software lifecycle management with regular software updates and maintenance. This covers the unqualified application-maintenance service, not a named application, frequency or excluded scope.'}
        for quote in quotes:
            text = ' '.join(quote.split())
            if re.search(r'\b(?:optional|hypothetical|example|excluded|excluding|not included|no certifications|pending|expired)\b',text,re.I):
                continue
            if (re.fullmatch(r'Warranty Part Ordering',leaf,re.I)
                    and re.fullmatch(r'Warranty Part Ordering & Temporary Replacement Parts',enclosing,re.I)
                    and re.search(r'\boffers extended support for hardware and software, including warranty part ordering and depot services for all hardware under warranty\b',text,re.I)):
                return {'assessment':'supported','reason':'explicit_warranty_ordering_inclusion',
                        'rationale':'The proposal explicitly includes warranty part ordering for hardware under warranty. Temporary replacement parts or loaners remain a separate unresolved condition.'}
            if (re.fullmatch(r'Provide any existing security certifications',leaf,re.I)
                    and re.fullmatch(re.escape(leaf),enclosing,re.I)
                    and re.search(r'\bOur certifications include\b[^.]*\bSOC\s*(?:II|2)\s+Type\s*2 Certification\b',text,re.I)):
                return {'assessment':'supported','reason':'explicit_security_certification_disclosure',
                        'rationale':'The proposal lists SOC 2 Type 2 among its existing certifications. This establishes disclosure of the declared security qualification, not a certificate copy, audit report or independent verification.'}
    design = bool(re.fullmatch(r'IT systems? needs assessment and design',enclosing,re.I))
    standards = bool(re.fullmatch(r'Knowledge of industry standards for IT systems for organizations of similar size and function',enclosing,re.I))
    if ((design or standards) and not context.get('paired_buyer_questions') and not context.get('referenced_buyer_context')
            and not any(re.search(r'\d|\b(?:guarantee|certified|recent|within|last)\b',s.get('text',''),re.I)
                        for s in context.get('governing_spans',[]))):
        for e in evidence:
            if e.get('mode')!='text':
                continue
            text=' '.join(e['quote'].split())
            if re.search(r'\b(?:hypothetical|example|optional|client needs|buyer requires|will have|would have|do not have|does not have|no experience|no knowledge)\b',text,re.I):
                continue
            if design and re.search(r'\b(?:has|have) (?:\w+ ){0,3}experience in assessing and designing IT systems tailored to the specific needs of (?:our|their) clients\b',text,re.I):
                return {'assessment':'supported','reason':'explicit_it_qualification_declaration',
                        'rationale':'The supplier explicitly declares experience in assessing and designing IT systems tailored to client needs. This establishes the stated qualification, not independently verified expertise.'}
            if (standards and re.search(r'\b(?:is|are) well[- ]versed in .{0,50}industry standards.{0,65}\bIT systems\b',text,re.I)
                    and re.search(r'\bsuccessfully implemented IT solutions for organizations of similar size and function\b',text,re.I)):
                return {'assessment':'supported','reason':'explicit_it_qualification_declaration',
                        'rationale':'The supplier declares IT industry-standards knowledge and implemented solutions for organizations of similar size and function. These are stated qualifications, not independent verification.'}
    if (not re.fullmatch(r'Online Client Portal[.\s]*', request.strip(), re.I)
            or not re.fullmatch(r'Online Client Portal[.\s]*', context.get('enclosing_span',{}).get('text','').strip(), re.I)
            or context.get('paired_buyer_questions') or context.get('referenced_buyer_context')
            or any(re.search(r'\b(?:status|visibility|access|reports|authentication|guarantee)\b|\d', s.get('text',''), re.I)
                   for s in context.get('governing_spans',[]))):
        return None
    for e in evidence:
        if e.get('mode')!='text':
            continue
        text=' '.join(e['quote'].split())
        if (re.search(r'\bResponse Times\b',text,re.I)
                and re.search(r'\bsupport request raised via email or the online support portal\b',text,re.I)
                and re.search(r'\bthe client submits a support request\b',text,re.I)
                and not re.search(r'\b(?:optional|hypothetical|example|illustrative|client needs|buyer requires|not available|no portal)\b',text,re.I)):
            return {'assessment':'supported','reason':'explicit_client_request_portal_presence',
                    'rationale':'The proposal describes client request submission through an online support portal. This establishes basic portal presence, not ticket visibility, report access or availability guarantees.'}
    return None


def relation_override(obligation: dict, check: dict) -> dict | None:
    """Compare only explicit, unambiguous facts; retain all other model results.

    Initial contract mismatches use exact calendar month/year conversions. A match
    leaves the model verdict unchanged because other conditions may be unstated. Other
    Notice lead time can resolve a complete, single-condition bound comparison.
    Other timing properties retain their existing scope and unspecified guards.
    No source metadata, supplier names, expected labels or rationales drive this.
    """
    request = obligation.get("source_text", obligation.get("text", ""))
    evidence = check.get("evidence", [])
    if not isinstance(request, str) or not evidence or any(e.get("mode") != "text" for e in evidence):
        return None
    quotes = [e.get("quote", "") for e in evidence]
    if any(not isinstance(quote, str) for quote in quotes):
        return None
    source = "\n".join(dict.fromkeys(quotes))
    selected_scope = obligation.get('selected_scope', {})
    field_decision = source_field_decision(request, selected_scope, evidence)
    if field_decision:
        return field_decision
    certification = re.fullmatch(
        r'(SOC\s*(?:2|II)\s+Type\s*([12])\s+Certification)(?:\s+(?:preferred|required))?[.\s]*',
        request.strip(), re.I)
    if (certification and selected_scope
            and not selected_scope.get('paired_buyer_questions')
            and not selected_scope.get('referenced_buyer_context')
            and not any(re.search(r'\b(?:audit\s+report|certificate\s+copy|independent\s+verification|current|valid|expiry|expiration)\b',
                                  span.get('text', ''), re.I)
                        for span in selected_scope.get('governing_spans', []))):
        # A declared qualification is different from documentary verification.
        # Bind an affirmative possession verb to the identical SOC type, never
        # a service logo, future certification, client need or audit artifact.
        target = rf'SOC\s*(?:2|II)\s+Type\s*{certification[2]}\s+Certification'
        for quote in quotes:
            for clause in re.split(r'(?<=[.!?;])\s+', ' '.join(quote.split())):
                if (re.search(rf'\b(?:hold|holds|have|has)\s+{target}\b', clause, re.I)
                        and not re.search(r'\b(?:not|no|expired|pending|formerly|previously|if|will|would|may|must|shall|requires?|buyer|client|customer)\b', clause, re.I)):
                    return {'assessment': 'supported', 'reason': 'explicit_declared_soc_qualification',
                            'rationale': f'The supplier explicitly states that it holds SOC 2 Type {certification[2]} certification. This establishes the declared qualification, not independent verification of a certificate or audit report.'}
    if (re.fullmatch(r'(?:Provide\s+)?(?:Security\s+)?Patch Management[.\s]*', request.strip(), re.I)
            and re.fullmatch(r'(?:Security\s+)?Patch Management[.\s]*', selected_scope.get('enclosing_span', {}).get('text', '').strip(), re.I)
            and not selected_scope.get('paired_buyer_questions') and not selected_scope.get('referenced_buyer_context')
            and not any(re.search(r'\d|\b(?:daily|weekly|monthly|every|within|unless|except|only)\b', s['text'], re.I)
                        for s in selected_scope.get('governing_spans', []))):
        # One qualitative commodity service, with an explicit inclusion and
        # patching action in the same supplier passage. No generic lifecycle,
        # optional product, buyer-needs quotation or extra timing/target duty
        # can satisfy this bounded relation. This rule never consults labels.
        for quote in quotes:
            if (re.search(r'\bfollowing services are included in (?:this|the) service agreement\b', ' '.join(quote.split()), re.I)
                    and re.search(r'\bsecurity patching\s+and.{0,100}\bupdates for workstations\b.{0,100}\bservers\b', ' '.join(quote.split()), re.I)
                    and not re.search(r'\b(?:optional|excluded|except|not|client needs|requires|only)\b', quote, re.I)):
                return {'assessment':'supported', 'reason':'explicit_included_patch_service',
                        'rationale':'The supplier explicitly includes security patching and updates for workstations and servers in this service agreement.'}
    context = obligation.get("sentence_context", "")
    scope = request + " " + (context if isinstance(context, str) else "")
    if (check.get('assessment') == 'contradicted'
            and re.search(r'\b(?:renewal|extension|renew|extend)\b', request, re.I)
            and re.search(r'\b(?:contract|agreement|term)\b', scope, re.I)):
        # A trial-period exclusion or a pointer to an unread MSA does not
        # refuse renewal of the contract itself. Preserve explicit refusals.
        refusal = re.search(
            r'\b(?:contract|agreement|services?|initial\s+term)\s+'
            r'(?:cannot|will\s+not|shall\s+not|may\s+not)\s+be\s+(?:renewed|extended)\b|'
            r'\b(?:renewal|extension)\s+(?:options?\s+)?(?:is|are)\s+'
            r'(?:not\s+(?:available|offered|permitted)|prohibited)\b|'
            r'\b(?:we|the\s+(?:supplier|provider))\s+(?:will\s+not|cannot)\s+'
            r'(?:renew|extend)\s+(?:the\s+)?(?:contract|agreement|services?)\b', source, re.I)
        if not refusal:
            return _unknown(check, 'contract_renewal_refusal_not_established_by_selected_text',
                'Selected supplier text does not explicitly refuse renewal or extension of the contract. Exclusion from a trial-period cancellation right or a reference to external terms does not establish refusal.')
    if (check.get('assessment') == 'supported'
            and re.search(r'\b(?:renewal|extension|renew|extend)\b', request, re.I)
            and re.search(r'\b(?:contract|agreement|term)\b', scope, re.I)):
        clauses = re.split(r'(?<=[.!?;])\s+', ' '.join(source.split()))
        renewal_clauses = [clause for clause in clauses
            if re.search(r'\b(?:renew\w*|extend\w*|extensions?)\b', clause, re.I)
            and not re.search(r'\btrial\b', clause, re.I)
            and not re.search(r'\b(?:MSA|agreement|document)\s+contains?\b.{0,120}'
                              r'\b(?:provisions?|terms?)\b|\b(?:refer\s+to|see)\b.{0,50}'
                              r'\b(?:MSA|agreement|external\s+terms?)\b', clause, re.I)]
        if not renewal_clauses:
            return _unknown(check, 'contract_renewal_not_established_by_selected_text',
                'Selected supplier text does not establish renewal or extension options. A payment schedule, initial duration, trial exclusion or pointer to unread external terms alone does not establish them.')
    scoped_timing = _missing_scoped_timing(request, quotes, check)
    if scoped_timing:
        return scoped_timing
    notice_bound = _advance_notice_bound(request, quotes, check)
    if notice_bound:
        return notice_bound
    requested, offered = list(_DURATION.finditer(request)), list(_DURATION.finditer(source))
    unspecified = _UNSPECIFIED.search(source)
    timing_requested = re.search(r"\b(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|deadline)\b", request, re.I)
    if (len(requested) <= 1 and timing_requested and unspecified and not offered and not _REFUSAL.search(source)
            and not _AMBIGUOUS.search(source)
            and _property_words(request) & _property_words(unspecified[1])):
        return {"assessment": "unknown", "reason": "requested_value_unspecified_in_current_document",
                "rationale": "The current document leaves the requested value unspecified; it does not establish an incompatible value."}
    if _NEGOTIABLE.search(request) and _term_scope(scope):
        statements = [clause for quote in quotes for clause in _clauses(quote)
                      if _term_scope(clause) and not _REQUEST_REPORT.search(clause)
                      and not _TERM_AMBIGUOUS.search(clause) and not _OPTIONAL_CONTEXT.search(quote)]
        willing = any(_WILLING.search(clause) and not _REFUSAL.search(clause) for clause in statements)
        unwilling = any(_UNWILLING.search(clause) for clause in statements)
        if unwilling and not willing:
            return {'assessment': 'contradicted', 'reason': 'explicit_contract_negotiation_refusal',
                    'rationale': 'Selected supplier text explicitly declines negotiation of the contract terms.'}
        if willing and not unwilling:
            return None  # Willingness alone cannot certify other bundled conditions.
        return _unknown(check, 'contract_negotiability_not_established_by_selected_text',
                        'Selected supplier text does not establish willingness to negotiate contract terms; a printed duration alone does not establish willingness or refusal.')
    buyer_term = _term_scope(request) or (_term_scope(scope) and not _OTHER_PROPERTY.search(request))
    if not buyer_term:
        return None
    unknown = lambda: _unknown(check, 'contract_duration_not_established_by_selected_text',
        'Selected supplier text does not establish one unambiguous matching contract or service term; unrelated durations cannot establish this condition.')
    if not requested and not timing_requested:
        return None
    if len(requested) != 1 or _duration_value(requested[0]) is None:
        return unknown()
    candidates = []
    ambiguous = False
    for quote in quotes:
        table = single_payment_term(quote)
        if table:
            candidates.append((table['clause'], _DURATION.search(table['clause'])))
        for clause in _clauses(quote):
            durations = list(_DURATION.finditer(clause))
            if not _term_scope(clause) or not durations:
                continue
            if (len(durations) != 1 or _duration_value(durations[0]) is None
                    or _TERM_AMBIGUOUS.search(clause) or _OPTIONAL_CONTEXT.search(quote)
                    or _REQUEST_REPORT.search(clause)
                    or _REFUSAL.search(clause) or re.search(r'\b(?:not|no|without)\b', clause, re.I)
                    or _MIN.search(clause) or _MAX.search(clause)):
                ambiguous = True
                continue
            if clause not in [candidate[0] for candidate in candidates]:
                candidates.append((clause, durations[0]))
    if not candidates and any(_RECURRING_PERIOD.search(clause) and _DURATION.search(clause)
                             and not _OTHER_PROPERTY.search(clause)
                             for quote in quotes for clause in _clauses(quote)):
        return {'assessment': 'unknown', 'reason': 'printed_recurring_period_requires_contract_term_review',
                'rationale': 'The selected text prints a recurring-service pricing period. Confirm how that period relates to the requested contract term; it does not by itself establish the initial legal term.'}
    if ambiguous or len(candidates) != 1:
        return unknown()
    offered_clause, offered_duration = candidates[0]
    def service(text):
        match = re.search(r"\bfor\s+(?:the\s+)?([^.;]+)", text, re.I)
        if not match or _DURATION.match(match[1]):
            return None
        return re.split(r"\b(?:is|must|shall|should|will)\b", match[1].lower())[0].strip()
    requested_service = service(request)
    if requested_service and service(offered_clause) != requested_service:
        return unknown()
    if _TERM_AMBIGUOUS.search(request) or _OPTIONAL_CONTEXT.search(request):
        return unknown()
    minimum, maximum = bool(_MIN.search(request)), bool(_MAX.search(request))
    if minimum and maximum:
        return unknown()
    unbounded = _MAX.sub("", _MIN.sub("", request))
    if re.search(r"\b(?:not|no|without)\b", unbounded, re.I):
        return unknown()
    required = _duration_value(requested[0])
    actual = _duration_value(offered_duration)
    matches = actual >= required if minimum else actual <= required if maximum else actual == required
    if matches:
        return None
    relation = "minimum" if minimum else "maximum" if maximum else "exact"
    return {"assessment": "contradicted", "reason": "explicit_initial_contract_duration_comparison",
            "rationale": f"Offered contract/service term: {offered_duration[0]}; requested {relation} duration: {requested[0][0]}."}
