# SPDX-License-Identifier: MPL-2.0
"""Conservative selected-evidence check, not a test for document existence."""
import re

_HISTORY = re.compile(r'\b(?:work[- ]history|employment[- ]history|career[- ]history|resumes?|résumés?|cv|curriculum vitae)\b', re.I)
_POINTER_REQUEST = re.compile(r'\b(?:urls?|links?|hyperlinks?|headings?|titles?|list of documents)\b', re.I)
_BIOGRAPHICAL_CONTENT = re.compile(r'\b(?:experience|worked|employed|employment|led|delivered|built|'
    r'implemented|developed|served|degree|education|university|previous\s+roles?)\b|\b(?:19|20)\d{2}\b', re.I)
_TEMPORARY = r'\b(?:temporary|interim|loaners?|loaned|lending|lend|loans?)\b'
_REPLACEMENT_ITEM = r'\b(?:parts?|replacements?|spares?|equipment|devices?|units?|hardware|routers?|servers?|workstations?|laptops?)\b'
_BUSINESS_APP = r'(?:line[- ]of[- ]business|LOB|business(?:[- ]critical)?|operational)\s+(?:applications?|software|systems?)'
_ENTITIES = {'server': r'servers?', 'workstation': r'(?:workstations?|desktops?)'}
_PROPERTIES = r'(?:stability|performance)'
_ENTITY_OBJECT = (rf'(?:all\s+|every\s+|the\s+|our\s+|your\s+)?(?:servers?|workstations?|desktops?)'
                  rf"(?:['’]s?)?(?:\s+{_PROPERTIES}(?:\s+and\s+{_PROPERTIES})?)?")
_MONITOR_OBJECT = rf'{_ENTITY_OBJECT}(?:\s+and\s+{_ENTITY_OBJECT})?'


def explain_scope_issues(issues):
    """Keep diagnostic codes in receipts and ordinary language in the report."""
    descriptions = {
        'opposing_commitment_for_same_condition_not_established': 'The passage does not state an opposing commitment for this condition. Missing evidence remains unresolved.',
        'advanced_malware_scope_not_established_by_selected_text': 'Advanced malware protection is not explicitly established by the quoted supplier scope.',
        'reference_postal_address_not_established_by_selected_text': 'Postal addresses are not established for the references. Confirm whether email addresses satisfy the buyer’s address requirement.',
        'recovery_sla_duration_not_established_by_selected_text': 'The passage does not establish a recovery SLA for this system and duration.',
        'recovery_point_duration_or_scope_not_established_by_selected_text': 'The passage does not establish the requested recovery point objective (allowed data-loss window) for this system. Backup monitoring, recovery time and support coverage do not establish an RPO.',
        'named_site_backup_does_not_establish_all_workstation_coverage': 'The quoted backup is tied to a named site; coverage of all workstations remains unestablished.',
        'buyer_needs_description_is_not_supplier_commitment': 'The passage describes buyer needs without establishing the supplier commitment.',
        'requested_subject_not_established_by_selected_text': 'The passage does not establish the requested subject or property.',
        'complete_service_scope_requires_full_requirement_review': 'A general service or payment passage cannot establish coverage of every required IT service. Review the individual requirements and unresolved scope together.',
        'complete_cost_proposal_requires_document_review': 'A price heading, pointer or hourly-rate schedule cannot establish that every required item and cost is included. Review the complete cost proposal against the required scope.',
        'reference_field_count_not_established': 'The selected client-reference section does not establish this field for the requested number of distinct contacts. Staff biographies and incomplete reference excerpts cannot supply it.',
    }
    return ' '.join(descriptions.get(issue, issue.replace('_', ' ').replace(':', ': ')) for issue in issues)


def _evidence_clauses(quotes):
    # Keep wrapped text together, but never borrow another sentence/bullet's
    # entity or action. Exact quotations are unchanged in the caller's records.
    return [' '.join(clause.split()) for quote in quotes for clause in
            re.split(r'(?<=[.!?;])\s+|[\u2022\uf0b7\uf0a7]+|\n[ \t]*\n', quote) if clause.strip()]


def _positive_clause(clause):
    # "At no charge" does not negate loaner provision.
    clause = re.sub(r'\bat\s+no\s+(?:additional\s+)?(?:charge|cost)\b', '', clause, flags=re.I)
    return not re.search(r"\b(?:not|never|cannot|can't|won't|optional|unless|if|may|might|could|"
                         r"aims?|targets?|hopes?|except|excluding)\b|\bno\s+(?:loaners?|temporary|replacement)", clause, re.I)


def _monitoring_objects(clause):
    boundary = (r'(?=$|[.,:;]|\s+(?:to|using|via|continuously|constantly|around|round|24)\b|'
                r'\s+(?:is|are)\s+(?:included|provided|offered)\b)')
    patterns = [
        rf'\bmonitor(?:s|ing)?(?:\s+and\s+manag(?:e|es|ing|ement))?\s+(?:(?:of|for)\s+)?(?P<object>{_MONITOR_OBJECT}){boundary}',
        rf'\b(?P<property>{_PROPERTIES}(?:\s+and\s+{_PROPERTIES})?)\s+of\s+(?P<object>{_MONITOR_OBJECT})\s+(?:is|are)\s+(?:continuously\s+)?monitored\b',
        rf'\bmonitor(?:s|ing)?\s+(?:the\s+)?(?P<property>{_PROPERTIES}(?:\s+and\s+{_PROPERTIES})?)\s+of\s+(?P<object>{_MONITOR_OBJECT}){boundary}',
        rf'\b(?P<object>{_MONITOR_OBJECT})\s+(?:(?:is|are)\s+(?:continuously\s+)?)?monitor(?:ing|ed)\b',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, clause, re.I):
            for obj in re.split(r'\s+and\s+(?=(?:(?:all|the|our|your)\s+)?(?:servers?|workstations?|desktops?)\b)', match['object'], flags=re.I):
                properties = set(re.findall(_PROPERTIES, obj, re.I))
                properties.update(re.findall(_PROPERTIES, match.groupdict().get('property') or '', re.I))
                yield obj, {word.casefold() for word in properties}


def _configuration_support_issue(target, clauses):
    requested = re.search(r'\bconfigur\w*\s*(?:/|&|and)\s*support\w*\s+(?:(?:of|for)\s+)?([^,;.]+)', target, re.I)
    if not requested:
        return []  # A true OR or a support-only condition does not require both.
    product = re.sub(r'\b(?:the|products?|software|services?)\b', '', requested[1], flags=re.I)
    product = ' '.join(product.split()).strip()
    if not product:
        return ['configuration_and_support_scope_not_established_by_selected_text']
    if re.search(r'\b(?:microsoft|m365)\b', product, re.I):
        # Keep the product binding: workstation configuration or Office support
        # alone cannot supply the missing configuration commitment.
        product_pattern = r'(?:Microsoft(?:\s*365)?|M365)\b(?:\s+(?:products?|services?|tenant|software))?'
    else:
        product_pattern = re.escape(product)
    found = set()
    for clause in clauses:
        if (not _positive_clause(clause)
                or re.search(r'\b(?:excluded|excludes?)\b', clause, re.I)):
            continue
        for action, pattern in [('configuration', r'(?:configur\w*|administ(?:er\w*|ration))'),
                                ('support', r'support\w*')]:
            if action == 'configuration' and re.search(
                    r'\b(?:billing|invoice|financial|payment|procurement)\s+administration\b', clause, re.I):
                pattern = r'configur\w*'  # Commercial administration is not product configuration.
            between = r'(?:\s*(?:and|&)\s*(?:configur\w*|support\w*))?'
            if (re.search(rf'\b{pattern}{between}\s+(?:(?:of|for|the|your|our)\s+)*{product_pattern}', clause, re.I)
                    or re.search(rf'\b{product_pattern}\s+{pattern}\b', clause, re.I)
                    or re.search(rf'\b{product_pattern}\s+(?:configuration|support)\s*(?:and|&)\s*{pattern}\b', clause, re.I)):
                found.add(action)
        # A named administration service can bind its platforms in parentheses,
        # e.g. MFA administration and support (DUO, M365, Google Workspace).
        # Wrapped PDF columns may interleave words; stay within this clause and
        # bind both actions to that service's own platform list. This admits
        # partial configuration evidence for model assessment, never a verdict
        # or installation/other named-product coverage.
        if (_positive_clause(clause)
                and not re.search(r'\b(?:excluded|excludes?|except|excluding)\b', clause, re.I)
                and re.search(r'\b(?:MFA|multi[- ]factor authentication|tenant|user|access|'
                              r'group policy|security policy)\s+administration\s*(?:and|&|/)'
                              rf'.{{0,80}}?\bsupport\s*\([^)]*\b{product_pattern}', clause, re.I)):
            found.update({'configuration', 'support'})
    return [] if found == {'configuration', 'support'} else ['configuration_and_support_scope_not_established_by_selected_text']


def _included_software_lifecycle_maintenance(quotes):
    """A broad included software commitment is relevant to business applications.

    Keep the full selected passage's inclusion and limits together. This only
    permits model assessment; it never creates a Supported verdict.
    """
    for quote in quotes:
        clauses = _evidence_clauses([quote])
        if any(re.search(r'\b(?:optional|except|excluding|excluded|only|limited)\b',
                         clause, re.I) or not _positive_clause(clause) for clause in clauses):
            continue
        included = any(
            re.search(r'\b(?:provid\w*|offer\w*)\b.{0,70}\bcomprehensive\b.{0,70}'
                      r'\b(?:software|applications?)\b.{0,40}\blife\s*cycle\s+management\b', clause, re.I)
            and re.search(r'\bas\s+part\s+of\b.{0,50}\b(?:services?|offering|agreement)\b', clause, re.I)
            for clause in clauses)
        if not included:
            continue
        software = r'(?:(?:hardware\s*(?:and|&|/)\s*)?(?:software|applications?))'
        boundary = r'(?=$|[.,;:]|\s+(?:to\s+ensure|for\s+optimal)\b)'
        if any(re.search(rf'\bmaintenance\s+(?:of|for)\s+{software}{boundary}|'
                         rf'\b(?:maintain\w*|upkeep\s+of)\s+{software}{boundary}|'
                         rf'\b{software}\s+(?:updates?\s+and\s+)?maintenance{boundary}',
                         clause, re.I) for clause in clauses):
            return True
    return False


def _service_scope_issues(target, selected_text, context):
    enclosing = (context or {}).get('enclosing_span', {}).get('text', target)
    governing = '\n'.join(span['text'] for span in (context or {}).get('governing_spans', []))
    clauses = [clause for clause in _evidence_clauses(selected_text) if _positive_clause(clause)]
    if re.search(r'\bgamif\w*\b', target, re.I) and not any(
            re.search(r'\bgamif\w*\b|\bgame[- ]based\b', clause, re.I) for clause in clauses):
        return ['requested_training_method_not_established_by_selected_text']
    # Vendor-prefixed product scope is narrower than generic monitoring or
    # management. MS/Microsoft spelling can vary; the actual product must occur.
    products = re.findall(r'\b(?:MS|Microsoft|Google|AWS|Azure|Oracle)\s+([A-Z][A-Za-z0-9]+)',
                          target if target != target.casefold() else (context or {}).get('selected_span', {}).get('text', ''))
    if products and any(not any(re.search(r'(?<!\w)' + re.escape(product) + r'(?!\w)', clause, re.I)
                                for clause in clauses) for product in products):
        return ['named_service_product_not_established_by_selected_text']
    if re.search(r'\b(?:MS|Microsoft)\s+Sentinel\b', target, re.I) and not any(
            re.search(r'\b(?:MS|Microsoft|Azure)\s+Sentinel\b', clause, re.I) for clause in clauses):
        return ['named_service_vendor_identity_not_established_by_selected_text']
    if re.search(r'\bdark[- ]web\b', target, re.I) and not any(
            re.search(r'\bdark[- ]web\b|\bdarknet\b', clause, re.I) for clause in clauses):
        return ['requested_security_research_scope_not_established_by_selected_text']
    cadence = re.search(r'\b(?:every|each)\s+(\d+)\s+(hours?|minutes?|days?)\b', target, re.I)
    if cadence and re.search(r'\bback[- ]?ups?\b', target, re.I) and not any(
            re.search(rf'\b(?:every|each)\s+{cadence[1]}\s+{cadence[2].rstrip("s")}s?\b', clause, re.I)
            and re.search(r'\bback[- ]?ups?\b', clause, re.I) for clause in clauses):
        return ['requested_backup_frequency_not_established_by_selected_text']
    configuration = _configuration_support_issue(target, clauses)
    if configuration:
        return configuration
    if (re.search(r'\binstall\w*\b', target, re.I)
            and re.search(r'\b(?:Microsoft|M365|MS\s+Office)\b', target + ' ' + enclosing, re.I)):
        # An installation fragment inherits its enclosing software scope.
        # Hardware installation and Office support are separate commitments.
        product = r'(?:Microsoft(?:\s+(?:365|Office|products?|software))?|M365|MS\s+Office)\b'
        actions = r'install\w*'
        joined = r'(?:\s*(?:,|and|&)\s*(?:sales?|configur\w*|support\w*|install\w*))*'
        patterns = [
            rf'\b{actions}{joined}\s+(?:(?:of|for|the|your|our)\s+)*{product}',
            rf'\b{product}\s+{actions}\b',
            rf'\b{product}\s+(?:sales?|configuration|support){joined}\s*(?:and|&)\s*{actions}\b',
        ]
        if not any(re.search(pattern, clause, re.I) for clause in clauses for pattern in patterns):
            return ['software_installation_product_scope_not_established_by_selected_text']
    if re.search(_TEMPORARY, target, re.I) and re.search(_REPLACEMENT_ITEM, target + ' ' + enclosing, re.I):
        has_loaner = any(re.search(_REPLACEMENT_ITEM, clause, re.I) and (
            (re.search(rf'{_TEMPORARY}\s+(?:replacement\s+|spare\s+)?{_REPLACEMENT_ITEM}|\bon loan\b', clause, re.I) and re.search(
                r'\b(?:provid\w*|suppl\w*|furnish\w*|includ\w*|offer\w*|loan\w*|lend\w*|available)\b', clause, re.I))
            or (re.search(r'\b(?:ship|supply|provide|furnish)\w*\b', clause, re.I)
                and re.search(r'\bfor use (?:during|while|until)\b.{0,65}\b(?:repair\w*|replaced|replacement)\b', clause, re.I)))
            for clause in clauses)
        if not has_loaner:
            return ['temporary_replacement_provision_not_established_by_selected_text']

    properties = {word.casefold() for word in re.findall(_PROPERTIES, target, re.I)}
    scope = target + '\n' + enclosing + '\n' + governing
    if properties and re.search(r'\bmonitor\w*\b', scope, re.I):
        entities = {name for name, pattern in _ENTITIES.items() if re.search(r'\b' + pattern + r'\b', target, re.I)}
        if not entities:
            entities = {name for name, pattern in _ENTITIES.items() if re.search(r'\b' + pattern + r'\b', enclosing, re.I)}
        if entities:
            found = set()
            for clause in clauses:
                for obj, observed_properties in _monitoring_objects(clause):
                    # Explicit monitoring of the entire entity is broader than
                    # an isolated metric. A metric-specific promise must match.
                    if observed_properties and not properties <= observed_properties:
                        continue
                    found.update(name for name, pattern in _ENTITIES.items()
                                 if re.search(r'\b' + pattern + r'\b', obj, re.I))
            if not entities <= found:
                return ['monitoring_entity_or_property_not_established_by_selected_text']

    app_scope = re.search(_BUSINESS_APP, target, re.I) or (
        re.search(r'\b(?:maintenan\w*|updat\w*)\b', target, re.I)
        and re.search(_BUSINESS_APP, enclosing, re.I))
    if app_scope and re.search(r'\b(?:maintenan\w*|maintain\w*)\b', target, re.I):
        app_object = rf'(?:(?:(?:all|every|our|your|the)\s+)?{_BUSINESS_APP}|(?:all|every)\s+(?:applications?|software))'
        patterns = [
            rf'\b(?:maintain\w*|upkeep|maintenance)(?:\s+and\s+updat\w*)?\s+(?:(?:of|for)\s+)?{app_object}\b',
            rf'\b{app_object}\s+(?:(?:updates?\s+and\s+)?maintenance|upkeep|(?:is|are)\s+maintained)\b',
            rf'\bupdat\w*\s+and\s+maintain\w*\s+{app_object}\b',
        ]
        if (not any(re.search(pattern, clause, re.I) for clause in clauses for pattern in patterns)
                and not _included_software_lifecycle_maintenance(selected_text)):
            return ['business_application_maintenance_not_established_by_selected_text']
    return []


def _buyer_need_only_issue(target, selected_text):
    """Do not convert quoted buyer needs into a supplier delivery promise.

    Check the requested delivery mode in the clauses that actually mention it;
    unrelated supplier marketing elsewhere in the citation cannot establish it.
    """
    modes = [pattern for pattern in (r'\bon[- ]site\b', r'\bremot(?:e|ely)\b')
             if re.search(pattern, target, re.I)]
    if not modes:
        return []
    clauses = _evidence_clauses(selected_text)
    reported = re.compile(r'\b(?:client\s+needs|(?:buyer|client|customer|town|authority)\b.{0,80}\b'
                          r'(?:seek\w*|requires?|requests?|needs?|expects?|wants?))', re.I)
    for mode in modes:
        relevant = [clause for clause in clauses if re.search(mode, clause, re.I)]
        if not relevant or all(reported.search(clause) for clause in relevant):
            return ['supplier_delivery_mode_not_established_by_selected_text']
    return []


def _full_source_scope_issues(obligation, check, context, quotes):
    """Reject absent properties and unrelated conflicts; never create support.

    These are service/property grammars, not document- or benchmark-specific
    rules. A broad category commitment remains eligible for model comparison;
    specific features and artifact contents need their own evidence.
    """
    target = obligation['source_text']
    # Completeness is an aggregate document/scope judgment. A single selected
    # passage cannot certify all services or all required price classifications.
    # Keep the actual individual service and hourly-rate conditions assessable.
    if re.search(r'\bcovers? all aspects of IT\b', target, re.I):
        return ['complete_service_scope_requires_full_requirement_review']
    if re.search(r'\bcomplete\s+(?:cost\s+proposal|work\s+estimate)\b', target, re.I):
        return ['complete_cost_proposal_requires_document_review']
    context = context or {}
    from bidtrace_core.local.relations import reference_field_request, reference_contact_field_fact
    if reference_field_request(target, context) and not reference_contact_field_fact(target, context, check.get('evidence', [])):
        return ['reference_field_count_not_established']
    enclosing = context.get('enclosing_span', {}).get('text', target)
    governing = [s['text'] for s in context.get('governing_spans', [])]
    questions = ' '.join(s['quote'] for s in context.get('paired_buyer_questions', []))
    # An incoming-provider fragment inherits its named service, not a different
    # sibling. A named application list inherits the paired maintenance question.
    if re.search(r'\b(?:provided by (?:current|incoming)|provided by incoming|will need to be provided)\b', enclosing, re.I):
        target = next((s for s in reversed(governing) if re.search(r'\w', s)), questions or enclosing)
    clauses = [c for c in _evidence_clauses(quotes) if _positive_clause(c)]
    source = '\n'.join(clauses)
    if check['assessment'] == 'contradicted':
        from bidtrace_core.local.relations import relation_override
        compared = relation_override({**obligation, 'sentence_context': enclosing, 'selected_scope':context}, check)
        if compared is not None:
            return []  # Same-property relation supplies its own exact scope check.
        # Absence of a promise, a different price, or response clock is not an
        # opposing commitment. Preserve literal refusals for model comparison.
        negative = [c for c in _evidence_clauses(quotes) if re.search(
            r'\b(?:not|no|never|cannot|excluded|excludes|unable|refus\w*)\b', c, re.I)]
        stop = set('the a an of for to and or as on in at by with is are be must shall should will '
                   'provide supplier condition selected containing buyer clause scope no not never cannot'.split())
        words = lambda t: {w.rstrip('s') for w in re.findall(r'\w+', t.casefold()) if w not in stop}
        if (re.search(_TEMPORARY, target, re.I) and
                any(re.search(r'\b(?:no loaners?|loaners? (?:are )?not provided)\b', c, re.I) for c in negative)):
            return []
        opposed = any(words(target) & words(c) for c in negative)
        # "Cannot guarantee resolution" cannot negate hardware ownership or a
        # transition option merely because both clauses contain "services".
        if re.search(r'\b(?:leased|lease|transferred|transfer|partner|transition|transiƟon)\b', target, re.I):
            opposed = any(re.search(r'\b(?:lease|leased|transfer\w*|partner|transition|transiƟon)\b', c, re.I)
                          for c in negative)
        if not opposed:
            return ['opposing_commitment_for_same_condition_not_established']
        return []
    if check['assessment'] != 'supported':
        return []
    if quotes and all(re.search(r'\bClient Needs\s*:', q, re.I) and not re.search(
            r'\bProposed Solution\s*:|\b(?:We|the supplier) (?:will|shall) provide\b', q, re.I) for q in quotes):
        return ['buyer_needs_description_is_not_supplier_commitment']
    requested = target + ' ' + enclosing
    # A duration fragment can inherit RPO from its enclosing buyer clause.
    # Supplier RTO, helpdesk/SOC hours or backup cadence are different clocks.
    # Only admit an explicit RPO duration for the selected system to model
    # comparison; this reject-only guard never supplies a positive verdict.
    rpo = r'\b(?:RPO|recovery[- ]point\s+objectives?|(?:maximum|allowed)\s+data[- ]loss(?:\s+window)?)\b'
    from bidtrace_core.local.relations import _DURATION
    if _DURATION.search(target) and re.search(rpo, requested, re.I):
        locations = {'cloud': r'\bcloud\b', 'on-prem': r'\bon[- ]prem(?:ises)?\b'}
        selected_locations = [pattern for pattern in locations.values() if re.search(pattern, target, re.I)]
        if not any(re.search(rpo, clause, re.I) and _DURATION.search(clause)
                   and all(re.search(pattern, clause, re.I) for pattern in selected_locations)
                   and not re.search(r'\b(?:excluded|excludes?|recommended|recommendation)\b|'
                                     r'\b(?:client|buyer|customer)\s+(?:needs|requires|requests)\b', clause, re.I)
                   for clause in clauses):
            return ['recovery_point_duration_or_scope_not_established_by_selected_text']
    if re.search(r'\badvanced malware protection\b', target, re.I) and not re.search(
            r'\badvanced (?:malware|anti[- ]malware|threat) protection\b', source, re.I):
        return ['advanced_malware_scope_not_established_by_selected_text']
    if re.fullmatch(r'(?:Configuration|Monitoring|management)[, &]*(?:and)?\s*', target.strip(), re.I):
        inherited = _service_scope_issues(enclosing, quotes, context)
        if inherited:
            return inherited
    policy = re.search(r'\b([A-Z][a-z]+) State (?:[A-Z][a-z]+ )*Standards\b', enclosing)
    if policy and not re.search(r'\b' + re.escape(policy[1]) + r'\b', source, re.I):
        return ['named_jurisdiction_policy_not_established_by_selected_text']
    if re.search(r'\b(?:one|\d+)\s+hours?\b', target, re.I) and re.search(r'\b(?:BCDR|RTO|recovery|restoration)\b', requested, re.I):
        if not any(re.search(r'\b(?:BCDR|recover\w*|restor\w*)\b', c, re.I)
                   and re.search(r'\b(?:one|two|three|four|\d+)\s+hours?\b', c, re.I)
                   and not re.search(r'\bresponse times?\b', c, re.I) for c in clauses):
            return ['recovery_sla_duration_not_established_by_selected_text']
    if re.search(r'\blicensure\b', requested, re.I) and re.search(r'\blicensure\s*,\s*certifications\b', source, re.I):
        if not re.search(r'\b(?:licen[sc]ed|licen[sc]e number|licen[sc]e held|state licen[sc]e)\b', source, re.I):
            return ['licensure_heading_does_not_establish_licensure_content']
    if re.search(r'\bAnti[- ]Virus\b', requested, re.I):
        if re.search(r'\bSubscription\b', target, re.I) and not any(re.search(
                r'\b(?:anti[- ]virus|antivirus)\b.{0,70}\b(?:subscription|licen[sc]|included)\w*\b', c, re.I) for c in clauses):
            return ['antivirus_subscription_not_established_by_selected_text']
        action = 'updat\\w*' if re.fullmatch(r'Updates[,\s]*', target.strip(), re.I) else 'manag\\w*' if re.fullmatch(r'Management[,\s]*', target.strip(), re.I) else None
        if action and not any(re.search(rf'\b(?:anti[- ]virus|antivirus)\s+{action}\b|\b{action}\s+(?:of\s+)?(?:anti[- ]virus|antivirus)\b', c, re.I) for c in clauses):
            return ['antivirus_action_scope_not_established_by_selected_text']
    if re.search(r'\bcomprehensive logging\b', enclosing, re.I) and re.search(r'\bconfigur\w*|monitor\w*\b', target, re.I):
        if not re.search(r'\b(?:logging|logs)\b', source, re.I):
            return ['logging_configuration_scope_not_established_by_selected_text']
    if re.search(r'\bonsite|on-site\b', enclosing, re.I) and re.search(r'\brequired\b', enclosing, re.I):
        delivery = _buyer_need_only_issue(enclosing, quotes)
        if delivery:
            return delivery
    if target != obligation['source_text']:
        inherited = _service_scope_issues(target, quotes, context)
        if inherited:
            return inherited
        if re.search(r'\bback[- ]?up\b', target, re.I):
            names = re.findall(r'\b[A-Z][a-z]+[A-Z][A-Za-z]*\b', target)
            if any(not re.search(r'\b' + re.escape(name) + r'\b', source, re.I) for name in names):
                return ['inherited_backup_product_scope_not_established_by_selected_text']
    if re.search(r'\b(?:Configuration,|monitoring and)', target, re.I) and re.search(
            r'\b(?:Defender|Purview)\b', enclosing, re.I):
        if any(not re.search(r'\b' + re.escape(name) + r'\b', source, re.I)
               for name in re.findall(r'\b(?:Defender|Purview)\b', enclosing, re.I)):
            return ['specific_named_product_not_established_by_selected_text']
    if re.search(r'\b(?:telemetry|logs?|M365 services|altering)\b', target, re.I):
        if not re.search(r'\b(?:logs?|telemetry|log alert\w*)\b', source, re.I):
            return ['requested_telemetry_or_log_scope_not_established_by_selected_text']
    if re.search(r'\bbackup job monitoring\b', target, re.I) and not any(re.search(
            r'\bback[- ]?up\b.{0,90}\bmonitor\w*\b|\bmonitor\w*\b.{0,90}\bback[- ]?up\b', c, re.I) for c in clauses):
        return ['backup_monitoring_not_established_by_selected_text']
    if re.search(r'\bback[- ]?ups?\b', target, re.I) and re.search(r'\bservers?\b', target, re.I):
        if not any(re.search(r'\bserver\s+back[- ]?up\b|\bback[- ]?ups?\s+(?:of|for)\s+(?:all\s+)?servers?\b', c, re.I) for c in clauses):
            return ['server_backup_scope_not_established_by_selected_text']
    # Brand identity alone never supplies an unmentioned feature. These names
    # are taken from a product-list grammar in the buyer clause.
    examples = re.search(r'\bsuch as\s+([^.!?]+)', enclosing, re.I)
    if examples and re.search(r'\b(?:Microsoft|Google|AWS|Azure|Oracle)\b', enclosing, re.I):
        for name in re.findall(r'\b[A-Z][A-Za-z0-9]+\b', examples[1]):
            if re.search(r'\b' + re.escape(name) + r'\b', target) and not re.search(
                    r'\b' + re.escape(name) + r'\b', source, re.I):
                return ['specific_named_product_not_established_by_selected_text']
    features = [
        (r'\bISP\b|\bconnectivity monitoring\b', r'\bISP\b|\b(?:internet|connectivity)\b.{0,70}\bmonitor\w*\b|\bmonitor\w*\b.{0,70}\b(?:internet|connectivity)\b'),
        (r'\bdisk space\b', r'\b(?:disk|storage)\s+(?:space|capacity)\b'),
        (r'\b(?:Windows )?event log\b', r'\b(?:Windows )?event logs?\b'),
        (r'\bautomatic restart\b', r'\b(?:automatic|automated)\b.{0,60}\brestart\w*\b|\brestart\w*\b.{0,60}\b(?:automatic|automated)\b'),
        (r'\bautomated fixes?\b|\bscripts?\b', r'\b(?:automated|automatic)\b.{0,60}\bfix\w*\b|\bscripts?\b'),
        (r'\bcomprehensive logging\b', r'\b(?:comprehensive|central\w*|all)\s+(?:logging|logs)\b'),
        (r'\bweekly\b', r'\bweekly\b|\bonce (?:a|per) week\b|\bevery (?:seven|7) days\b|\bdaily\b|\bevery day\b'),
    ]
    for requested_feature, evidence_feature in features:
        if re.search(requested_feature, target, re.I) and not re.search(evidence_feature, source, re.I):
            return ['specific_service_property_not_established_by_selected_text']
    if re.search(r'\b(?:client portal|ability to see|status of (?:previously|open)|'
                 r'access those monthly|request work|open a ticket)\b', requested, re.I):
        if not re.search(r'\b(?:portal|ticket)\b', source, re.I):
            return ['portal_function_not_established_by_selected_text']
        if re.search(r'\b(?:status|access.*reports|see open)\b', requested, re.I) and not re.search(
                r'\b(?:status|view|access|see)\b.{0,70}\b(?:tickets?|incidents?|reports?|requests?|issues)\b', source, re.I):
            return ['portal_visibility_not_established_by_selected_text']
    if re.search(r'\b(?:incidents from|effects those incidents|client requests|their status)\b', target, re.I) and re.search(
            r'\breport\w*\b', ' '.join(governing), re.I):
        if not (re.search(r'\breport\w*\b', source, re.I) and re.search(
                r'\b(?:incidents?|SLAs?|requests?|status)\b', source, re.I)):
            return ['requested_report_contents_not_established_by_selected_text']
    if re.search(r'\b(?:associated costs|cost proposal|work estimate)\b', target, re.I) and not re.search(
            r'\$\s*[\d,]+|\b\d+(?:[.,]\d+)?\s*(?:USD|EUR|SEK|dollars|euros)\b', source, re.I):
        return ['itemized_pricing_not_established_by_selected_text']
    if re.search(r'\b(?:transition|transiƟon)\b', requested, re.I) and re.search(r'\binformation will be needed\b', requested, re.I):
        if not re.search(r'\b(?:transition|onboarding)\b.{0,100}\b(?:need|required|information|documents)\b', source, re.I):
            return ['transition_input_requirements_not_established_by_selected_text']
    if _HISTORY.search(target) and not _POINTER_REQUEST.search(target):
        resume_part = re.split(r'\b(?:Business References|References:)\b', '\n'.join(quotes), flags=re.I)[0]
        if re.search(r'https?://', resume_part) and not _BIOGRAPHICAL_CONTENT.search(
                re.sub(r'https?://\S+', '', resume_part)):
            return ['work_history_content_not_established_by_selected_text']
    if re.search(r'\blicensure\b', requested, re.I) and not re.search(r'\b(?:licensure|licen[sc]ed|licen[sc]e number)\b', source, re.I):
        return ['requested_licensure_not_established_by_selected_text']
    if re.search(r'\b(?:standard SLAs?|enhanced SLAs?)\b', requested, re.I) and not re.search(
            r'\b(?:SLA|service level|response times?|response stage|resolution times?)\b', source, re.I):
        return ['requested_service_level_terms_not_established_by_selected_text']
    if re.search(r'\benhanced SLAs?\b', target, re.I) and not re.search(
            r'\b(?:enhanced|premium|alternative)\s+(?:SLAs?|service levels?)\b', source, re.I):
        return ['enhanced_service_level_option_not_established_by_selected_text']
    if re.search(r'\bscheduling of work\b|\bprogression through the project\b', requested, re.I) and not re.search(
            r'\b(?:schedule|milestones?|phases?|progression|implementation plan)\b', source, re.I):
        return ['project_schedule_not_established_by_selected_text']
    if re.search(r'\bcost,?\s*plus\b|\bsoftware will be based on cost\b', requested, re.I) and not re.search(
            r'\bcost\s*(?:plus|\+)\b|\badministrative fee\b', source, re.I):
        return ['procurement_pricing_basis_not_established_by_selected_text']
    if re.search(r'\b(?:references|reference)\b', enclosing, re.I) and re.search(
            r'\b(?:title|address|phone|fax|name)\b', target, re.I) and not re.search(
            r'\b(?:business references|clients.*agreed|reference contact)\b', source, re.I):
        return ['reference_contact_field_not_established_by_selected_text']
    if re.search(r'\breferences\b', enclosing, re.I) and re.fullmatch(r'address[,\s]*', target.strip(), re.I):
        if not re.search(r'\b\d+\s+[A-Z][\w .-]+\b(?:Street|St|Road|Rd|Lane|Ln|Avenue|Ave|Drive|Dr)\b', source):
            return ['reference_postal_address_not_established_by_selected_text']
    if re.fullmatch(r'workstations[,\s]*', target.strip(), re.I) and re.search(r'\bback[- ]?ups?\b', enclosing, re.I):
        if re.search(r'\b(?:workstations?|desktops?)\b\s*(?:\([^)]*\))?\s*[-–]\s*[A-Z]', source, re.I) and not re.search(
                r'\ball (?:workstations|desktops)\b', source, re.I):
            return ['named_site_backup_does_not_establish_all_workstation_coverage']
    # A named application may be covered by comprehensive included software
    # maintenance. A price table cannot establish that scope.
    if re.search(r'\bapplications\b', questions, re.I) and re.search(r'\bmaintenance\b', questions, re.I):
        if not (_included_software_lifecycle_maintenance(quotes) or
                re.search(re.escape(target.strip(' ,.;')), source, re.I) and re.search(r'\bmaintain\w*|maintenance\b', source, re.I)):
            return ['application_maintenance_scope_not_established_by_selected_text']
        return []
    # A fragment must share a substantive subject/property with its supplier
    # passage. Generic management, marketing, invoicing or qualification words
    # cannot establish a different named service. This only rejects; matching
    # words never create a positive finding.
    stop = set('the a an of for to and or as on in at by with is are be must shall should will '
               'provide provided providing supplier provider contractor current incoming condition selected '
               'containing buyer clause scope service services management including includes include '
               'any all our your their those these such part requirements required expected '
               'information appropriate complete comprehensive relevant knowledge firm firms '
               'need needs requested successful proposed statement within it as well request '
               'premise support supports'.split())
    aliases = {'applications':'software', 'application':'software', 'pcs':'pc', 'reports':'report',
               'microsoft':'microsoft', 'm365':'microsoft', 'updates':'update', 'patching':'update',
               'patch':'update', 'licensing':'license', 'licences':'license', 'licenses':'license',
               'purchasing':'sales', 'procurement':'sales', 'consulting':'consultation'}
    words = lambda t: {aliases.get(w, w.rstrip('s')) for w in re.findall(r'\w+', t.casefold()) if w not in stop}
    requested_words = words(target)
    if len(requested_words) <= 1:
        requested_words |= words(enclosing)
    if context and requested_words and not requested_words & words(source):
        return ['requested_subject_not_established_by_selected_text']
    return []


def content_evidence_issues(obligation: dict, check: dict, *, context: dict | None = None) -> list[str]:
    target = obligation['source_text'].casefold()
    selected_text = [e['quote'] for e in check['evidence'] if e['mode'] == 'text']
    text_only = selected_text and len(selected_text) == len(check['evidence'])
    from bidtrace_core.local.relations import source_field_decision
    if text_only and source_field_decision(obligation['source_text'], context or {}, check['evidence']):
        return []  # A bounded affirmative source fact resolves raw model disagreement.
    if (check['assessment'] == 'contradicted' and text_only
            and re.search(_BUSINESS_APP, target, re.I)
            and re.search(r'\b(?:maintenan\w*|updat\w*)\b', target, re.I)
            and _included_software_lifecycle_maintenance(selected_text)):
        # Included broad maintenance is not an opposing commitment for this
        # qualitative duty. Reject a hallucinated contradiction, never promote
        # it to Supported. Actual exclusion/refusal fails the inclusion check.
        return ['software_maintenance_contradiction_not_established_by_selected_text']
    # Contract cancellation/renewal clocks and biographical references cannot
    # establish a hardware-parts delivery promise, or an incompatible one.
    # This rejects absent subject scope only; it never supplies a positive label.
    if (check['assessment'] in {'supported', 'contradicted'} and text_only
            and re.search(r'\b(?:replacement|spare)\s+parts?\b', target)
            and re.search(r'\b(?:days?|hours?|minutes?|SLA|delivery|deliver)\b', target, re.I)
            and not any(re.search(r'\b(?:parts?|components?|spares?)\b|\breplac\w*\b.{0,50}'
                                 r'\b(?:hardware|devices?|equipment|desktops?|servers?)\b', quote, re.I | re.S)
                        for quote in selected_text)):
        return ['parts_delivery_subject_not_established_by_selected_text']
    if (check['assessment'] in {'supported', 'contradicted'} and text_only
            and re.search(r'\b(?:replacement|spare)\s+parts?\b', target)
            and re.search(r'\b(?:days?|hours?|minutes?|SLA|delivery|deliver)\b', target, re.I)):
        # Replacing a whole PC is not a timed parts-delivery commitment. Bind
        # timing to a parts clause, not a helpdesk clock elsewhere in the quote.
        timed_parts = any(
            re.search(r'\b(?:parts?|components?|spares?)\b', clause, re.I)
            and re.search(r'\b(?:next|same)\s+(?:business|working)\s+day\b|'
                          r'\b(?:within|by|in|no later than)\s+(?:\d+|one|two|three|four|five)\s+'
                          r'(?:(?:business|working)\s+)?(?:days?|hours?|minutes?)\b', clause, re.I)
            for clause in _evidence_clauses(selected_text))
        refused_parts = any(re.search(r'\b(?:replacement\s+)?parts?\b.{0,50}\b'
                                      r'(?:not\s+(?:included|provided|supplied|available)|excluded)\b',
                                      clause, re.I) for clause in _evidence_clauses(selected_text))
        if not timed_parts and not refused_parts:
            return ['parts_delivery_deadline_not_established_by_selected_text']
    if text_only and check['assessment'] == 'contradicted':
        full_scope = _full_source_scope_issues(obligation, check, context, selected_text)
        if full_scope:
            return full_scope
    if check['assessment'] == 'supported' and text_only:
        if re.search(r'\bmeetings?\b|\breview\s+(?:calls?|sessions?)\b', target) and not any(
                re.search(r'\bmeetings?\b|\b(?:review|consultation)\s+(?:calls?|sessions?)\b', quote, re.I)
                for quote in selected_text):
            return ['requested_meeting_service_not_established_by_selected_text']
        if re.search(r'\breports?\b', target) and not any(
                re.search(r'\breports?\b|\breporting\b', quote, re.I) for quote in selected_text):
            return ['requested_reporting_service_not_established_by_selected_text']
        enclosing = (context or {}).get('enclosing_span', {}).get('text', target)
        if (re.search(r'\bexecutive\b', target)
                and re.search(r'\breports?\b|\breporting\b', target + ' ' + enclosing, re.I)):
            # A service report or an Executive Summary heading cannot establish
            # executive reporting. Bind the audience/type to the report clause,
            # never to another bullet or an unrelated proposal heading.
            audience = r'(?:executives?|executive[- ]level|senior\s+(?:leadership|management)|board\s+of\s+directors)'
            patterns = [rf'\b{audience}\s+(?:\w+\s+){{0,2}}(?:reports?|reporting)\b',
                        rf'\b(?:reports?|reporting)\s+(?:(?:are\s+)?(?:provided|delivered|prepared)\s+)?'
                        rf'(?:for|to)\s+(?:(?:the|our|your)\s+)?{audience}\b']
            if not any(_positive_clause(clause)
                       and not re.search(r'\b(?:excluded|excludes?)\b', clause, re.I)
                       and any(re.search(pattern, clause, re.I) for pattern in patterns)
                       for clause in _evidence_clauses(selected_text)):
                return ['executive_reporting_scope_not_established_by_selected_text']
        # Product-scoped backup cannot be inferred from a generic backup
        # promise or an optional add-on. These literal camel-case product
        # names come from the selected buyer span, never an answer key.
        scope = target + ' ' + (context or {}).get('enclosing_span', {}).get('text', '')
        if re.search(r'\bback[- ]?ups?\b', scope, re.I):
            products = [name for name in obligation.get('named_features', [])
                        if re.search(r'[a-z][A-Z]', name)]
            included = [quote for quote in selected_text if not re.search(
                r'\boptional\b|\brecommends?\b.{0,80}\b(?:additional|add[- ]on)\b', quote, re.I | re.S)]
            if any(not any(re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', quote, re.I)
                           for quote in included) for name in products):
                return ['included_backup_product_scope_not_established_by_selected_text']
        need_issue = _buyer_need_only_issue(target, selected_text)
        if need_issue:
            return need_issue
        service_issues = _service_scope_issues(obligation['source_text'], selected_text, context)
        if service_issues:
            return service_issues
        full_scope = _full_source_scope_issues(obligation, check, context, selected_text)
        if full_scope:
            return full_scope
        lines = [line.strip() for quote in selected_text for line in quote.splitlines() if line.strip()]
        if _HISTORY.search(target) and not _POINTER_REQUEST.search(target):
            labelled_pointer = any(_HISTORY.search(line) and re.search(r'https?://\S+', line) for line in lines)
            # A history-field URL remains a pointer when neighboring selected
            # lines discuss unrelated fields. Those lines cannot supply a CV.
            residual = _HISTORY.sub('', re.sub(r'https?://\S+', '', '\n'.join(lines)))
            history_content_requested = re.search(r'\b(?:work|employment|career)[- ]history\b', target)
            if (labelled_pointer or history_content_requested) and not _BIOGRAPHICAL_CONTENT.search(residual):
                return ['work_history_content_not_established_by_selected_text']
        if re.search(r'\b(?:continue|retain|keep)\b.{0,55}\bexisting\b', target):
            clauses = [clause for quote in selected_text for clause in re.split(r'(?<=[.!?;])\s+|\n+', quote)]
            blanket_promise = any(re.search(r'\b(?:will|shall)\b.{0,50}\b(?:all|every)\b.{0,30}\b(?:existing|current)\b', clause, re.I)
                                  for clause in clauses)
            for feature in obligation.get('named_features', []):
                mentions = [clause for clause in clauses if re.search(r'(?<!\w)' + re.escape(feature) + r'(?!\w)', clause, re.I)]
                if (mentions and not blanket_promise and all(re.search(
                        r'^\s*(?:current(?:[- ]environment)?|existing(?:[- ]environment)?)\s+inventory\s*:', clause, re.I)
                        and not re.search(r'\b(?:will|shall|promise|commit|continue|retain|keep)\b', clause, re.I)
                        for clause in mentions)):
                    return ['current_inventory_does_not_establish_supplier_continuation']
    # An acknowledgement can prove that declaration itself, but cannot stand in
    # for the capabilities/actions contained in the document it refers to.
    if check['assessment'] == 'supported' and not re.search(r'\backnowledg(?:e|es|ed|ement|ment|ing)\b', target):
        selected = [e['quote'] for e in check['evidence'] if e['mode'] == 'text']
        if selected and len(selected) == len(check['evidence']) and all(
                re.search(r'\backnowledg\w*\b', quote, re.I)
                and re.search(r'\b(?:RFP|tender|bid|proposal|conditions|requirements)\b', quote, re.I)
                and not re.search(r'\b(?:we\s+(?:provide|perform|deliver|maintain)|our\s+(?:service|system))\b', quote, re.I)
                for quote in selected):
            return ['generic_acknowledgement_does_not_establish_requested_condition']
    if (check['assessment'] != 'supported'
            or not (_HISTORY.search(target) or re.search(r'\b(?:manuals?|reports?|certificates?)\b', target))
            or _POINTER_REQUEST.search(target)):
        return []
    if any(e['mode'] == 'image' for e in check['evidence']):
        return []  # Caller has already validated selected image provenance.
    lines = [line.strip() for e in check['evidence'] for line in e['quote'].splitlines() if line.strip()]

    def pointer_or_heading(line):
        if line.endswith(':') and len(line) <= 160 and not re.search(r'\d', line):
            return True
        if re.fullmatch(r'(?:please )?(?:refer to|see|visit|download|follow)\b.{0,150}', line, re.I):
            return True
        if not re.search(r'https?://\S+', line):
            return False
        remainder = re.sub(r'https?://\S+', '', line).strip(' |,:;.-')
        if not remainder:
            return True
        if re.search(r'\b(?:years?|experience|worked|led|delivered|built|degree|education|university)\b', remainder, re.I):
            return False
        # A short name/title/URL entry is a pointer, not the linked resume.
        return '|' in remainder and all(len(p.split()) <= 8 and not re.search(r'\d', p)
                                       for p in remainder.split('|'))

    if lines and all(pointer_or_heading(line) for line in lines):
        return ['document_content_not_established_by_selected_text']
    return []
