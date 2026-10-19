# SPDX-License-Identifier: MPL-2.0
"""Conservative source routing; these rules do not assess supplier compliance.

Buyer background is retained as buyer context. Delivery and bid-process clauses
need an operational artifact or owner confirmation, not capability inference.
Unrecognized, mixed, or unresolved source passages stay in the assessment queue.
Rules use source grammar only, never document names, IDs, or reference labels.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import re


ROUTING_VERSION = "source_role_routing_v7_buyer_flexibility"
SUPPLIER_INDEX_VERSION = "source_observations_v1"
_LABEL = re.compile(r"^\s*(?:[QA][- ]?\d+(?:\.\d+)*(?:[.:]\s*[a-z][.)]?)?[.:]?\s*)?(?:\(\*\)\s*)?", re.I)
_NORMATIVE = re.compile(
    r"\b(?:must|shall|should|require[sd]?|requirements?|necessary|mandatory|preferred|"
    r"expect(?:s|ed|ations?)?|needs?\s+to|have\s+to|has\s+to|will|would|could|"
    r"incoming|successful|selected|new\s+(?:provider|supplier|contractor|MSP)|"
    r"retain(?:ed|s|ing)?|continu(?:e[sd]?|ing)|maintain(?:ed|s|ing)?|"
    r"replac(?:e[sd]?|ing)|upgrad(?:e[sd]?|ing)|transition|transfer|"
    r"requests?|seeks?|intend(?:s|ed)?|plans?|planned|slated|pending)\b", re.I)
_QUESTION_REQUIREMENT = re.compile(
    r"\b(?:require[sd]?|requirements?|necessary|expect(?:s|ed|ations?)?|"
    r"would\s+(?:like|need)|will\s+(?:you|we)|shall|must|should|"
    r"new\s+(?:provider|supplier|contractor|MSP)|incoming)\b", re.I)
_CURRENT_QUESTION = re.compile(r"\b(?:current(?:ly)?|existing|age|make\s+and\s+model|"
    r"license\s+expirations?|under\s+warranty)\b", re.I)
_BUYER = r"(?:the\s+)?(?:buyer|client|authority|council|town|city|organization|organisation)(?:\s+of\s+[\w'-]+){0,4}"
_BUYER_BACKGROUND = re.compile(rf"^{_BUYER}\s+(?:currently\s+)?(?:"
    r"is\s+(?:a|an)\s+|has\s+|owns\s+|operates\s+|uses\s+|utilizes\s+|"
    r"reviews\s+|does\s+not\s+divulge\s+)", re.I)
_CURRENT_DESCRIPTION = re.compile(r"\b(?:currently\s+(?:running|using|under|has|have)|"
    r"(?:is|are)\s+(?:running|subscribed|located|installed|hosted|hybrid\s+joined|covered)|"
    r"(?:was|were)\s+(?:purchased|installed|migrated|migrated/upgraded)|"
    r"(?:is|are)\s+not\s+(?:implemented|installed)|expires?|auto-renew)\b", re.I)
_GOVERNING_REQUIREMENT = re.compile(r"\b(?:scope\s+of\s+(?:work|services)|"
    r"(?:must|shall|should)\s+(?:include|provide)|(?:services|requirements)\s+include|"
    r"following\s+(?:services|requirements))\b", re.I)
_COMMUNITY_SERVICES = re.compile(
    r"^(?P<subject>[\w' -]+?)\s+provides\s+(?:the\s+)?(?:following\s+)?services\s+"
    r"to\s+(?:the\s+)?(?:community[’']s\s+)?residents\s*:", re.I)
_BLANK_FORM_FIELD = re.compile(r"^[\w ,/()-]+:\s*[_ .…]+$")
_PROCUREMENT_DISCLAIMER = re.compile(
    rf"^(?:Furthermore,\s*)?(?:This|the)\s+(?:RFP|RFQ|tender|solicitation|request\s+for\s+proposals)\s+"
    rf"does\s+not\s+obligate\s+{_BUYER}\s+to\s+(?:pay|accept|contract)\b", re.I)
_BUYER_POLICY_DESCRIPTION = re.compile(
    rf"^(?:It\s+is\s+{_BUYER}[’']s\s+policy\s+to\s+|"
    rf"{_BUYER}\s+is\s+committed\s+to\s+(?:a\s+)?program\s+of\s+equal\s+employment\s+opportunity\b)", re.I)
_SUPPLIER_DUTY = re.compile(
    r"\b(?:supplier|provider|contractor|vendor|consultant|bidder|respondent)s?\s+"
    r"(?:(?:is|are)\s+)?(?:must|shall|should|will|required|responsible|agrees?|needs?\s+to)\b", re.I)
_DELIVERY_RESPONSIBILITY = re.compile(
    r"\b(?:submitter|bidder|respondent)s?\b.{0,90}\bensure\s+delivery\b", re.I)
_ADDENDUM_PUBLICATION = re.compile(
    r"\baddend(?:a|um)\b.{0,200}\b(?:posted|published|issued)\b|"
    r"\baddend(?:a|um)\b.{0,200}\bissued\b.{0,200}\bposted\b", re.I)
_REFERENCE_POINTER = re.compile(r"^(?:Please\s+)?(?:see|refer\s+to)\s+[QA][- ]?\d+(?:\.\w+)*[.]?$", re.I)
_INSURANCE_ARTIFACT = re.compile(
    r"\b(?:consultant|contractor|supplier|provider|firm)\s+(?:shall|must|will)\b.{0,90}"
    r"\b(?:procure|maintain)\b.{0,90}\binsurance\b|"
    r"\bcost\s+of\s+(?:such\s+|the\s+)?insurance\b.{0,90}\bpaid\s+by\s+(?:the\s+)?"
    r"(?:consultant|contractor|supplier|provider|firm)\b", re.I)
_PUBLICITY_CONSENT = re.compile(
    r"\bmake\s+no\s+reference\s+to\b.{0,150}\b(?:literature|promotional|brochures|sales\s+presentation)\b", re.I)

# These patterns require a process subject; "submit an incident report within
# an hour" is a service obligation, not a tender-delivery rule.
_SUBMISSION = re.compile(
    r"\b(?:sealed\s+)?(?:proposals?|bids?|submittals?|tenders?)\s+"
    r"(?:will\s+be\s+|must\s+be\s+|shall\s+be\s+|are\s+|be\s+)?"
    r"(?:received|accepted|delivered|submitted)\b", re.I)
_SUBMISSION_WHEN = re.compile(r"\b(?:due\s+date|deadline|late\s+delivery|"
    r"on\s+or\s+before|no\s+later\s+than|until|after|before|by\s+\d)\b", re.I)
_DEADLINE_FIELD = re.compile(r"^(?:due\s+date|(?:bid|proposal|tender|submittal)\s+deadline)\s*:", re.I)
_QUESTION_PROCESS = re.compile(r"\bquestions\s+(?:from\s+(?:potential\s+)?(?:bidders?|"
    r"suppliers?|respondents?)\s+)?(?:are\s+due|must\s+be\s+(?:submitted|sent)|"
    r"shall\s+be\s+(?:submitted|sent))\b", re.I)
_BUYER_RIGHT = re.compile(rf"^{_BUYER}\s+(?:further\s+)?reserves?\s+the\s+right\s+to\s+"
    r"(?:accept|reject|waive|make\s+an\s+award|negotiate)\b", re.I)
_BIDDER_SERVICE = re.compile(r"\b(?:supplier|contractor|provider|vendor|consultant)\s+"
    r"(?:must|shall|should|will)\b|\b(?:provide|deliver|perform|maintain|retain)\s+"
    r"(?:the\s+)?(?:services?|support|backups?|reports?|certificates?)\b", re.I)
_CONTENT = re.compile(r"\b(?:include|contain|attach|enclose|resume|résumé|certificates?|"
    r"signature|signed|qualifications?|references|insurance|licenses?|licences?)\b", re.I)
_MIXED_RIGHT = re.compile(r"\b(?:must|shall|should|require[sd]?|unless|provided\s+that|"
    r"on\s+condition|subject\s+to)\b", re.I)
_FILING = re.compile(r"^(?:by\s+)?filing\s+(?:with|at|to)\b[^.!?]*\buntil\s*:\s*$", re.I)
_DATE_FIELD = re.compile(r"^(?:date|time)\s*:", re.I)
_ACCESS_INSTRUCTION = re.compile(r"\b(?:visit|consult|check|refer\s+to|see)\b.{0,220}"
    r"\b(?:website|web\s+site|deadlines\s+and\s+important\s+dates)\b", re.I)
_PROCUREMENT = re.compile(r"\b(?:RFP|RFQ|ITT|RFI|proposals?|tenders?|bidding|submission|submittals?)\b", re.I)
_SUBMISSION_RISK = re.compile(r"\b(?:submitters?|bidders?|respondents?)\s+accept\s+"
    r"(?:all\s+)?risks?\s+of\s+late\s+delivery\b", re.I)
_SOLICITATION = re.compile(rf"^{_BUYER}\s+is\s+(?:seeking|soliciting)\s+(?:bids|proposals|tenders)\s+for\b", re.I)
_RECEIPT_NOTICE = re.compile(r"^Notice\s+is\s+hereby\s+given\s+that\s+"
    r"(?:proposals|bids|tenders)\s+will\s+be\s+received\b", re.I)
_PURE_POLICY_MARKER = re.compile(r"^EEO[.]?$", re.I)

# Whole-block source roles: lexical fragments of a definition or publication
# notice cannot be assessed as supplier commitments. Keep the entire block as
# route evidence, and reject mixed blocks rather than dropping their duties.
_DEFINITION_ENTRY = re.compile(r"^[\w][\w /().&'-]{0,69}\s+[–—-]\s+\S")
_INFORMATION_POINTER = re.compile(
    r"See\s+(?:Appendix|Annex|Attachment|Schedule)\s+[^.!?\n]+\s+for\s+"
    r"(?:additional|further|more)\s+information[.]?", re.I)
_DIRECT_DUTY = re.compile(
    r"(?:^|[.;:]\s*|\n\s*)(?:please\s+)?(?:provide|perform|maintain|retain|"
    r"submit|include|attach|supply|deliver|ensure|install|monitor|support|configure|"
    r"visit|consult|check|review)\b|"
    r"\b(?:supplier|provider|contractor|vendor|bidder)\s+(?:provides?|performs?|"
    r"maintains?|retains?|submits?|supplies|delivers?|ensures?|installs?|monitors?|"
    r"supports?|configures?)\b", re.I)
_PUBLICATION = re.compile(
    r"^(?:The\s+)?(?:referenced\s+)?(?:Appendix|Annex|Attachment|Document|Schedule)\b"
    r"[^.!?]*?\b(?:was|were|has\s+been|have\s+been|is|are)\s+"
    r"(?:uploaded|published|posted|available)\b", re.I)
_PERMISSIVE_ACCESS = re.compile(
    r"You\s+can\s+access\s+(?:this|the)\s+(?:document|appendix|annex|attachment|schedule)\s+"
    r"(?:directly\s+)?(?:by\s+visiting|at|via)\s+\S+[.]?", re.I)
_PUBLICATION_SERVICE = re.compile(
    r"\b(?:backups?|monitoring|maintenance|configuration|installation|"
    r"provid(?:e[sd]?|ing)|perform(?:s|ed|ing)?|deliver(?:s|ed|ing)?|"
    r"support\s+(?:is|are|for)|services?\s+(?:is|are|include))\b", re.I)


def _informational_block_rule(text, question_text):
    """Recognize two bounded, wholly informational source grammars.

    This does not resolve the referenced appendix or any supplier capability.
    Normative paired questions still prevent routing an answer as background.
    """
    block = re.sub(r"^\s*[QA][- ]?\d+(?:\.\d+)*[.:]\s*", "", text,
                   count=1, flags=re.I).strip()
    if _QUESTION_REQUIREMENT.search(question_text):
        return None
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if lines and _INFORMATION_POINTER.fullmatch(lines[-1]):
        lines = lines[:-1]
    entries = []
    for line in lines:
        if _DEFINITION_ENTRY.match(line):
            entries.append(re.split(r"\s+[–—-]\s+", line, maxsplit=1)[1])
        elif (entries and not re.search(r"[.!?]\s*$", entries[-1])
              and (line[:1].islower() or entries[-1].count("(") > entries[-1].count(")"))):
            entries[-1] += "\n" + line
        else:
            break
    else:
        if (len(entries) >= 2 and not _NORMATIVE.search("\n".join(lines))
                and all(not re.match(r"(?:all|each|every|we|you|our|the\s+(?:supplier|provider))\b", body, re.I)
                        for body in entries)
                and all(not _DIRECT_DUTY.search(body) for body in entries)):
            return "pure_definition_block"

    plain = " ".join(block.split())
    # An optional final access sentence must itself be a permissive URL pointer.
    access = re.search(r"\bYou\s+can\s+access\b", plain, re.I)
    notice = plain[:access.start()].strip() if access else plain
    if access and not _PERMISSIVE_ACCESS.fullmatch(plain[access.start():]):
        return None
    if not _PUBLICATION.search(notice):
        return None
    # Remove URL tokens solely for the duty check; exact original text is always
    # retained in the source-role receipt (e.g. request-for-proposals URL paths).
    prose = re.sub(r"\S*(?:https?://|www\.|\.[a-z]{2,}/)\S*", "", plain, flags=re.I)
    if (_NORMATIVE.search(prose) or _DIRECT_DUTY.search(prose)
            or _PUBLICATION_SERVICE.search(prose)):
        return None
    # More than one notice sentence can introduce an unrelated service claim.
    # A decimal date is harmless; other sentences require manual assessment.
    if re.search(r"[.!?]\s+[A-Z]", notice.rstrip(".!?")):
        return None
    return "document_publication_or_access_notice"


def _plain(text):
    text = _LABEL.sub("", text, count=1)
    text = re.sub(r"^\s*[•\uf0b7▪▫*-]\s+", "", text, count=1)
    section = re.match(r"^\s*([A-Z][A-Z0-9 /&()-]{1,59})[ \t]*\n[ \t]*(?:[•\uf0b7▪▫*-]\s+)?", text)
    if section and not _NORMATIVE.search(section[1]) and not _GOVERNING_REQUIREMENT.search(section[1]):
        text = text[section.end():]
    text = re.sub(r"^\s*ABOUT\s+[^:\n]+:\s*", "", text, count=1, flags=re.I)
    heading = re.match(r"^\s*([A-Z][A-Z0-9 /&()-]{1,59}):\s+", text)
    if heading and not _NORMATIVE.search(heading[1]) and not _GOVERNING_REQUIREMENT.search(heading[1]):
        text = text[heading.end():]
    return " ".join(text.split())


def _bound_span(text, start, end):
    if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int)
            or isinstance(end, bool) or not 0 <= start < end <= len(text)):
        raise ValueError("Routing requires valid exact source offsets")
    return {"text": text[start:end], "source_start": start, "source_end": end}


def _review_source_role(passage, selected, question):
    """Bounded source grammars found in the full-pack audit, without verdicts.

    Metadata has no independent compliance meaning. Future/project and legal
    conditions remain assessable Unknown until their specific artifact exists.
    Never route a service list by its heading or by a sibling's buyer facts.
    """
    leaf = _plain(selected).strip(' ,.;:')
    # Only the exact descriptive fragment is background. A neighboring term
    # preference or an explicit duty to negotiate remains independently assessed.
    if re.fullmatch(r'(?:The )?(?:terms of the contract|contract terms|terms) (?:are|is) negotiable', leaf, re.I):
        return 'buyer_context', 'buyer_contract_flexibility'
    if re.fullmatch(r'The following is a direct excerpt from (?:the|this) (?:RFP|tender|buyer) document', leaf, re.I):
        return 'buyer_context', 'buyer_document_introduction'
    if re.search(r'^\s*Provided by current (?:MSP|provider|contractor)\s+and?\s*$', _plain(selected), re.I):
        return 'buyer_context', 'current_provider_description_fragment'
    if (selected.rstrip().endswith(':') and re.search(
            r'\b(?:following (?:areas|qualifications)|to include|will require the firm|'
            r'special backup requirement|client portal will provide)\b', leaf, re.I)):
        return 'buyer_context', 'governing_list_introduction'
    if (not leaf and selected.strip() == '(*)') or re.fullmatch(r'(?:\(\*\)|including|etc|and|or|Additionally|therefore|For clarity|'
                    r'General Information|SCOPE OF (?:SERVICES|WORK)|AWARD OF CONTRACT|'
                    r'\d[\d.]* QUESTIONS/RESPONSES|continues next page)', leaf, re.I):
        return 'buyer_context', 'source_heading_or_structural_marker'
    if re.fullmatch(r'See Abbreviations and Acronyms at the end of the Scope of Services for clarification[.]?', passage, re.I):
        return 'buyer_context', 'definition_reading_instruction'
    if re.search(r'^(?:This list is not intended to be a comprehensive itemization|'
                 r'(?:As indicated in the RFP,\s*)?the (?:Town|buyer|client)[’\']s (?:current|existing) IT contractor has '
                 r'(?:received|been provided)|Questions are listed below|Where those responses were provided|'
                 r'If a question was in duplicate)', passage, re.I):
        return 'buyer_context', 'buyer_document_or_incumbent_process_description'
    if re.search(r'^(?:Evaluations will be based on the criteria|The (?:Town|buyer|client) may select a limited number)', passage, re.I):
        return 'buyer_context', 'buyer_evaluation_process'
    if re.search(r'\b(?:equal opportunity requirements|comply with federal, state and local laws|'
                 r'nondiscrimination and affirmative action|appropriate licensure|'
                 r'proprietary information revealed in the proposal|signature form at the end|'
                 r'failure to sign acknowledgement|following selection|'
                 r'final contract|contract within ten|both parties have executed)\b', passage, re.I):
        return 'needs_artifact', 'specific_legal_or_submission_artifact'
    if re.search(r'\b(?:Questions regarding this project must be submitted|Conversations with individual potential bidders|'
                 r'firm selected as the apparently successful|transiƟon period would be considered)\b', passage, re.I):
        return 'needs_artifact', 'procurement_or_future_contract_condition'
    if re.search(r'\b(?:is leased but can be transferred|are slated to be upgraded|'
                 r'will be upgraded during the next)\b', passage, re.I):
        return 'needs_artifact', 'conditional_asset_transfer_or_future_upgrade'
    if re.search(r'^The intent is to have a local provider|^Is this device backing up', passage, re.I):
        return 'needs_artifact', 'physical_provider_or_device_scope_review'
    if re.search(r'^Is this device backing up', _plain(selected), re.I) or re.search(
            r'\bIs there a Backup.{0,65}device in place today\b', passage, re.I):
        return 'needs_artifact', 'physical_provider_or_device_scope_review'
    if (re.search(rf'(?:Please note:\s*)?{_BUYER}\s+is exempt from\b.{{0,50}}\bsales tax\b', passage, re.I)
            and not _SUPPLIER_DUTY.search(passage)):
        return 'buyer_context', 'buyer_tax_exemption_description'
    if (re.search(r'\b(?:current age|current.*OS|existing servers)\b', question, re.I)
            and re.search(r'\bVirtual machines were migrated\s*/\s*upgraded\b', passage, re.I)):
        return 'buyer_context', 'buyer_historical_server_migration'
    if re.fullmatch(r'parts of the successful firm[’\']s proposal[.]?', passage, re.I):
        return 'buyer_context', 'buyer_reserved_award_scope_continuation'
    if re.search(r'^(?:The Town(?: of [\w -]+)? reviews its IT project needs annually|'
                 r'For more information on the genesis of the RFP)', passage, re.I):
        return 'buyer_context', 'buyer_planning_or_process_description'
    if (re.search(r'\b(?:submittals|proposals|oral responses|questions from potential bidders|'
                  r'posting of responses|due date for proposals|aware of any such addenda)\b', passage, re.I)
            and not _ADDENDUM_PUBLICATION.search(passage)
            and re.search(r'\b(?:dates?/times?|non-binding|posting|due date|deadline|addenda|'
                          r'prior to the bid submission date)\b', passage, re.I)):
        return 'needs_artifact', 'procurement_process_artifact'
    if re.search(r'\b(?:being (?:considered|planned)|are being planned|will be upgraded|'
                 r'transition period would be considered|contractual relationship to the|'
                 r'under agreement with the successful contractor)\b', passage, re.I):
        return 'needs_artifact', 'conditional_future_or_contract_scope'
    if re.fullmatch(r'(?:The )?(?:Town|buyer|client)(?: of [\w -]+)? does not have a local preference[.]?', passage, re.I):
        return 'buyer_context', 'buyer_local_preference_policy'
    # A buyer's data-location answer can describe the environment even when a
    # neighboring question asks about retention. It is not a supplier promise.
    if re.search(r'^(?:All other workstations do not store end user|'
                 r'All other workstations are hybrid joined|Primary user folders|'
                 r'all user data is stored)', passage, re.I) and not re.search(
                     r'\b(?:incoming|must|shall|required|will|should)\b', passage, re.I):
        return 'buyer_context', 'buyer_current_data_location'
    return None


def route_obligation(contract: dict, obligation: dict) -> dict | None:
    """Return a source role, never a supported/gap/unknown assessment.

    The complete enclosing span is intentional: when extraction leaves a mixed
    current/future statement or list together, routing abstains rather than using
    only its convenient fragment. Input buyer citations must already be verified
    by the workflow; returned context is copied without inventing page locations.
    """
    text = contract["text"]
    if not isinstance(text, str) or sha256(text.encode()).hexdigest() != contract.get("source_sha256"):
        raise ValueError("Routing source hash does not match the source text")
    selected = _bound_span(text, obligation["source_start"], obligation["source_end"])
    enclosing = _bound_span(text, obligation["context_start"], obligation["context_end"])
    if (selected["text"] != obligation["source_text"]
            or not enclosing["source_start"] <= selected["source_start"] < selected["source_end"] <= enclosing["source_end"]):
        raise ValueError("Routed obligation does not match its enclosing source span")
    # Canonical enclosing spans retain governing list introductions. An unrelated
    # normative paragraph elsewhere must not change this paragraph's source role.
    passage = _plain(enclosing["text"])
    context = contract.get("buyer_context", [])
    questions = [c for c in context if c.get("role") == "current" and c.get("relation") == "paired_question"]
    question_text = " ".join(c.get("quote", "") for c in questions)
    informational_rule = _informational_block_rule(text, question_text)
    issues = obligation.get("decomposition_issues", [])
    # This flag can be caused by the words "additional document/information".
    # A proven informational block may retain it without a supplier assessment;
    # all actual scope conflicts and other review flags still block routing.
    if (contract.get("buyer_context_issues") or
            (issues and (not informational_rule or
                         set(issues) != {"relative_baseline_requires_review"}))):
        return None
    from bidtrace_core.local.context import obligation_context
    focused = obligation_context(contract, obligation)
    extra_role = _review_source_role(_plain(focused['enclosing_span']['text']), selected['text'], question_text)
    if (_plain(selected['text']).startswith('All virtual machines are covered by Extended Support')
            and re.search(r'\b(?:current age|current.*OS|existing servers)\b', question_text, re.I)):
        extra_role = ('buyer_context', 'buyer_current_enterprise_support')
    if (_plain(focused['enclosing_span']['text']).startswith('This includes providing responses to questions')
            and re.search(r'(?:current|existing) IT contractor', text[:selected['source_start']], re.I)):
        extra_role = ('buyer_context', 'incumbent_answer_provision_process')
    if any(re.search(r'\bEVALUATION CRITERIA\b', span['text']) for span in focused['governing_spans']):
        extra_role = ('buyer_context', 'buyer_evaluation_criteria')
    rule = informational_rule or (extra_role[1] if extra_role else None)
    category = "buyer_context" if informational_rule else extra_role[0] if extra_role else None
    governing = []
    if informational_rule:
        governing.append(_bound_span(text, 0, len(text)))
    if category is None and _DATE_FIELD.search(passage):
        # A printed Date field alone might be a contract start date or an
        # artifact-completion field. Only an adjacent delivery introduction can
        # bind it to a bid deadline; retain that exact source as route evidence.
        preceding = text[:enclosing['source_start']].rstrip()
        separators = list(re.finditer(r"\n[ \t]*\n", preceding))
        start = separators[-1].end() if separators else 0
        candidate = preceding[start:].strip()
        if _FILING.fullmatch(" ".join(candidate.split())):
            left = preceding.index(candidate, start)
            governing.append(_bound_span(text, left, left + len(candidate)))
            category, rule = "needs_artifact", "deadline_field_bound_to_delivery_instruction"

    if category is not None:
        pass
    elif _BLANK_FORM_FIELD.fullmatch(passage):
        category, rule = "needs_artifact", "blank_form_field_requires_completion_evidence"
    elif _REFERENCE_POINTER.fullmatch(passage):
        category, rule = "needs_artifact", "buyer_reference_pointer_requires_source_review"
    elif (_INSURANCE_ARTIFACT.search(passage) or
          (_PUBLICITY_CONSENT.search(passage) and re.search(r"\bwritten\s+consent\b", passage, re.I))):
        category, rule = "needs_artifact", "contractual_insurance_or_publicity_artifact"
    elif (_PROCUREMENT.search(passage) and _DELIVERY_RESPONSIBILITY.search(passage)
          and not _CONTENT.search(passage) and not _BIDDER_SERVICE.search(passage)):
        category, rule = "needs_artifact", "bidder_delivery_responsibility"
    elif (_ADDENDUM_PUBLICATION.search(passage) and _PROCUREMENT.search(passage)
          and re.search(r"\b(?:website|web\s+site|bidding\s+site|portal)\b", passage, re.I)
          and not _SUPPLIER_DUTY.search(passage) and not _DIRECT_DUTY.search(passage)):
        category, rule = "buyer_context", "buyer_addendum_publication"
    elif (not _QUESTION_REQUIREMENT.search(question_text) and not _SUPPLIER_DUTY.search(passage)
          and not _DIRECT_DUTY.search(passage)
          and (_PROCUREMENT_DISCLAIMER.search(passage) or _BUYER_POLICY_DESCRIPTION.search(passage))):
        category, rule = "buyer_context", "buyer_policy_or_procurement_disclaimer"
    elif (not _NORMATIVE.search(passage) and not _QUESTION_REQUIREMENT.search(question_text)
          and (community := _COMMUNITY_SERVICES.search(passage))
          and (re.fullmatch(_BUYER, community['subject'], re.I)
               or re.search(r"\b(?:Town|City|Council)\s+of\s+" +
                            re.escape(community['subject']) + r"\b",
                            text[:enclosing['source_start']], re.I))):
        # "Following services" alone could introduce a supplier scope list.
        # This bounded grammar instead identifies services the named buyer
        # already provides to residents. A supplier duty in the same enclosing
        # passage or a normative paired question prevents this route.
        category, rule = "buyer_context", "buyer_community_service_description"
    elif (not _CONTENT.search(passage) and not _BIDDER_SERVICE.search(passage)
          and (_FILING.fullmatch(passage) or _SUBMISSION_RISK.search(passage)
               or (_ACCESS_INSTRUCTION.search(passage) and _PROCUREMENT.search(passage)))):
        category, rule = "needs_artifact", "bid_delivery_or_document_access_process"
    elif (not _CONTENT.search(passage) and not _BIDDER_SERVICE.search(passage)
            and ((_SUBMISSION.search(passage) and _SUBMISSION_WHEN.search(passage))
                 or _DEADLINE_FIELD.search(passage) or _QUESTION_PROCESS.search(passage))):
        category, rule = "needs_artifact", "bid_delivery_or_question_process"
    elif _PURE_POLICY_MARKER.fullmatch(passage):
        category, rule = "needs_artifact", "unexpanded_policy_marker_requires_review"
    elif ((_SOLICITATION.search(passage) or _RECEIPT_NOTICE.search(passage))
          and not _MIXED_RIGHT.search(passage) and not _CONTENT.search(passage)
          and not re.search(r"\b(?:including|covers?|provides?|deliver(?:ed|y)?)\b", passage, re.I)):
        category, rule = "buyer_context", "buyer_procurement_notice"
    elif (_BUYER_RIGHT.search(passage) and not _BIDDER_SERVICE.search(passage)
          and not _MIXED_RIGHT.search(passage)):
        category, rule = "buyer_context", "buyer_reserved_procurement_right"
    elif (not _NORMATIVE.search(passage) and not _QUESTION_REQUIREMENT.search(question_text)
            and not _GOVERNING_REQUIREMENT.search(passage)):
        if _BUYER_BACKGROUND.search(passage):
            category, rule = "buyer_context", "explicit_buyer_background"
        elif questions and _CURRENT_QUESTION.search(question_text) and _CURRENT_DESCRIPTION.search(passage):
            category, rule = "buyer_context", "current_state_answer_to_current_state_question"
    if category is None and re.fullmatch(r'This (?:requirement|condition|provision) includes\s*,?', obligation['source_text'].strip(), re.I):
        category, rule = 'buyer_context', 'sentence_lead_in_without_standalone_duty'
    if category is None:
        return None
    reason = ("Buyer background is retained as context; it does not establish supplier compliance."
              if category == "buyer_context" else
              "Confirm this bid-process condition against a delivery artifact or with the bid owner.")
    if rule == "unexpanded_policy_marker_requires_review":
        reason = "Confirm the duty behind this standalone policy abbreviation; a general acknowledgement is not evidence of compliance."
    elif rule == 'sentence_lead_in_without_standalone_duty':
        reason = 'Sentence lead-in retained as context; the following substantive duties are assessed separately.'
    elif rule == 'buyer_reference_pointer_requires_source_review':
        reason = 'Read the referenced buyer source and verify its linked duties; this pointer is not a supplier commitment.'
    elif rule == 'contractual_insurance_or_publicity_artifact':
        reason = 'Verify the insurance, contractual acceptance or written consent artifact with the bid owner; service capabilities do not establish this commitment.'
    elif rule in {'conditional_future_or_contract_scope', 'conditional_asset_transfer_or_future_upgrade',
                  'physical_provider_or_device_scope_review', 'procurement_or_future_contract_condition'}:
        reason = 'Confirm this future, physical or contractual scope with the bid owner; the supplier passages do not establish this condition.'
    return {"obligation_id": obligation["obligation_id"], "route": category, "rule": rule,
        "routing_version": ROUTING_VERSION, "selected_span": selected, "enclosing_span": enclosing,
        "governing_spans": governing,
        "decomposition_issues": deepcopy(issues),
        "source_sha256": contract["source_sha256"], "buyer_context": deepcopy(context),
        "reason": reason}


def route_contract(contract: dict) -> dict:
    """Partition every supplied ID once without deleting or relabeling a clause."""
    routes, remaining, seen = [], [], set()
    for obligation in contract["obligations"]:
        identifier = obligation["obligation_id"]
        if identifier in seen:
            raise ValueError("Routing requires unique obligation IDs")
        seen.add(identifier)
        route = route_obligation(contract, obligation)
        if route is None:
            remaining.append(identifier)
        else:
            routes.append(route)
    return {"routing_version": ROUTING_VERSION, "routes": routes,
            "remaining_obligation_ids": remaining, "source_sha256": contract["source_sha256"]}


_CONTEXT_FLAGS = {
    "optional_or_alternative": re.compile(r"\b(?:optional|options?|alternatives?|recommends?)\b", re.I),
    "conditional": re.compile(r"\b(?:if|unless|subject\s+to|provided\s+that|upon\s+request)\b", re.I),
    "target_or_endeavor": re.compile(r"\b(?:targets?|endeavou?r|best\s+efforts?|aims?)\b", re.I),
    "negation": re.compile(r"\b(?:not|no|never|cannot|unable|without)\b", re.I),
    "historical_or_existing": re.compile(r"\b(?:current(?:ly)?|existing|previous|formerly|since|was|were)\b", re.I),
    "external_pointer": re.compile(r"https?://|\b(?:refer\s+to|please\s+see)\b", re.I),
}


def build_supplier_fact_index(documents: list[dict]) -> dict:
    """Catalogue reusable exact supplier observations once, without interpreting them.

    These are *declared source observations*, not independently verified facts.
    No numeric value, entity, inclusion status, or compliance result is inferred.
    Each paragraph has an exact page anchor; the complete page and conservative
    page-level flags stay available so a quotation cannot erase nearby optional,
    conditional, target, or historical wording. Cross-page scope is unresolved:
    callers must retain the source documents when interpreting an observation.
    Unflagged observations are not assumed unconditional or included.
    """
    pages, observations, identities = [], [], set()
    for document in documents:
        if document.get("role") != "evidence":
            continue
        if not document.get("id"):
            raise ValueError("Supplier observations require a source document ID")
        source_pages = document.get("pages")
        if not isinstance(source_pages, list) or not source_pages:
            raise ValueError("Supplier observations require explicit source pages")
        for page in source_pages:
            number, text = page.get("page"), page.get("text")
            if (not isinstance(number, int) or isinstance(number, bool) or number < 1
                    or not isinstance(text, str)):
                raise ValueError("Supplier observations require a page number and exact text")
            identity = (document["id"], number)
            if identity in identities:
                raise ValueError("Supplier observation source pages must be unique")
            identities.add(identity)
            page_hash = sha256(text.encode()).hexdigest()
            page_id = "P" + sha256((document["id"] + ":" + str(number) + ":" + page_hash).encode()).hexdigest()[:20]
            flags = [name for name, pattern in _CONTEXT_FLAGS.items() if pattern.search(text)]
            common = {"document_id": document["id"], "title": document.get("title", ""),
                      "page": number, "role": "evidence", "version": document.get("version")}
            pages.append({**common, "page_id": page_id, "text": text,
                          "source_sha256": page_hash, "context_flags": flags})
            # Keep all non-whitespace source content, including headings and
            # contact fields. No source clause is silently discarded as metadata.
            for match in re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)", text):
                start, end = match.start(), match.end()
                while end > start and text[end - 1].isspace():
                    end -= 1
                quote = text[start:end]
                observations.append({**common,
                    "fact_id": "F" + sha256((page_id + ":" + str(start) + ":" + str(end)).encode()).hexdigest()[:20],
                    "kind": "declared_source_observation", "quote": quote,
                    "page_id": page_id, "source_start": start, "source_end": end,
                    "source_sha256": page_hash, "context_flags": list(flags)})
    # Sort the source identity for stable cache invalidation independent of input
    # ordering. The observations themselves remain in the supplied source order.
    index_hash = sha256("\n".join(sorted(p["page_id"] + ":" + str(p["version"]) for p in pages)).encode()).hexdigest()
    return {"version": SUPPLIER_INDEX_VERSION, "source_sha256": index_hash,
            "pages": pages, "observations": observations,
            "scope": "Exact supplier text observations; inclusion, entity, value and cross-page scope require interpretation."}
