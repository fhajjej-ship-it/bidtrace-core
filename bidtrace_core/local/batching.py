# SPDX-License-Identifier: MPL-2.0
"""Local-only packing; production's three-obligation contract is unchanged."""
from bidtrace_v2.models import ModelFailure
from bidtrace_v2.obligations import decompose_requirement


def _limit(value):
    if type(value) is not int or value not in (3, 6, 8, 12):
        raise ValueError('Local batch limit must be 3, 6, 8 or 12')


def select_local_obligations(requirement, identifiers, batch_limit):
    _limit(batch_limit)
    contract = decompose_requirement(requirement)
    ordered = [o['obligation_id'] for o in contract['obligations']]
    if (not isinstance(identifiers, list) or not 1 <= len(identifiers) <= batch_limit
            or any(not isinstance(i, str) or i not in ordered for i in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or identifiers != [i for i in ordered if i in identifiers]):
        raise ModelFailure('Local batch requires bounded, unique source IDs in canonical order')
    return {**contract, 'obligations': [o for o in contract['obligations']
                                      if o['obligation_id'] in identifiers]}


def plan_batches(requirements, batch_limit=3):
    _limit(batch_limit)
    result = []
    for requirement in requirements:
        contract = decompose_requirement(requirement)
        ids = [o['obligation_id'] for o in contract['obligations']]
        for start in range(0, len(ids), batch_limit):
            result.append({'requirement_id': requirement['id'],
                           'source_sha256': contract['source_sha256'],
                           'obligation_ids': ids[start:start + batch_limit]})
    return result
