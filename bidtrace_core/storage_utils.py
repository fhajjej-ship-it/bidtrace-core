# SPDX-License-Identifier: MPL-2.0
"""Shared immutable-source helpers without network-storage dependencies."""
import re
from datetime import date, datetime

def _serialize(row):
    if row is None:
        return None
    return {k: v.isoformat() if isinstance(v, (datetime, date)) else v for k, v in row.items()}


def _source_payload(document):
    # Active pack membership changes; all original source metadata stays immutable.
    return {k: v for k, v in document.items() if k != 'active'}


def validate_object_key(key):
    if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_.\-/]+', key) or any(p in {'', '.', '..'} for p in key.split('/')):
        raise ValueError('Invalid object key.')
    return key

