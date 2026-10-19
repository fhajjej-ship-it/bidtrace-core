# SPDX-License-Identifier: MPL-2.0
"""Offline reject-only pre-screen; ambiguous pairs still use the local LLM.

Neutral scores are uncalibrated model outputs, never compliance probabilities.
No supported or gap decision is made here. Complete oversized pairs bypass the
classifier; no buyer or supplier text is truncated to fit its context window.
"""
from pathlib import Path
import os
import hashlib
import json
import re
import threading
import time
import uuid

from .condition_assessment import condition_hypothesis
from bidtrace_core.local.content import content_evidence_issues, explain_scope_issues
from bidtrace_core.local.context import obligation_context

MODEL_ID = 'cross-encoder/nli-deberta-v3-xsmall'
REVISION = 'a150876415327c80daeff35ca6f68f5ed8cf5c24'
NEUTRAL_THRESHOLD = .995
GATE_VERSION = 'complete_sources_reject_only_v3_recovery_point_scope'
MODEL_DIR = Path(os.getenv('BIDTRACE_NLI_MODEL_PATH', str(Path(__file__).resolve().parents[1] / 'data-v2/local-models/nli-deberta-v3-xsmall')))
_STOP = set('the a an of for to and or as on in at by with is are be must shall should will provide supplier condition selected containing buyer clause scope'.split())


def content_words(text):
    return {word.rstrip('s') for word in re.findall(r'\w+', text.casefold()) if word not in _STOP}


def reject_neutral(probabilities, lexical_overlap):
    # Preserve plausible literal service matches for actual model assessment,
    # including Office/Microsoft spelling variants. This only abstains.
    return lexical_overlap < 2 and probabilities.get('neutral', 0) >= NEUTRAL_THRESHOLD


class LocalNliGate:
    def __init__(self):
        self.model = self.tokenizer = None
        self.lock = threading.RLock()
        self.runtime_id = uuid.uuid4().hex
        self.calls = []

    def _load(self):
        if self.model is not None:
            return
        manifest = json.loads((MODEL_DIR / 'download-manifest.json').read_text())
        if (manifest['repository'], manifest['revision']) != (MODEL_ID, REVISION):
            raise ValueError('Local NLI model provenance mismatch')
        for item in manifest['files']:
            path = MODEL_DIR / item['file']
            if path.parent != MODEL_DIR or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Local NLI artifact hash mismatch')
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR,
            local_files_only=True, trust_remote_code=False, use_safetensors=True).to('cpu').eval()
        if [self.model.config.id2label[i] for i in range(3)] != ['contradiction','entailment','neutral']:
            raise ValueError('Local NLI label mapping mismatch')

    def screen(self, contract, ranking, *, run_id=None, stage=None):
        with self.lock:
            start = time.perf_counter()
            import torch
            rejected, receipts, tokens = {}, [], 0
            for obligation, condition in zip(contract['obligations'], ranking['conditions']):
                if obligation['obligation_id'] != condition['obligation_id']:
                    raise ValueError('Pre-screen condition identity mismatch')
                evidence = [{'mode': 'text', 'citation_id': c['citation_id'], 'quote': c['quote'],
                             'document_id':c.get('document_id'), 'page':c.get('page')}
                            for c in condition['selected_evidence'][:2]]
                issues = content_evidence_issues(obligation, {'assessment': 'supported', 'evidence': evidence},
                    context=obligation_context(contract, obligation))
                premise = '\n\n'.join(e['quote'] for e in evidence)
                hypothesis = condition_hypothesis(contract, obligation)
                receipt = {'obligation_id': obligation['obligation_id'], 'source_sha256': contract['source_sha256'],
                           'selected_citation_ids': [e['citation_id'] for e in evidence], 'issues': issues}
                if issues:
                    rejected[obligation['obligation_id']] = explain_scope_issues(issues)
                elif not premise:
                    rejected[obligation['obligation_id']] = 'No supplier evidence was selected.'
                else:
                    from bidtrace_core.local.relations import source_field_decision
                    if source_field_decision(obligation['source_text'], obligation_context(contract,obligation), evidence):
                        receipt['bypass'] = 'source_bound_field_fact_requires_grounding'
                        receipt['decision'] = 'llm_required'
                        receipts.append(receipt)
                        continue
                    self._load()
                    overlap = len(content_words(obligation['source_text']) & content_words(premise))
                    receipt['lexical_overlap'] = overlap
                    # A literal match already forces LLM assessment. Avoid work
                    # whose result cannot change the reject-only decision.
                    if overlap >= 2:
                        receipt['bypass'] = 'plausible_literal_match_requires_llm'
                    else:
                        comparisons = []
                        for source in evidence:
                            inputs = self.tokenizer(source['quote'], hypothesis,
                                                    return_tensors='pt', truncation=False)
                            count = inputs['input_ids'].shape[1]
                            comparison = {'citation_id': source['citation_id'], 'tokens': count}
                            if count > self.model.config.max_position_embeddings:
                                comparison['bypass'] = 'complete_pair_exceeds_classifier_context'
                            else:
                                tokens += count
                                with torch.inference_mode():
                                    values = torch.softmax(self.model(**inputs).logits[0], dim=-1).tolist()
                                comparison['probabilities'] = dict(zip(['contradiction','entailment','neutral'], values))
                            comparisons.append(comparison)
                        receipt['source_comparisons'] = comparisons
                        # Keep jointly relevant sources: the combined lexical
                        # shield above applies before splitting any premise.
                        # One oversized or uncertain source forces the LLM.
                        if all(reject_neutral(c.get('probabilities', {}), overlap) for c in comparisons):
                            rejected[obligation['obligation_id']] = 'Local source comparison did not establish this condition; review required.'
                receipt['decision'] = 'unknown' if obligation['obligation_id'] in rejected else 'llm_required'
                receipts.append(receipt)
            call = {'id': len(self.calls)+1, 'runtime_id': self.runtime_id, 'run_id': run_id, 'stage': stage,
                    'method': 'entailment_prescreen', 'kind': 'nli', 'provider': 'mlx', 'model': MODEL_ID,
                    'revision': REVISION, 'device': 'cpu', 'status': 'completed', 'hosted_inference': False,
                    'input_tokens': tokens, 'output_tokens': 0, 'seconds': round(time.perf_counter()-start,3),
                    'gate_version': GATE_VERSION, 'neutral_threshold': NEUTRAL_THRESHOLD,
                    'items': len(receipts), 'obligation_ids': [o['obligation_id'] for o in contract['obligations']],
                    'rejected_count': len(rejected), 'pair_receipts': receipts}
            self.calls.append(call)
            return rejected, call


gate = LocalNliGate()
