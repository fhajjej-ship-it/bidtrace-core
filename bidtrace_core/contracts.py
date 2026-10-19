# SPDX-License-Identifier: MPL-2.0
"""Shared execution contracts without API, storage SDK or provider imports."""

MODELS = {
    "embedding": "Qwen/Qwen3-Embedding-0.6B", "reranker": "Qwen/Qwen3-Reranker-0.6B",
    "visual": "vidore/colqwen2-v1.0", "generation": "Qwen/Qwen3-VL-4B-Instruct",
}
PIPELINE_VERSION = "full-stack-1"
ACTIVE_JOBS = {"queued", "running", "stopping"}


class ImmutableSnapshotError(ValueError):
    """An existing source or audit event cannot be overwritten."""


class LeaseLostError(RuntimeError):
    """Only the current unexpired lease owner may finish a job."""


class JobCancelled(RuntimeError):
    """The current operation may finish, but no further work may start."""


VALIDATION_REASONS = {
    "invalid_json": "the response was not valid JSON",
    "invalid_obligation_schema": "the obligation checks had an invalid structure",
    "obligation_id_coverage": "the response did not check each requested obligation ID exactly once",
    "invalid_assessment": "an assessment label was invalid",
    "invalid_rationale": "an assessment rationale was missing or exceeded its length limit",
    "invalid_citation_ids": "a citation selection was invalid or outside the supplied evidence catalog",
    "missing_required_citation": "a substantive assessment lacked a required evidence citation",
    "citation_source_mismatch": "a selected citation failed source verification",
    "non_supplier_citation": "an obligation check cited a source outside supplier evidence",
    "candidate_count": "the provider did not return exactly one response candidate",
    "candidate_incomplete": "the provider did not finish its response",
    "output_token_limit": "the response hit its token limit before completion (thinking and answer tokens share this limit)",
    "candidate_text_missing": "the response candidate contained no usable text",
    "invalid_reranking": "the evidence ranking was incomplete or invalid",
    "invalid_embedding": "the embedding vector was invalid",
    "malformed_response": "the provider response was malformed",
}
