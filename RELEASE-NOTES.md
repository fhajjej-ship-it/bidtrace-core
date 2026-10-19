<!-- SPDX-License-Identifier: MPL-2.0 -->
# BidTrace Core 0.1.4 — developer alpha

Local source-bound tender analysis with a basic CLI. Compare buyer requirements
against supplied evidence, inspect quotations and page references, and export
JSON/CSV. Insufficient evidence remains Unknown. Findings need human review;
this software does not certify compliance.

## Included

- 54 explicitly selected runtime modules, matching the core used by the Mac app.
- Analyze a new local workspace; export saved results without replacing outputs.
- MPL-2.0 source and wheel, license boundary and separate dependency/model notices.
- Explicit pinned model preparation, separate analysis/MLX dependency constraints,
  a clearly labelled fictional example and CLI/setup tests.

No Mac app, web UI, commercial API, Codex adapter, model weights, customer data,
private evaluation inputs, expected answers or launch drafts are included.
The maintained Mac app remains proprietary. No pricing or support commitment
comes with this developer release.

## Verified on the maintainer's Apple Silicon Mac, 19 October 2026

The newly built 0.1.4 wheel ran the complete 4-document, 42-page Berlin pack
through the standalone CLI in **147.374 seconds** (worker
142.339 seconds). This included parsing and cold local model
loading; service startup was excluded. It retained **20/20 primary and 9/12
broader frozen selected-case agreement**, with 954 validated
citation occurrences, 162 local calls, no hosted calls and no reused
assessments. Genuine evidence gaps and both unsupported cloud RPO claims stayed
Unknown; the explicit term shortfall stayed a gap. JSON and CSV exports were
read back and compared against the saved findings.

Separately, the signed native Mac app using the same 54 runtime modules reached
visible completed Berlin findings in 154.241 seconds in its earlier fresh run.
This native timing and the CLI timing are different measurements.

These are tuned regression checks on this Mac, not independent accuracy,
performance guarantees for all machines/tenders, or evidence of customer time
saved. Private source documents and expected labels are intentionally excluded.
Independent professional evaluation remains pending.

All 12 public CLI/setup checks passed. The newly built wheel was installed into
separate core environments reusing the previously registry-installed dependency
sets (89 analysis packages; 57 model-service packages). Import paths, core source
bytes, dependency compatibility and all 3 pinned model artifacts were verified.
This check did not repeat a full registry dependency installation or model download.

## Setup and limitations

Use the source archive README for the complete setup and synthetic example.
The tested profile uses Python 3.12 on Apple Silicon, 2 separate environments,
Qwen3.5-2B MLX 4-bit, BGE-small ONNX and DeBERTa NLI. The 10 GiB MLX allocator
budget excludes the OS, Python and retrieval overhead. Python 3.13 and other
platforms are not release-tested. Explicit initial model setup needs a network
connection; the analysis itself uses local offline models and embedded storage.
Models and dependencies retain separate upstream licenses.

The minimal CLI has no resume command or amendment/review UI. Keep the entire
workspace for backup; JSON export alone does not include immutable source files.
Some local Qdrant versions emit an ignored shutdown cleanup warning after a
successful CLI run. Check the recorded run outcome and command exit status.
Headings can remain Unknown rows; inspect the duty and per-obligation evidence.
Report bugs with fictional or redacted examples; never upload private documents,
credentials or workspace databases to a public issue.

See CHANGELOG.md, LICENSE-BOUNDARY.md, MODEL-NOTICES.md and SECURITY.md.
