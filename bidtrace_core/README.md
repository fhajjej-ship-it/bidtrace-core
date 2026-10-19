<!-- SPDX-License-Identifier: MPL-2.0 -->
# BidTrace Core 0.1.4

The MPL-2.0 engine and CLI compare buyer requirements against supplier evidence
with source quotations, page references, embedded storage and review history.
The separate graphical workspace and Mac application remain proprietary.

See the root [README](../README.md) in the source archive for supported setup,
explicit pinned model preparation, analysis and JSON/CSV export instructions.
The source and wheel contain the same 54 runtime modules selected by
`package_manifest.py`. Legacy `bidtrace` and `bidtrace_v2` imports remain for
compatibility. Commercial APIs/UI, the native app, hosted-provider
implementations, the Codex adapter, evaluation labels, weights and user data
are excluded. The dispatcher retains optional hooks, but standalone settings
accept only loopback MLX inference with embedded storage.

Python 3.12 on Apple Silicon is the tested full-analysis setup. Install analysis
and MLX service extras in separate environments. The CLI never downloads models
and never reads application dotenv files. Analysis requires a new workspace;
export reads an existing workspace without replacing existing output files.
JSON export alone is not a backup of immutable source objects. Interrupted work
is retained; this minimal CLI has no resume or amendment/review UI.

See the root release notes for measured regression results and alpha limits.
These are proposed findings for human review, not compliance certification.
