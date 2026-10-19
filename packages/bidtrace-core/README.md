<!-- SPDX-License-Identifier: MPL-2.0 -->
# BidTrace Core 0.1.4

Local tender-analysis engine and CLI, licensed under MPL-2.0. It compares buyer
requirements with supplier evidence, retaining exact quotations, page references,
source versions and review history. Insufficient evidence remains Unknown.
Findings require human review and do not certify compliance.

The wheel includes 54 explicitly selected engine modules and basic analyze/export
commands. It excludes the proprietary web/Mac app, development Codex adapter,
hosted-provider implementations, private data, evaluation labels and model weights.

Python 3.12 on Apple Silicon is the tested MLX setup; Python 3.13 and other
platforms are not release-tested. Use separate environments for `[analysis]` and
`[model-service]`, which need different Transformers versions. Prepare models
explicitly before offline analysis. The CLI itself never downloads weights.

Use the companion source archive's root README, pinned dependency constraints,
`tools/prepare_models.py`, model notices and synthetic example for complete setup.
Run `bidtrace-core --help` or `python -m bidtrace_core --help` after installation.
See `LICENSES/MPL-2.0.txt` and `THIRD-PARTY-NOTICES.md` for separate rights.
