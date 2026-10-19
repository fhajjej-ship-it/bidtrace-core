<!-- SPDX-License-Identifier: MPL-2.0 -->
# Core dependencies and separate rights

The selected original BidTrace core files are licensed under MPL-2.0; see
LICENSES/MPL-2.0.txt. Existing third-party attribution must be retained.

This source/wheel does not contain dependency installations or model weights.
Its declared dependencies and separately installed models retain their own
licenses. A downstream bundle must preserve the notices and source obligations
of the versions it actually includes; this document is not a replacement for
those texts or permission to relicense dependencies.

The verified local profile uses Qwen3.5-2B MLX 4-bit (Apache-2.0), DeBERTa NLI
(conversion Apache-2.0; original Microsoft MIT), and BGE-small ONNX
(conversion Apache-2.0; original BGE MIT). Document parsing uses pypdf and
pypdfium2/PDFium under their respective BSD/Apache and transitive terms.
Python, MLX, Torch, ONNX and other runtime dependencies keep their own notices.

The commercial Mac bundle retains its exact component license texts, model
attribution, inventory and required upstream source archives under
Contents/Resources/ThirdPartyNotices. No model weights, frontend/native app,
private data, evaluation labels or Codex adapter form part of this core release.
