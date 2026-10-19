<!-- SPDX-License-Identifier: MPL-2.0 -->
# BidTrace Core

A local tender-analysis engine and CLI. It compares buyer requirements with
supplier evidence and records source quotations, page references and review
history. Its findings are proposals for a human reviewer, not compliance
certification. Insufficient evidence remains **Unknown**.

**Developer alpha · 0.1.4 · MPL-2.0.** The separate graphical web workspace and
paid Mac app are proprietary and excluded. This source tree includes no model
weights, customer documents, private benchmark answers or Codex adapter.

## Supported setup

Python 3.12 or 3.13. Full local assessment requires Apple Silicon and compatible
MLX models. This release was installation-tested on macOS with Python 3.12;
other operating systems and Python 3.13 are not release-tested. Export does not
need model weights. The MLX allocator budget is 10 GiB, with additional memory
needed for Python, retrieval, the OS and other applications. Under-five-minute
performance is a measurement on the benchmark Mac, not a hardware guarantee.

## Install from this source tree

Use two separate environments: the analysis runtime uses Transformers 4.57,
while the MLX service uses Transformers 5. Do not combine the extras. From the
root of a fresh source checkout:

```sh
python3.12 -m venv .venv-analysis
python3.12 -m venv .venv-model
.venv-analysis/bin/python -m pip install './packages/bidtrace-core[analysis,test]'
.venv-model/bin/python -m pip install './packages/bidtrace-core[model-service]'
.venv-analysis/bin/python -m pip check
.venv-model/bin/python -m pip check
.venv-analysis/bin/bidtrace-core --help
.venv-analysis/bin/python -m pytest tests_v2 tests -q --import-mode=importlib
```

The dependency sets were independently resolved and installed from the package
registry during earlier release preparation on this Mac. Their pinned version
lists are `requirements-analysis-macos-py312.txt` and
`requirements-model-macos-py312.txt`; use them as constraints (`pip install -c
FILE ...`) for that platform. The lists do not install models or the core
package by themselves. The 0.1.4 release check installs its newly built wheel
in separate environments reusing those dependency sets, with package and import
checks. It does not claim a new full registry installation.

Build a wheel with `python -m pip wheel --no-deps ./packages/bidtrace-core`.
Do not copy the commercial app environment into this checkout.

## Prepare models once

Review the model identities and separate upstream licenses in
[MODEL-NOTICES.md](MODEL-NOTICES.md). The following explicit setup command fetches
only pinned public model artifacts from Hugging Face, verifies their SHA-256
hashes, and creates the BGE cache, MLX provenance and NLI manifest required by the
runtime. It sends no tender documents and performs no inference. It needs an
internet connection and several GB of free disk space; use a new path outside
the checkout. The analysis CLI itself never downloads models.

```sh
.venv-analysis/bin/python tools/prepare_models.py "$HOME/BidTrace-models"
.venv-analysis/bin/python tools/prepare_models.py "$HOME/BidTrace-models" --verify-only
```

On failure, an incomplete new directory is retained for inspection; choose a
new destination to retry. The tool does not overwrite existing model folders.
Set the location in both terminal sessions:

```sh
export BIDTRACE_MODELS="$HOME/BidTrace-models"
```

In the model-service session:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
BIDTRACE_MLX_MODEL_PATH="$BIDTRACE_MODELS/qwen" \
BIDTRACE_MLX_ASSESSMENT_STRATEGY=condition_nli \
BIDTRACE_MLX_MEMORY_LIMIT_GIB=10 \
.venv-model/bin/python -m uvicorn bidtrace_v2.mlx_server:app \
  --host 127.0.0.1 --port 8375 --workers 1
```

## Analyze the fictional example

The examples below are synthetic and explicitly say so. They are not customer
material or benchmark labels. The supplier mentions backup monitoring but does
not commit to a cloud RPO; do not treat monitoring alone as that commitment.
Run from the analysis session, choosing a workspace path that does not exist:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-analysis/bin/bidtrace-core analyze \
  --workspace "$HOME/BidTrace-example-1" \
  --buyer examples/buyer.txt --supplier examples/supplier.txt \
  --model-url http://127.0.0.1:8375 \
  --embedding-cache "$BIDTRACE_MODELS/embedding-cache" \
  --nli-model "$BIDTRACE_MODELS/nli" --title 'Synthetic developer example'

.venv-analysis/bin/bidtrace-core export \
  --workspace "$HOME/BidTrace-example-1" --format json \
  --output "$HOME/BidTrace-example-1.json"
.venv-analysis/bin/bidtrace-core export \
  --workspace "$HOME/BidTrace-example-1" --format csv \
  --output "$HOME/BidTrace-example-1.csv"
```

Review exact buyer scope and supplier quotations before acting on any result.
Analysis refuses an existing workspace; export refuses an existing output file.
Keep the entire workspace for backup: JSON export alone is not a complete
backup of immutable source objects. Interrupted work is retained, but this
minimal CLI does not yet provide a resume command or amendment/review UI.

## Evidence and limits

The core 0.1.4 runtime in the signed native Mac app completed the full Berlin
regression on 19 October 2026: **20/20 primary and 9/12 broader selected-case
agreement**, with visible completed findings within **154.241 seconds** and
954 validated citation occurrences. Genuine evidence gaps stayed Unknown.
These repeatedly tuned checks are regression evidence, not independent unseen
accuracy or proof of time saved for customers. The buyer/supplier PDFs and
expected answers are not part of this release. This timing is from the bundled
Mac app, not a promise for all machines or document packs.

The standalone 0.1.4 wheel is checked separately during release preparation;
see [RELEASE-NOTES.md](RELEASE-NOTES.md) for its measured CLI result. Neither
hosted inference nor the development Codex adapter is included in this release.
The maintained Mac app is a separate commercial product. See
[CHANGELOG.md](CHANGELOG.md) for the source changes in this version.

## Feedback and security

This is an early developer release with no support SLA. Report reproducible
issues using fictional or redacted inputs. Do not upload customer tenders,
credentials or workspace databases to a public issue. The source repository is [fhajjej-ship-it/bidtrace-core](https://github.com/fhajjej-ship-it/bidtrace-core).
For a sensitive vulnerability, use [private vulnerability reporting](https://github.com/fhajjej-ship-it/bidtrace-core/security/advisories/new). Do not put sensitive details in public issues.

See [LICENSE](LICENSE), [LICENSE-BOUNDARY.md](LICENSE-BOUNDARY.md) and
[dependency notices](packages/bidtrace-core/THIRD-PARTY-NOTICES.md). Preserve
third-party notices when redistributing installed dependencies or model weights.

## Known alpha limitation

Some Qdrant versions print an ignored cleanup ImportError at Python shutdown
after a completed CLI analysis. In the release smoke test, the command exited
with status 0 and the saved workspace exported and read back successfully.
Check the exit status and recorded run outcome. This warning is retained as a
known alpha issue; it is not an assessment failure or accuracy result.
Source headings may also remain Unknown review rows; inspect the actual duty
and its per-obligation evidence instead of treating every Unknown as a gap.
