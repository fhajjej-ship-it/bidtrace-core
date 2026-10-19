<!-- SPDX-License-Identifier: MPL-2.0 -->
# BidTrace Core

A local tender-analysis engine and CLI. It compares buyer requirements with
supplier evidence and records source quotations, page references and review
history. Its findings are proposals for a human reviewer, not compliance
certification. Insufficient evidence remains **Unknown**.

**Developer alpha · 0.1.4 · MPL-2.0.** The separate graphical web workspace and
paid Mac app are proprietary and excluded. This source tree includes no model
weights, customer documents, private benchmark answers or Codex adapter.

**Want the maintained Mac app? [Request a paid Mac pilot](#request-a-paid-mac-pilot).**

## Start here: supported hardware and models

This developer CLI currently supports **our pinned local model stack on an
Apple Silicon Mac**. `--model-url` connects to BidTrace's own MLX service; it is
not an OpenAI-compatible endpoint. OpenAI API keys, ChatGPT subscriptions, Ollama
and arbitrary replacement models are not supported by this release.

| Requirement | Supported / tested setup |
| --- | --- |
| Computer | Apple Silicon Mac (`uname -m` prints `arm64`) |
| Python | Python **3.12**, with `venv` and `pip`; Git must also be installed |
| macOS and memory | Verified on macOS 27.0.1 with 24 GiB unified memory. This is a tested configuration, not an established minimum. MLX gets a 10 GiB allocator cap; Python, retrieval and macOS need additional memory. |
| Other platforms | Full assessment on Intel Macs, Windows and Linux is not supported by this CLI profile. Package metadata permits Python 3.13, but it is not release-tested. |
| Disk and network | About 2.1 GB of pinned model downloads, plus dependency environments and workspaces. Initial setup needs internet; document analysis uses offline local models. |

| Role | Model prepared by the setup command |
| --- | --- |
| Assessment | `mlx-community/Qwen3.5-2B-MLX-4bit` |
| Text retrieval | BGE-small, using `qdrant/bge-small-en-v1.5-onnx-q` |
| Evidence checking | `cross-encoder/nli-deberta-v3-xsmall` |

Exact revisions, hashes and separate upstream licenses are in
[MODEL-NOTICES.md](MODEL-NOTICES.md) and [tools/models.json](tools/models.json).
Weights are downloaded separately and are not bundled in the source or wheel.
No paid model account or API key is needed for this local setup.

## Copy-and-paste quickstart

Use **two Terminal windows**, with zsh or bash. Install Git and Python 3.12 first;
check `git --version`, `python3.12 --version` and `uname -m` before continuing.
Choose a **new** `BIDTRACE_HOME` directory. If the example path already exists,
change it in both windows; these commands intentionally refuse to replace
existing models, workspaces or reports. Stop and inspect any command that fails.

### 1. Terminal A: install and prepare models

This installs tagged core **0.1.4**. Keep the two environments separate: analysis
uses Transformers 4.57 and the MLX service uses Transformers 5. The supplied
constraints pin the dependency versions verified on this Mac.

```sh
export BIDTRACE_HOME="$HOME/BidTrace-quickstart"
mkdir "$BIDTRACE_HOME"
git clone --branch v0.1.4 --depth 1 https://github.com/fhajjej-ship-it/bidtrace-core.git "$BIDTRACE_HOME/core"
cd "$BIDTRACE_HOME/core"
python3.12 -m venv .venv-analysis
python3.12 -m venv .venv-model
.venv-analysis/bin/python -m pip install -c requirements-analysis-macos-py312.txt './packages/bidtrace-core[analysis]'
.venv-model/bin/python -m pip install -c requirements-model-macos-py312.txt './packages/bidtrace-core[model-service]'
.venv-analysis/bin/python -m pip check
.venv-model/bin/python -m pip check
.venv-analysis/bin/bidtrace-core --help
.venv-analysis/bin/python tools/prepare_models.py "$BIDTRACE_HOME/models"
.venv-analysis/bin/python tools/prepare_models.py "$BIDTRACE_HOME/models" --verify-only
```

Wait for both `pip check` commands to report **No broken requirements found**
and model verification to report **`"verified": true`** and **`"models": 3`**.
Setup downloads pinned public artifacts and checks their sizes and SHA-256 hashes;
it sends no documents or inference requests. An interrupted download leaves an
incomplete directory for inspection; choose a new destination to retry.

### 2. Terminal A: start the local model service

Run this in the same window, from the cloned `core` directory, and leave it open.
The service runs in the foreground; **Ctrl-C in this window** stops this service.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
BIDTRACE_MLX_MODEL_PATH="$BIDTRACE_HOME/models/qwen" \
BIDTRACE_MLX_ASSESSMENT_STRATEGY=condition_nli \
BIDTRACE_MLX_MEMORY_LIMIT_GIB=10 \
.venv-model/bin/python -m uvicorn bidtrace_v2.mlx_server:app \
  --host 127.0.0.1 --port 8375 --workers 1
```

### 3. Terminal B: check the connection

Set `BIDTRACE_HOME` to the **same directory** you chose in Terminal A. This check
uses a short timeout and verifies the local service, strategy and expected model
identity. It does not perform a full assessment.

```sh
export BIDTRACE_HOME="$HOME/BidTrace-quickstart"
cd "$BIDTRACE_HOME/core"
.venv-analysis/bin/python - <<'PYTHON'
import json, sys, urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8375/health", timeout=5) as response:
        health = json.load(response)
    assert health["provider"] == "mlx" and health["hosted_inference"] is False, "Expected BidTrace local MLX service"
    assert health["assessment_strategy"] == "condition_nli", "Expected condition_nli strategy"
    assert any(m["id"] == "mlx-community/Qwen3.5-2B-MLX-4bit"
               and m["revision"] == "93760be4f1f69842a46bc13dbdc0f19e291392a3"
               and m["weights_ready"] for m in health["models"]), "Expected pinned Qwen weights ready"
except Exception as error:
    sys.exit("Connection check failed: " + str(error) +
             ". Check Terminal A, the model directory and port 8375 before continuing.")
print("Connected: local MLX service; expected Qwen weights ready; condition_nli; no hosted inference.")
PYTHON
```

Expected: **`Connected: local MLX service; expected Qwen weights ready; condition_nli; no hosted inference.`**
The health field `loaded_model` can still be `null`: the model loads lazily when
the first assessment starts. BGE and NLI readiness was checked by model
verification in step 1; the example below verifies actual inference end to end.

### 4. Terminal B: analyze and export the synthetic example

These are fictional documents, not customer material. The supplier explicitly
offers the stated helpdesk hours and mentions backup monitoring, but does not
state a cloud recovery point objective (RPO). Monitoring is not that commitment.
Run this only after the connection check passes, from the same `core` directory.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv-analysis/bin/bidtrace-core analyze \
  --workspace "$BIDTRACE_HOME/example-1" \
  --buyer examples/buyer.txt --supplier examples/supplier.txt \
  --model-url http://127.0.0.1:8375 \
  --embedding-cache "$BIDTRACE_HOME/models/embedding-cache" \
  --nli-model "$BIDTRACE_HOME/models/nli" --title 'Synthetic developer example'

.venv-analysis/bin/bidtrace-core export \
  --workspace "$BIDTRACE_HOME/example-1" --format json \
  --output "$BIDTRACE_HOME/example-1.json"
.venv-analysis/bin/bidtrace-core export \
  --workspace "$BIDTRACE_HOME/example-1" --format csv \
  --output "$BIDTRACE_HOME/example-1.csv"
```

## Expected example result

Successful analysis exits with status **0** and prints a JSON summary containing
`"status": "completed"` and `"hosted_inference": false`, plus generated project/run
IDs and the workspace path. IDs and timing vary. The exports are
`$BIDTRACE_HOME/example-1.json` and `$BIDTRACE_HOME/example-1.csv`.

| Buyer duty | Expected obligation-level assessment | Why |
| --- | --- | --- |
| Helpdesk Monday–Friday, 09:00–17:00 | `supported` | The supplier explicitly offers those hours. |
| Cloud backup RPO of 24 hours or less | `unknown` | The supplier does not state a cloud RPO. Daily backup monitoring alone does not establish one. |

Inspect the JSON's `detail.requirements[].proposal.obligation_checks`, including
its source quotations and page references. These are **per-obligation** results;
headings or compound rows can have different overall statuses. A completed
command is not proof that every finding is correct. Human review is required.

Review exact buyer scope and supplier quotations before acting. Analysis refuses
an existing workspace; export refuses an existing report. For another run use
new workspace/output names. Keep the entire workspace for backup: JSON alone
does not include immutable source files. This minimal CLI has no resume command
or amendment/review UI.

## If setup or the connection check fails

- **`python3.12: command not found` or `uname -m` is not `arm64`:** check the
  prerequisites and supported hardware above before installing dependencies.
- **Connection refused / timed out:** keep Terminal A running, inspect its error
  and confirm it uses port 8375. If that port belongs to another service, choose
  an unused port and update the service, connection-check URL and `--model-url`
  together. Do not stop unrelated applications to free a port.
- **Weights not ready / wrong model:** check `BIDTRACE_HOME` in both windows and
  rerun step 1's `--verify-only` command. Use the pinned preparation tool; an
  arbitrary model folder does not satisfy the model/revision checks.
- **Dependency conflicts:** use separate fresh environments with the supplied
  constraints; do not combine the two extras or copy the commercial app runtime.
- **Metal or memory errors:** inspect Terminal A, confirm compatible MLX/macOS and
  available memory, and close only applications you choose. The allocator cap
  does not limit the entire computer's memory use or guarantee speed.
- **Existing directory / report:** choose a new path. A failed or interrupted run
  retains files for inspection; do not overwrite another workspace to retry.

Export needs no running model service or model weights. Developer tests can be
run separately after installing the `test` extra. To build a wheel, use
`python -m pip wheel --no-deps ./packages/bidtrace-core`.

Quickstart verification on 19 October 2026 used fresh registry dependency
installs with the listed constraints and actual downloads/hash checks of all
3 models. The connection check failed clearly with the service stopped and
passed when running. The synthetic analysis plus JSON/CSV exports completed in
15.423 seconds with the two obligation-level results above and valid citations.
Initial installation and model downloads are excluded from that example timing;
this is a smoke test on the tested Mac, not an accuracy or speed guarantee.

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

## Request a paid Mac pilot

Small IT suppliers and bid consultancies can enquire about an **assisted paid
pilot of the proprietary BidTrace Mac app**, with a graphical local review
workspace. The open-source core remains free under MPL-2.0.

**[Message Farouk on X (@Faroucky_ha) to request a paid Mac pilot](https://x.com/Faroucky_ha).**
Start your message with **“BidTrace Mac pilot”** and include your role/company,
a short description of the tender-review workflow you want to test, and your
Mac model, memory and macOS version.

Pilot fit, scope and pricing are discussed and agreed separately; an enquiry
does not commit you to payment. Keep your initial message high-level: do not
attach confidential tender documents, supplier proposals or credentials to
X messages or public GitHub issues. The Mac app and installer are not included
in this repository or its release downloads.

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
