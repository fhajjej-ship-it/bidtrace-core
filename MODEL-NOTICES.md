<!-- SPDX-License-Identifier: MPL-2.0 -->
# Separately downloaded models

No model weights are distributed in this source release. Exact repositories,
revisions, filenames and SHA-256 hashes are in [tools/models.json](tools/models.json).
The setup command downloads those public artifacts from Hugging Face.

| Function | Repository | Pinned revision | Upstream rights |
| --- | --- | --- | --- |
| Assessment | mlx-community/Qwen3.5-2B-MLX-4bit | 93760be4f1f69842a46bc13dbdc0f19e291392a3 | Apache-2.0 |
| NLI | cross-encoder/nli-deberta-v3-xsmall | a150876415327c80daeff35ca6f68f5ed8cf5c24 | Apache-2.0 conversion; original Microsoft DeBERTa MIT |
| Text retrieval | qdrant/bge-small-en-v1.5-onnx-q | 52398278842ec682c6f32300af41344b1c0b0bb2 | Apache-2.0 conversion; original BGE MIT |

Retained upstream license texts and cards are in [model-notices/](model-notices/).
Those third-party documents retain their own rights; the core MPL grant does
not relicense them. If redistributing installed dependencies or downloaded
weights, retain their notices and satisfy their respective license obligations.
