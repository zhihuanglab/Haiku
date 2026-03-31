# Example: Cross-Modal Retrieval

Demonstrates Haiku's cross-modal retrieval capabilities using preprocessed demo data from `dataset/`.

## Notebooks

### 01 — Patch Visualization

[`01_patching_process.ipynb`](01_patching_process.ipynb) visualizes preprocessed CODEX + H&E patches:
- Multi-channel biomarker views (DAPI, CD3e, CD8, PanCK, Ki67, CD20, etc.)
- RGB composite overlays (R=PanCK, G=CD3e, B=DAPI)
- Whole-region mosaics reconstructed from all 116 patches

### 02 — Embedding Inference + Retrieval

[`02_text_and_embedding.ipynb`](02_text_and_embedding.ipynb) runs the full Haiku pipeline:
1. Loads the pretrained Haiku model (mIF + H&E + Text encoders)
2. Extracts trimodal embeddings for 959 patches across 4 tissue regions
3. Computes Text-to-CODEX and H&E-to-CODEX retrieval with ground-truth comparison
4. Visualizes top-5 retrieved patches as multi-channel composites with similarity scores

**Pre-executed versions** (`*_executed.ipynb`) are included with all outputs for reference.

## Data

All data is loaded from `dataset/` — no external paths or downloads required (beyond the model checkpoint).
