# 📊 Downstream Analysis

Notebooks for evaluating Haiku embeddings on downstream tasks.

## 📚 Notebooks

| # | Notebook | Task |
|---|----------|------|
| 03 | `03_knn_retrieval.ipynb` | kNN-based cross-modal retrieval with Recall@K metrics |
| 04 | `04_zero_shot.ipynb` | Zero-shot classification via text-prototype similarity |
| 05 | `05_retrieval.ipynb` | Cross-modal retrieval analysis (Codex/H&E/Text) |
| 06 | `06_biomarker_inference.ipynb` | Fusion retrieval + per-biomarker Pearson correlation (PCC) |
| 07 | `07_linear_probing.ipynb` | Linear probes on precomputed (frozen) embeddings |
| 08 | `08_mil_classification.ipynb` | MIL patient-level classification, sweep-best HPs + early stopping on val |
| 09 | `09_mil_survival.ipynb` | MIL survival prediction, sweep-best HPs + early stopping on val |
| 11 | `11_perturbation_tnbc_metadata_only.ipynb` | Counterfactual metadata perturbation (TNBC, BH-FDR adjusted p-values) |
| 12 | `12_perturbation_lung_metadata_only.ipynb` | Counterfactual metadata perturbation (Lung, BH-FDR adjusted p-values) |

## 🏷️ Canonical Label Harmonization (03 / 04 / 05)

Notebooks `03_knn_retrieval`, `04_zero_shot`, and `05_retrieval` apply the
`canon_tissue` / `canon_disease` maps (ported from
[`sphere-vlm/src/training/zero_shot_canon.py`](../../sphere-vlm/src/training/zero_shot_canon.py))
to collapse surface-form variants of tissue and disease labels into a small
canonical vocabulary (e.g. `colon`, `rectum`, `sigmoid colon` → `Colon/Rectum`;
`breast cancer`, `breast disease` → `Breast Cancer`). Labels that map to
`None` (undefined / `u/a` / `unknown` / etc.) are dropped before retrieval
and zero-shot evaluation.

## 🔬 Biomarker Inference (06)

Uses **fusion retrieval** (H&E + metadata text embeddings) to predict
biomarker expression via weighted top-K nearest neighbors, scored with
per-biomarker Pearson correlation (PCC) against held-out CODEX ground truth.
Compared against H&E-only and MUSK baselines.

## 🧪 Linear Probing & MIL (07 / 08 / 09)

Linear probes (07) run over frozen Haiku embeddings. MIL classification
(08) and MIL survival (09) use sweep-best hyper-parameters with
early stopping on the validation split.

## 🧬 Counterfactual Perturbation (11 / 12)

Metadata-only in-silico perturbation: disease-level text is counterfactually
edited while fixing tissue morphology, and fusion retrieval recovers the
shifted niche. Disease-level violin comparisons use Mann–Whitney U with
**Benjamini–Hochberg FDR correction**; the significance stars overlaid on
plots are driven by the BH-adjusted q-values (thresholds `<0.001 / 0.01 / 0.05`).

## ⚙️ Requirements

These notebooks require precomputed Haiku embeddings. Generate them with:

```bash
python examples/extract_haiku_multimodal_embeddings.py \
    --checkpoint checkpoints/Trimodal_20260303-0300_full_trainset/clip_checkpoint_epoch_24.pth \
    --output-dir outputs/multimodal_embeddings
```
