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
| 13 | `13_cohort_counterfactual_lung.ipynb` | Cohort-scale per-patient counterfactual (71 Deceased lung patients; patient-level tests, clustering, grade classifier) |

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

## 👥 Cohort-Scale Counterfactual (13)

Repeats the notebook-12 Deceased → Alive survival edit independently for every
lung cancer patient recorded as Deceased (n = 71; manuscript Fig. 6e–h,
Supplementary Fig. 9, Methods "Population-scale per-patient counterfactual
inference"). Each patient keeps their own record; the edit sets the status to
Alive and multiplies their own survival duration by 2.4 (the 25 → 60 months
ratio of the single-patient edit). Retrieval settings match notebook 12
(`w_he = 0.6`, top-50, full atlas). Each patient is reduced to one shift per
marker (θ<sub>p,m</sub>, the score-weighted change in marker prevalence among
retrieved mIF neighbours, averaged over that patient's patches), and all
inference is at the patient level:

- one-sample two-sided Wilcoxon signed-rank test across patients per marker,
  BH q values over the 53 testable channels, paired d<sub>z</sub>, patient
  bootstrap 95% CI (B = 2,000);
- average-linkage / correlation-distance clustering of the standardized shift
  vectors;
- grade 3 vs grade 2 ℓ<sub>1</sub>-logistic regression under nested CV (outer
  stratified 5-fold, inner stratified 3-fold over C), pooled out-of-fold AUROC
  with a 2,000-permutation P.

The logic lives in [`cohort/`](cohort/) (`cohort_counterfactual.py`,
`cohort_stats.py`, `cohort_plots.py`); outputs (effect matrix, per-marker
table, coefficients, figures) are written to `downstream/cohort_outputs/`.
Inputs, set in the notebook's config cell (or the environment variables of
the same name): `EMBEDDINGS_DIR` (precomputed `he_embedding.pt`,
`codex_embedding.pt`, `region_label.pt`, `region_id_mapping.json`),
`PATCH_LABELS_NPZ` (per-patch binary marker positivity, rows aligned with the
embeddings; `cohort.cohort_counterfactual.build_label_matrix` builds it from the
per-patch label JSONs) and `METADATA_DIR` (region metadata CSVs). Captions are
encoded with `Haiku.from_pretrained("zhihuanglab/Haiku")`. The whole cohort runs
in under a minute on one GPU.

> These inputs are derived from the held-out paired atlas, which is not part of
> `zhihuanglab/Haiku-demo-data`; the notebook therefore cannot be re-run from the
> public demo data alone. The executed notebook keeps its outputs so the
> results can be inspected without the data.

## ⚙️ Requirements

These notebooks require precomputed Haiku embeddings. Generate them with:

```bash
python examples/extract_haiku_multimodal_embeddings.py \
    --checkpoint checkpoints/Trimodal_20260303-0300_full_trainset/clip_checkpoint_epoch_24.pth \
    --output-dir outputs/multimodal_embeddings
```
