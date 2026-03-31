# Downstream Analysis

Notebooks for evaluating Haiku embeddings on downstream tasks.

## Notebooks

| # | Notebook | Task |
|---|----------|------|
| 03 | `03_knn_retrieval.ipynb` | kNN-based cross-modal retrieval with Recall@K metrics |
| 04 | `04_zero_shot.ipynb` | Zero-shot classification using text-based prototypes |
| 05 | `05_retrieval.ipynb` | Comprehensive retrieval analysis across modalities |
| 06 | `06_biomarker_inference.ipynb` | Fusion retrieval + per-biomarker PCC (Pearson correlation) |
| 07 | `07_linear_probing.ipynb` | Linear probe classification on frozen embeddings |
| 08 | `08_mil_classification.ipynb` | Multiple Instance Learning for patient-level classification |
| 09 | `09_mil_survival.ipynb` | MIL-based survival prediction (Cox regression) |
| 10 | `10_perturbation.ipynb` | In-silico text perturbation analysis |
| 11 | `11_perturbation_metadata_only.ipynb` | Metadata-only perturbation |
| 12 | `12_perturbation_lung_metadata_only.ipynb` | Lung-specific metadata perturbation |

## Biomarker Inference (Notebook 06)

Uses **fusion retrieval** (H&E + metadata text embeddings) to predict biomarker expression via weighted top-K nearest neighbors, evaluated with per-biomarker Pearson correlation (PCC). Compares three methods: H&E-only, Fusion, and MUSK baseline.

## Requirements

These notebooks require pre-computed Haiku embeddings. Generate them using:

```bash
python examples/extract_haiku_multimodal_embeddings.py \
    --checkpoint checkpoints/Trimodal_20260303-0300_full_trainset/clip_checkpoint_epoch_24.pth \
    --output-dir outputs/multimodal_embeddings
```
