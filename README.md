# Haiku: Linking Spatial Biology and Clinical Histology

**A tri-modal contrastive learning model that aligns multiplexed immunofluorescence (mIF), H&E histology, and clinical text within a shared embedding space.**

<p align="center">
  <img src="figures/image (3).png" width="30%" alt="Haiku Logo">
</p>

Integrating molecular, morphological, and clinical data is essential for translational biomedical research, yet systematic frameworks for jointly modeling these modalities remain limited. Haiku is pretrained on **26.7 million spatial proteomics patches** from **3,218 tissue sections**, enabling cross-modal retrieval, downstream clinical prediction, zero-shot biomarker inference, and counterfactual perturbation analysis.

<p align="center">
  <img src="figures/figure1.png" width="100%" alt="Haiku Overview">
</p>

> **(a)** Training data composition and registered mIF + H&E images.
> **(b)** Tri-modal contrastive learning with modality-specific encoders and projection heads.
> **(c)** Cross-modality retrieval in shared embedding space.
> **(d)** Linear probing for unimodal and fused classification.
> **(e)** Slice-level MIL prediction for survival and treatment response.
> **(f)** Fusion retrieval combining H&E and text embeddings.
> **(g)** Metadata-enhanced biomarker inference via fusion retrieval + PCC.
> **(h)** Counterfactual prediction through in-silico metadata perturbation.

## Highlights

- **Three-way cross-modal retrieval** across mIF, H&E, and clinical text
- **Zero-shot biomarker inference** through fusion retrieval conditioned on metadata-only text descriptions that exclude explicit biomarker information
- **Counterfactual prediction framework** that modifies clinical metadata while fixing tissue morphology, revealing niche-specific molecular remodeling programs associated with breast cancer stage progression and lung cancer survival outcome
- **Improved downstream performance** over unimodal baselines on classification and clinical prediction tasks

---

## Installation

```bash
conda env create -f environment.yml
conda activate haiku
```

### Requirements

- Python 3.11+
- PyTorch 2.6+
- CUDA 12.x (GPU recommended)

Key dependencies: `transformers`, `timm`, `omegaconf`, `h5py`, `tifffile`, `scikit-image`

---

## Quick Start

### 1. Patch Visualization

[`example_retrieval/patch_visualization.ipynb`](example_retrieval/patch_visualization.ipynb) -- Visualize preprocessed CODEX + H&E patches with multi-channel biomarker overlays and whole-region mosaics.

### 2. Cross-Modal Retrieval

[`example_retrieval/case_example.ipynb`](example_retrieval/case_example.ipynb) -- Load a pretrained Haiku model, extract trimodal embeddings across 4 tissue regions (959 patches), and run Text-to-CODEX and H&E-to-CODEX retrieval with ground-truth comparison.

### 3. Downstream Analysis

[`downstream/`](downstream/) -- Biomarker inference (fusion PCC), linear probing, MIL classification/survival, and perturbation analysis.

Pre-executed notebooks with all outputs are provided as `*_executed.ipynb` for reference.

---

## Directory Structure

```
Haiku/
├── README.md
├── environment.yml
├── src/
│   ├── configs/config.yaml           # Model and training configuration
│   ├── models/
│   │   ├── haiku_model.py            # Haiku trimodal model
│   │   ├── encoders.py               # Text (BiomedBERT), mIF (VirTues), H&E (MUSK) encoders
│   │   └── embedding_module.py       # Marker embedding (ESM + learnable)
│   ├── data/dataset.py               # Dataset classes and collate functions
│   ├── utils/                        # Loss functions and transforms
│   ├── haiku/                        # Notebook utility package
│   └── virtues/                      # VirTues MAE encoder (submodule)
├── preprocessing/                    # Data preprocessing pipeline
│   ├── mask.py                       # CNN tissue segmentation
│   ├── codex_patch_single_region.py  # CODEX patch extraction
│   ├── he_patch_from_codex_ids.py    # H&E patch extraction
│   ├── text_gen_mp.py                # Text description generation
│   └── enhance_des.py                # Text enhancement
├── dataset/                          # Demo data (5 regions, 1075 patches)
│   ├── demo_samples.json
│   ├── vocab.pkl
│   ├── esm_embeddings/
│   ├── codex_patches/{region_id}/
│   ├── he_patches/{region_id}/
│   └── text/{region_id}/
├── example_retrieval/                # Retrieval example notebooks
└── downstream/                       # Downstream analysis notebooks
```

---

## Data

The `dataset/` directory contains preprocessed demo data for **5 tissue regions** (1,075 patches total). For the full dataset, see our data release on [Zenodo](#) (link forthcoming).

Each patch consists of:
| Modality | Format | Shape | Description |
|----------|--------|-------|-------------|
| mIF (CODEX) | `.pkl` | (C, 256, 256) | 54-channel multiplexed immunofluorescence |
| H&E | `.npy` | (256, 256, 3) | Registered histology patch |
| Text | `.txt` | -- | Clinical metadata + biomarker expression narrative |

## Preprocessing

To preprocess your own data from raw CODEX + H&E TIFFs, see the [preprocessing guide](preprocessing/README.md).

---

## Model Architecture

| Component | Backbone | Embedding Dim |
|-----------|----------|:---:|
| mIF Encoder | VirTues (ViT MAE) | 512 |
| H&E Encoder | MUSK (ViT-Large) | 1024 |
| Text Encoder | BiomedBERT | 768 |
| Projection Heads | Per-modality MLP | 1024 |
| Marker Embedding | ESM + learnable | 1152 &rarr; 512 |

---

## Acknowledgments

We gratefully acknowledge the following open-source projects that Haiku builds upon:

- **[MUSK](https://github.com/lilab-stanford/MUSK)** -- H&E vision encoder pretrained on large-scale pathology data
- **[VirTues](https://github.com/Boehringer-Ingelheim/VirTues)** -- Vision Transformer MAE for multiplexed tissue imaging
- **[BiomedBERT](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)** -- Biomedical language model
- **[ESM](https://github.com/facebookresearch/esm)** -- Protein language model for marker embeddings

## Citation

```bibtex
@article{haiku2026,
  title={Linking Spatial Biology and Clinical Histology via Haiku},
  author={...},
  year={2026}
}
```
