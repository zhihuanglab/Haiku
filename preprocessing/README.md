# Preprocessing Pipeline

This directory contains the complete preprocessing pipeline to go from raw CODEX + H&E TIFFs to Haiku-ready patch data.

## Input Data Requirements

For each tissue region, you need:

```
raw_data/{region_id}/
├── DAPI.tif              # Nuclear stain (required for tissue mask)
├── CD3e.tif              # One TIF per biomarker channel
├── CD8.tif
├── PanCK.tif
├── ...                   # Any number of biomarker channels
└── HandE.tif             # Registered H&E image (RGB)
```

- All biomarker TIFFs should be co-registered to the same coordinate space
- H&E image (`HandE.tif`) must be spatially aligned with the CODEX channels
- Files named `tissue*.tif`, `segmentation*.tif`, or `HandE*.tif` are excluded from biomarker extraction

## Pipeline Steps

### Step 1: Tissue Segmentation

Generate a binary tissue foreground mask from the DAPI channel using a pretrained CNN model.

```bash
# Requires: tissue segmentation checkpoint
python mask.py
```

- **Input**: `{region_dir}/DAPI.tif`
- **Output**: `{mask_dir}/{region_id}/foreground_mask.ome.tif`
- **Model**: DeepLabV3-ResNet50 trained on tissue segmentation

### Step 2: CODEX Patch Extraction

Extract normalized CODEX patches using a sliding window over the tissue mask.

```bash
# Single region
python codex_patch_single_region.py \
    --region-id amm-49489 \
    --data-root /path/to/raw_data \
    --mask-root /path/to/masks \
    --output-root /path/to/output/codex_patches \
    --patch-size 256 \
    --stride-ratio 0.7

# Batch processing (multi-job parallel)
python new_patch_mp.py --job_id 0 --max_job_number 20
```

- **Input**: Raw biomarker TIFFs + tissue mask
- **Output**: `{output}/{region_id}/{patch_name}.pkl` containing:
  - `codex`: numpy array of shape `(C, 256, 256)` — normalized uint8 per channel
  - `biomarker_name`: list of channel names
- **Normalization**: Per-channel histogram-based clipping + uint8 scaling (via `PIF.full_img_utils`)

### Step 3: H&E Patch Extraction

Extract H&E patches at the same coordinates as CODEX patches.

```bash
python he_patch_from_codex_ids.py \
    --region-id amm-49489 \
    --codex-patch-root /path/to/codex_patches \
    --full-regions-root /path/to/raw_data \
    --he-output-root /path/to/output/he_patches
```

- **Input**: `HandE.tif` + CODEX patch filenames (coordinates encoded in names)
- **Output**: `{output}/{region_id}/{patch_name}.npy` — shape `(256, 256, 3)` uint8

### Step 4: Text Description Generation

Generate biomarker expression descriptions for each patch.

```bash
python text_gen_mp.py --job_id 0 --max_job_number 20 \
    --region-json /path/to/region_list.json
```

- **Input**: CODEX patches + region metadata CSV
- **Output**: `{output}/{region_id}/{patch_name}_backgroud.txt`
- **Process**: computes per-channel z-scores and percentiles relative to other patches in the same region, classifies spatial distribution patterns (clustered, sparse, uniform, etc.)

### Step 5: Text Enhancement (Optional)

Rephrase raw descriptions into natural-language narratives with clinical context.

```bash
python enhance_des.py --job_id 0 --max_job_number 20 \
    --region-json /path/to/region_list.json
```

- **Input**: Background text files from Step 4
- **Output**: Enhanced captions with tissue type, disease context, and biomarker narratives

## Directory Overview

| File | Purpose |
|------|---------|
| `mask.py` | CNN tissue segmentation (DeepLabV3) |
| `codex_patch_single_region.py` | Single-region CODEX patching (CLI) |
| `new_patch_mp.py` | Multi-job parallel CODEX patching |
| `he_patch_from_codex_ids.py` | H&E patch extraction aligned to CODEX |
| `text_gen_mp.py` | Text description generation with spatial metrics |
| `enhance_des.py` | Text enhancement with clinical narratives |
| `text_preprocess_utils.py` | Shared config and utilities |
| `biomarker_name_mapping.json` | Biomarker name standardization |
| `patchsum/` | Patch summarization: image loading, ROI generation, biomarker stats |
| `PIF/` | Image normalization and feature extraction utilities |

## Output Structure

After running the full pipeline:

```
output/
├── codex_patches/{region_id}/*.pkl    # Normalized CODEX patches
├── he_patches/{region_id}/*.npy       # H&E patches
└── text/{region_id}/*.txt             # Text descriptions
```

This matches the layout expected by `dataset/` and the Haiku dataset loader.
