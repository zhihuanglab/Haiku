"""Bundle Haiku checkpoint + tokenizer + marker assets and push to HuggingFace Hub.

Usage
-----
    python scripts/upload_to_hf.py \
        --checkpoint checkpoints/Trimodal_20260303-0300_full_trainset/clip_checkpoint_epoch_24.pth \
        --repo-id C1vy/Haiku \
        --private

Requires HF login (``hf auth login`` / ``huggingface-cli login``) or ``--token``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import torch
from transformers import BertTokenizer, BertConfig

HAIKU_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(HAIKU_ROOT / "src"))


def build_config(vocab_path: Path, esm_dir: Path) -> dict:
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    vocab = [v.replace(".", "_") if "." in v else v for v in vocab]

    esm_marker_names = []
    for pt_file in sorted(esm_dir.glob("*.pt")):
        esm_marker_names.append(pt_file.stem)

    known_markers = [m for m in vocab if m not in esm_marker_names]

    return {
        "hf_model": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "codex_dim": 512,
        "text_dim": 768,
        "he_dim": 1024,
        "projection_dim": 1024,
        "shared_projection": False,
        "embedding_dim": 1152,
        "marker_model_dim": 512,
        "freeze_bert_layers": True,
        "tune_bert_layers": [10, 11],
        "freeze_he_encoder": True,
        "freeze_codex_encoder": True,
        "esm_marker_names": esm_marker_names,
        "known_markers": known_markers,
    }


def save_tokenizer_bundle(dest: Path, hf_model: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(hf_model)
    tokenizer.save_pretrained(str(dest))
    config = BertConfig.from_pretrained(hf_model)
    config.save_pretrained(str(dest))


def extract_model_state_dict(ckpt_path: Path, out_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    torch.save(state_dict, str(out_path))


def copy_esm_embeddings(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for pt in src.glob("*.pt"):
        shutil.copy2(pt, dst / pt.name)
        n += 1
    return n


README_TEMPLATE = """---
library_name: pytorch
tags:
  - multimodal
  - histology
  - codex
  - retrieval
license: other
---

# Haiku — Trimodal (CODEX + H&E + Text) Retrieval Model

This repo bundles a fine-tuned Haiku checkpoint together with the tokenizer
and marker assets needed to run inference without any additional downloads
from `xiangjx/musk` or `microsoft/BiomedNLP-BiomedBERT-*`.

## Contents
- `haiku_state_dict.pt` — model weights (CODEX + H&E + Text encoders + projections)
- `config.json` — architecture config + marker lists
- `tokenizer/` — BiomedBERT tokenizer files (+ bert config)
- `esm_embeddings/` — per-biomarker ESM embeddings (also embedded in state_dict; kept here for downstream use)
- `vocab.pkl` — marker vocabulary

## Quick start
```python
from models import Haiku

model, tokenizer, marker_embedding = Haiku.from_pretrained(
    "{repo_id}",
    device="cuda",
    token="hf_...",  # omit if HF_TOKEN / hf auth login is set
)
model.eval()
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", default="C1vy/Haiku")
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=HAIKU_ROOT / "hf_bundle",
        help="Local directory to assemble the bundle before upload",
    )
    parser.add_argument("--vocab", type=Path, default=HAIKU_ROOT / "dataset" / "vocab.pkl")
    parser.add_argument("--esm-dir", type=Path, default=HAIKU_ROOT / "dataset" / "esm_embeddings")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None, help="HF access token (optional)")
    parser.add_argument("--no-upload", action="store_true", help="Only build the bundle locally")
    args = parser.parse_args()

    staging = args.staging_dir
    staging.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Building config from {args.vocab} + {args.esm_dir}")
    cfg = build_config(args.vocab, args.esm_dir)
    with open(staging / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"[2/5] Extracting model_state_dict from {args.checkpoint}")
    extract_model_state_dict(args.checkpoint, staging / "haiku_state_dict.pt")

    print("[3/5] Saving BiomedBERT tokenizer + config to tokenizer/")
    save_tokenizer_bundle(staging / "tokenizer", cfg["hf_model"])

    print(f"[4/5] Copying ESM embeddings from {args.esm_dir}")
    n_esm = copy_esm_embeddings(args.esm_dir, staging / "esm_embeddings")
    print(f"    copied {n_esm} ESM embedding files")

    shutil.copy2(args.vocab, staging / "vocab.pkl")

    with open(staging / "README.md", "w") as f:
        f.write(README_TEMPLATE.format(repo_id=args.repo_id))

    print(f"[5/5] Bundle ready at: {staging}")
    print("      Contents:")
    for p in sorted(staging.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            rel = p.relative_to(staging)
            print(f"        {str(rel):60s}  {size_mb:8.2f} MB")

    if args.no_upload:
        print("--no-upload set — skipping HF upload. Bundle kept locally.")
        return

    print(f"\nUploading bundle to HF Hub: {args.repo_id} (private={args.private})")
    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=args.token)
    create_repo(
        repo_id=args.repo_id,
        private=args.private,
        exist_ok=True,
        token=args.token,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        folder_path=str(staging),
        commit_message="Upload Haiku trimodal checkpoint + tokenizer + marker assets",
        token=args.token,
    )
    print(f"Done. View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
