"""Assemble the Haiku(Bi) (CODEX + H&E only) Hugging Face bundle locally.

Mirrors ``hf_bundle/`` (built by ``upload_to_hf.py``) but for the bi-modal
ablation checkpoint trained with only the H&E <-> mIF contrastive term.  The
text encoder (an untouched copy of pretrained BiomedBERT) and the untrained
text projection are dropped, and no tokenizer is shipped; ``config.json`` sets
``"use_text": false`` so ``Haiku.from_pretrained`` builds the bi-modal model.

This script only builds the folder; it never uploads.

Usage
-----
    python scripts/build_hf_bundle_bi.py \
        --checkpoint <path/to/haiku_bi_checkpoint>.pth
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

HAIKU_ROOT = Path(__file__).resolve().parents[1]
TEXT_PREFIXES = ("text_encoder.", "text_projection.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tri-bundle", type=Path, default=HAIKU_ROOT / "hf_bundle",
                        help="Existing tri-modal bundle (source of config fields, esm_embeddings/, vocab.pkl)")
    parser.add_argument("--staging-dir", type=Path, default=HAIKU_ROOT / "hf_bundle_bi")
    args = parser.parse_args()

    staging = args.staging_dir
    staging.mkdir(parents=True, exist_ok=True)

    with open(args.tri_bundle / "config.json") as f:
        tri_cfg = json.load(f)
    drop = {"hf_model", "text_dim", "freeze_bert_layers", "tune_bert_layers"}
    cfg = {
        "model_name": "Haiku(Bi)",
        "modalities": ["codex", "he"],
        "use_text": False,
        **{k: v for k, v in tri_cfg.items() if k not in drop},
        "training": {
            "source_checkpoint": f"{args.checkpoint.parent.name}/{args.checkpoint.name}",
            "loss_modalities": ["HandE", "codex"],
            "epochs": 25,
            "seed": 1001,
        },
    }
    with open(staging / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    kept = {k: v for k, v in sd.items() if not k.startswith(TEXT_PREFIXES)}
    print(f"state_dict: kept {len(kept)} / {len(sd)} tensors (dropped text_encoder.* + text_projection.*)")
    torch.save(kept, staging / "haiku_state_dict.pt")

    shutil.copytree(args.tri_bundle / "esm_embeddings", staging / "esm_embeddings", dirs_exist_ok=True)
    shutil.copy2(args.tri_bundle / "vocab.pkl", staging / "vocab.pkl")

    for p in sorted(staging.rglob("*")):
        if p.is_file() and p.parent == staging:
            print(f"  {p.name:30s} {p.stat().st_size / 1e6:10.2f} MB")
    print(f"Bundle ready at {staging} (README.md is written separately; not uploaded)")


if __name__ == "__main__":
    main()
