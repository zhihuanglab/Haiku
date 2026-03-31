#!/usr/bin/env python3
"""Create H&E .npy patches using CODEX patch file names as coordinates."""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import tifffile


def parse_patch_name(stem: str, region_id: str):
    # Expected: region_x1-x2-y1-y2
    m = re.match(rf"^{re.escape(region_id)}_(\d+)-(\d+)-(\d+)-(\d+)$", stem)
    if not m:
        return None
    x1, x2, y1, y2 = map(int, m.groups())
    return x1, x2, y1, y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-id", default="amm-49489")
    parser.add_argument("--codex-patch-root", default="/home/yancui/Haiku/resources/example_data/amm-49489/processed/codex_patches")
    parser.add_argument("--full-regions-root", default="/project/zhihuanglab/common/datasets/enable_data/full_regions")
    parser.add_argument("--he-output-root", default="/home/yancui/Haiku/resources/example_data/amm-49489/processed/he_patches")
    parser.add_argument("--max-patches", type=int, default=None)
    args = parser.parse_args()

    codex_dir = Path(args.codex_patch_root) / args.region_id
    out_dir = Path(args.he_output_root) / args.region_id
    out_dir.mkdir(parents=True, exist_ok=True)

    he_path = Path(args.full_regions_root) / args.region_id / "HandE.tif"
    he_img = tifffile.imread(str(he_path))
    if he_img.ndim == 2:
        he_img = np.repeat(he_img[:, :, None], 3, axis=2)

    patch_files = sorted([p for p in codex_dir.glob("*.pkl")])
    if args.max_patches is not None:
        patch_files = patch_files[: args.max_patches]

    wrote = 0
    for pkl_path in patch_files:
        coords = parse_patch_name(pkl_path.stem, args.region_id)
        if coords is None:
            continue
        x1, x2, y1, y2 = coords
        he_patch = he_img[y1:y2, x1:x2]
        np.save(out_dir / f"{pkl_path.stem}.npy", he_patch)
        wrote += 1

    print(f"region={args.region_id} wrote={wrote} out={out_dir}")


if __name__ == "__main__":
    main()
