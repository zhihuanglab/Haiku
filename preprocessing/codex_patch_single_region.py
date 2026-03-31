#!/usr/bin/env python3
"""Run CODEX patching for one region with Haiku preprocessing stack."""

import argparse
import os
import pickle
import time

from PIF.full_img_utils import normalize_full_image_fast
from patchsum.raw_bm_info import get_full_image_for_region_local, get_tissue_mask_for_region_local
from patchsum.utils import get_sliding_window_ROIs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-id", default="amm-49489")
    parser.add_argument("--data-root", default="/project/zhihuanglab/common/datasets/enable_data/full_regions")
    parser.add_argument("--mask-root", default="/data/enable_data/new_mask")
    parser.add_argument("--output-root", default="/home/yancui/Haiku/resources/example_data/amm-49489/processed/codex_patches")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride-ratio", type=float, default=0.7)
    parser.add_argument("--max-patches", type=int, default=None, help="Optional cap for fast demo.")
    args = parser.parse_args()

    region_id = args.region_id
    region_dir = os.path.join(args.data_root, region_id)
    out_dir = os.path.join(args.output_root, region_id)
    os.makedirs(out_dir, exist_ok=True)

    biomarkers = sorted(
        [
            f.split(".tif")[0]
            for f in os.listdir(region_dir)
            if f.endswith(".tif") and not f.startswith(("tissue", "segmentation", "HandE"))
        ]
    )
    if not biomarkers:
        raise RuntimeError(f"No biomarker tif files found for {region_id}")

    t0 = time.time()
    full_img, _ = get_full_image_for_region_local(region_id, biomarkers, root=args.data_root)
    tissue_mask = get_tissue_mask_for_region_local(region_id, root=args.mask_root).astype(int)
    normalized_img = normalize_full_image_fast(full_img, tissue_mask)
    h, w = tissue_mask.shape

    rois = get_sliding_window_ROIs(
        region_id,
        w,
        h,
        patch_size=args.patch_size,
        tissue_mask=tissue_mask,
        tissue_mask_coverage_ratio=0.9,
        num_repetition=1,
        stride_ratio=args.stride_ratio,
    )
    if args.max_patches is not None:
        rois = rois[: args.max_patches]

    wrote = 0
    for roi in rois:
        cx, cy = roi["center_x"], roi["center_y"]
        sx, sy = int(cx - args.patch_size / 2), int(cy - args.patch_size / 2)
        if sx < 0 or sy < 0 or sx + args.patch_size > w or sy + args.patch_size > h:
            continue
        codex_patch = normalized_img[:, sy : sy + args.patch_size, sx : sx + args.patch_size]
        out_data = {"codex": codex_patch, "biomarker_name": biomarkers}
        with open(os.path.join(out_dir, f"{roi['patch_name']}.pkl"), "wb") as f:
            pickle.dump(out_data, f)
        wrote += 1

    print(f"region={region_id} rois={len(rois)} wrote={wrote} elapsed={time.time()-t0:.1f}s out={out_dir}")


if __name__ == "__main__":
    main()
