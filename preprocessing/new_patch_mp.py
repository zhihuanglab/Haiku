import os
import sys
import numpy as np
import pickle
import tifffile
from tqdm import tqdm
import time
import argparse
import logging

from PIF.full_img_utils import normalize_full_image_fast
from patchsum.raw_bm_info import get_full_image_for_region_local, get_tissue_mask_for_region_local
from patchsum.utils import get_sliding_window_ROIs


# =========================
# Config
# =========================
DATA_ROOT = "/data/enable_data/full_regions"        # contains all acq_ids folders
MASK_DIR = "/data/enable_data/new_mask"
SAVE_DIR = "/data/enable_data/new_individual_samples"
LOG_DIR = os.path.join('/home/yancui/Haiku/preprocessing/', "logs")

ROI_SIZE = 256
STRIDE_RATIO = 0.7


def read_txt(filepath):
    """Read lines from a text file, strip whitespace, and return as list."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


# =========================
# Logging setup
# =========================
def setup_logger(job_id: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"patch_job_{job_id}.log")

    logger = logging.getLogger(f"patch_job_{job_id}")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if main() is somehow called twice
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fh_formatter)

        # Stream handler (terminal)
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        sh.setFormatter(sh_formatter)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger


# =========================
# 1. AUTO-SCAN ALL REGION IDS
# =========================
def load_all_region_ids():
    """Return a sorted list of all acq_id directories inside DATA_ROOT."""
    region_ids = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ])
    return region_ids

def load_all_wrong_region_ids():
    # Read failed region ids from jobs 0 through 19
    wrong_region_ids = set()
    log_dir = "/home/yancui/Haiku/preprocessing/logs"
    for job_id in range(0, 20):
        log_path = os.path.join(log_dir, f"failed_regions_job_{job_id}.txt")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                for line in f:
                    region = line.strip()
                    if region:
                        wrong_region_ids.add(region)
    wrong_region_ids = sorted(list(wrong_region_ids))
    return wrong_region_ids


# =========================
# 2. CHUNK SPLITTING
# =========================
def chunk_region_ids(region_ids, job_id, max_job_number):
    total = len(region_ids)
    chunk_size = (total + max_job_number - 1) // max_job_number  # ceiling division

    start = job_id * chunk_size
    end = min(start + chunk_size, total)

    if start >= total:
        return []  # this job gets no work

    return region_ids[start:end]


# =========================
# 3. PATCHING FUNCTION
# =========================
def patch_region(region_id, idx, total_ids, logger, failed_regions):
    logger.info(f"({idx+1}/{len(total_ids)}) Processing region_id: {region_id}")

    region_dir = os.path.join(DATA_ROOT, region_id)
    region_save_dir = os.path.join(SAVE_DIR, region_id)

    skip_ids = read_txt('/data/enable_data/image_skip.txt')

    if region_id in skip_ids:
        logger.info(f"[SKIP] {region_id}: in skip list.")
        return

    # Skip if already processed
    if os.path.exists(region_save_dir) and len(os.listdir(region_save_dir)) > 5:
        logger.info(f"[SKIP] {region_id}: already processed (existing files > 5).")
        return

    os.makedirs(region_save_dir, exist_ok=True)

    # Load biomarkers
    try:
        biomarkers = [
            f.split(".tif")[0] for f in os.listdir(region_dir)
            if f.endswith(".tif") and not f.startswith(("tissue", "segmentation", "HandE"))
        ]
    except Exception as e:
        logger.error(f"[FAIL] {region_id}: error listing region_dir. Error: {e}")
        failed_regions.append(region_id)
        return

    biomarkers = sorted(biomarkers)

    if len(biomarkers) == 0:
        logger.error(f"[FAIL] {region_id}: no biomarker .tif files found.")
        failed_regions.append(region_id)
        return



    # Load images
    try:
        full_img, _ = get_full_image_for_region_local(region_id, biomarkers, root=DATA_ROOT)
        tissue_mask = get_tissue_mask_for_region_local(region_id, root=MASK_DIR).astype(int)
    except Exception as e:
        logger.error(f"[FAIL] {region_id}: missing image or mask. Error: {e}")
        failed_regions.append(region_id)
        return

    logger.info(f"[INFO] Normalizing region {region_id}, full_img.shape={full_img.shape}, "
                f"tissue_mask.shape={tissue_mask.shape}")

    # Skip extremely large regions
    if full_img.shape[1] >  50000:
        logger.warning(f"[SKIP] {region_id}: image too large (>7000 px in width).")
        return
    # Normalize
    t0 = time.time()
    try:
        normalized_img = normalize_full_image_fast(full_img, tissue_mask)
    except Exception as e:
        logger.error(f"[FAIL] {region_id}: error during normalization. Error: {e}")
        failed_regions.append(region_id)
        return

    H, W = tissue_mask.shape
    logger.info(f"[TIME] normalization = {time.time() - t0:.2f}s")

    # Sliding window patch extraction
    logger.info("[INFO] Running sliding window...")
    t0 = time.time()
    try:
        all_ROIs = get_sliding_window_ROIs(
            region_id,
            W, H,
            patch_size=ROI_SIZE,
            tissue_mask=tissue_mask,
            tissue_mask_coverage_ratio=0.9,
            num_repetition=1,
            stride_ratio=STRIDE_RATIO
        )
    except Exception as e:
        logger.error(f"[FAIL] {region_id}: error during sliding window ROI generation. Error: {e}")
        failed_regions.append(region_id)
        return

    logger.info(f"[TIME] sliding window = {time.time() - t0:.2f}s")
    logger.info(f"[INFO] Found {len(all_ROIs)} ROIs for {region_id}")

    # Write patches
    num_written = 0
    for ROI in all_ROIs:
        cx, cy = ROI["center_x"], ROI["center_y"]
        sx, sy = int(cx - ROI_SIZE/2), int(cy - ROI_SIZE/2)

        if sx < 0 or sy < 0 or sx+ROI_SIZE > W or sy+ROI_SIZE > H:
            # out of bounds, skip
            continue

        codex_patch = normalized_img[:, sy:sy+ROI_SIZE, sx:sx+ROI_SIZE]

        out_data = {
            "codex": codex_patch,
            "biomarker_name": biomarkers
        }

        out_path = os.path.join(region_save_dir, f"{ROI['patch_name']}.pkl")
        try:
            with open(out_path, 'wb') as f:
                pickle.dump(out_data, f)
            num_written += 1
        except Exception as e:
            logger.error(f"[FAIL] {region_id}: error writing patch {ROI['patch_name']}.pkl. Error: {e}")
            failed_regions.append(region_id)
            # Continue to next ROI; partial failure shouldn't kill the whole region

    logger.info(f"[DONE] {region_id}: wrote {num_written} patches.")


# =========================
# 4. MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--max_job_number", type=int, required=True)
    args = parser.parse_args()

    logger = setup_logger(args.job_id)
    logger.info(f"===== STARTING PATCH JOB {args.job_id} / {args.max_job_number} =====")

    # 1) auto-scan full_regions
    #region_ids = load_all_region_ids()
    #region_ids = load_all_wrong_region_ids()
    region_ids = read_txt('/home/yancui/Haiku/preprocessing/missing_biomarker_regions.txt')
    logger.info(f"[INFO] Total regions found under DATA_ROOT={DATA_ROOT}: {len(region_ids)}")

    # 2) split by job
    chunk_ids = chunk_region_ids(region_ids, args.job_id, args.max_job_number)

    if not chunk_ids:
        logger.info(f"[INFO] Job {args.job_id}: no regions assigned. Exiting.")
        return

    logger.info(f"[INFO] Job {args.job_id}: processing {len(chunk_ids)} regions:")
    for rid in chunk_ids:
        logger.info(f"  - {rid}")

    failed_regions = []

    # 3) run patching
    for idx, region_id in enumerate(chunk_ids):
        patch_region(region_id, idx, chunk_ids, logger, failed_regions)

    # 4) write failed list
    if failed_regions:
        failed_regions = sorted(set(failed_regions))
        failed_path = os.path.join(LOG_DIR, f"failed_regions_job_{args.job_id}_new.txt")
        with open(failed_path, "w") as f:
            for rid in failed_regions:
                f.write(rid + "\n")
        logger.warning(f"[SUMMARY] Job {args.job_id}: {len(failed_regions)} regions failed. "
                       f"Written to {failed_path}")
    else:
        logger.info(f"[SUMMARY] Job {args.job_id}: all assigned regions processed without recorded failures.")

    logger.info(f"===== FINISHED PATCH JOB {args.job_id} =====\n")


if __name__ == "__main__":
    main()
