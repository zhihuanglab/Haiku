import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

from text_preprocess_utils import TextPreprocessPaths, bm_summary_to_text, ensure_dir, load_region_ids, shard_items

NUM_WORKERS = 32


def calculate_spatial_distribution_metrics(channel_data):
    if np.max(channel_data) > 0:
        normalized = ((channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data)) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(channel_data, dtype=np.uint8)

    non_zero_mask = channel_data > 0
    if np.any(non_zero_mask):
        mean_intensity = np.mean(channel_data[non_zero_mask])
        std_intensity = np.std(channel_data[non_zero_mask])
        cv = std_intensity / mean_intensity if mean_intensity > 0 else 0.0
    else:
        cv = 0.0

    if np.any(non_zero_mask):
        grad_y, grad_x = np.gradient(channel_data.astype(float))
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        spatial_clustering = 1.0 / (1.0 + np.mean(gradient_magnitude[non_zero_mask]))
    else:
        spatial_clustering = 0.0

    try:
        if np.any(normalized > 0):
            glcm = graycomatrix(normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
            contrast = graycoprops(glcm, "contrast")[0, 0]
        else:
            homogeneity = 0.0
            contrast = 0.0
    except Exception:
        homogeneity = 0.0
        contrast = 0.0

    coverage = float(np.sum(non_zero_mask)) / float(channel_data.size)
    return {
        "coefficient_of_variation": float(cv),
        "spatial_clustering": float(spatial_clustering),
        "homogeneity": float(homogeneity),
        "contrast": float(contrast),
        "coverage": float(coverage),
    }


def classify_spatial_distribution(metrics):
    cv = metrics["coefficient_of_variation"]
    clustering = metrics["spatial_clustering"]
    homogeneity = metrics["homogeneity"]
    coverage = metrics["coverage"]

    if coverage < 0.1:
        return "sparse"
    if coverage > 0.7:
        if cv < 0.5 and homogeneity > 0.6:
            return "uniform_dense"
        if clustering > 0.7:
            return "clustered_dense"
        return "heterogeneous_dense"
    if coverage > 0.3:
        if clustering > 0.6:
            return "clustered_moderate"
        if cv < 0.6:
            return "uniform_moderate"
        return "heterogeneous_moderate"
    if clustering > 0.5:
        return "clustered_sparse"
    return "scattered_sparse"


def get_sample_tabular_metadata(sample_id, region_metadata_dir):
    region_id = sample_id.split("_")[0]
    csv_path = os.path.join(region_metadata_dir, f"{region_id}.metadata.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty or "FEATURE_NAME" not in df.columns or "FEATURE_VALUE" not in df.columns:
            return None
        feature_descriptions = [
            f"{row['FEATURE_NAME']}: {row['FEATURE_VALUE']}"
            for _, row in df.iterrows()
            if pd.notna(row["FEATURE_NAME"]) and pd.notna(row["FEATURE_VALUE"])
        ]
        return "Metadata: " + ", ".join(feature_descriptions) if feature_descriptions else None
    except Exception:
        return None


def check_region_completed(region_id, region_path, output_dir):
    region_output_dir = os.path.join(output_dir, region_id)
    if not os.path.exists(region_output_dir):
        return False
    sample_files = [f for f in os.listdir(region_path) if f.endswith(".pkl")]
    for sample_file in sample_files:
        sample_id = sample_file.replace(".pkl", "")
        output_file = os.path.join(region_output_dir, f"{sample_id}_backgroud.txt")
        if not os.path.exists(output_file):
            return False
    return True


def load_sample_stats(sample_file, region_path, metadata_dir):
    sample_id = sample_file.replace(".pkl", "")
    file_path = os.path.join(region_path, sample_file)
    spatial_descriptions = {}
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        if "codex" not in data:
            return sample_id, None, {}, None, "No CODEX data", None

        codex_data = data["codex"]
        biomarkers = data["biomarker_name"]
        if len(biomarkers) != codex_data.shape[0]:
            return sample_id, None, {}, None, "Channel/biomarker mismatch", None

        channel_means = []
        for channel_idx in range(codex_data.shape[0]):
            channel_data = codex_data[channel_idx]
            non_zero_mask = channel_data > 0
            mean_intensity = float(np.mean(channel_data[non_zero_mask])) if np.any(non_zero_mask) else 0.0
            channel_means.append(mean_intensity)
            spatial_metrics = calculate_spatial_distribution_metrics(channel_data)
            spatial_descriptions[biomarkers[channel_idx]] = classify_spatial_distribution(spatial_metrics)

        tabular_description = get_sample_tabular_metadata(sample_id, metadata_dir)
        return sample_id, np.asarray(channel_means, dtype=float), spatial_descriptions, tabular_description, None, biomarkers
    except Exception as e:
        return sample_id, None, {}, None, str(e), None


def describe_and_write_region(region_id, region_path, output_dir, metadata_dir):
    region_output_dir = os.path.join(output_dir, region_id)
    ensure_dir(region_output_dir)

    sample_files = [f for f in os.listdir(region_path) if f.endswith(".pkl")]
    if not sample_files:
        return

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futures = {ex.submit(load_sample_stats, sf, region_path, metadata_dir): sf for sf in sample_files}
        for fut in tqdm(as_completed(futures), total=len(futures), leave=False, desc=f"[{region_id}] Loading samples"):
            results.append(fut.result())

    valid = [r for r in results if (r[1] is not None and r[-1] is not None and len(r[1]) == len(r[-1]))]
    if not valid:
        return

    biomarkers = valid[0][-1]
    sample_ids = [r[0] for r in valid]
    sample_means = np.stack([r[1] for r in valid], axis=0)
    spatial_maps = {r[0]: r[2] for r in valid}
    tabular_map = {r[0]: r[3] for r in valid}

    n_samples, n_channels = sample_means.shape
    eps = 1e-6
    z = np.zeros_like(sample_means, dtype=float)
    perc = np.zeros_like(sample_means, dtype=float)

    for c in range(n_channels):
        col = sample_means[:, c]
        for i in range(n_samples):
            ref = np.delete(col, i)
            ref = ref[np.isfinite(ref)]
            if ref.size == 0:
                z[i, c] = 0.0
                perc[i, c] = 50.0
                continue
            z[i, c] = (float(col[i]) - float(np.mean(ref))) / (float(np.std(ref)) + eps)
            perc[i, c] = 100.0 * (np.sum(ref <= col[i]) / ref.size)

    for idx, sample_id in enumerate(sample_ids):
        target_summary = {
            biomarkers[j]: {"z": float(z[idx, j]), "percentile": float(perc[idx, j])}
            for j in range(n_channels)
        }

        base_description, _ = bm_summary_to_text(
            target_summary,
            z_threshold=1.0,
            percentile_threshold=80,
            always_include_topk=3,
            include_high=True,
            include_low=True,
            include_others=True,
        )

        spatial_info = "\n\nSpatial Distribution Patterns:\n"
        for biomarker in biomarkers:
            pattern = spatial_maps[sample_id].get(biomarker, "unknown")
            spatial_info += f"- {biomarker}: {pattern.replace('_', ' ')}\n"

        caption = ""
        tabular_description = tabular_map.get(sample_id)
        if tabular_description:
            caption += f"{tabular_description}\n\n"
        caption += f"Biomarker Expression Analysis (absolute, cross-patch reference):\n{base_description}{spatial_info}"

        out_path = os.path.join(region_output_dir, f"{sample_id}_backgroud.txt")
        with open(out_path, "w") as f:
            f.write(caption)


def main(job_id, max_job_number, paths):
    individual_samples_dir = paths.individual_samples_dir
    output_dir = paths.background_text_dir
    ensure_dir(output_dir)

    region_list = load_region_ids(paths.region_json)
    assigned_regions = shard_items(region_list, job_id, max_job_number)

    for region_id in tqdm(assigned_regions, desc=f"[Job {job_id}] Processing regions"):
        region_path = os.path.join(individual_samples_dir, region_id)
        if not os.path.exists(region_path):
            continue
        if check_region_completed(region_id, region_path, output_dir):
            continue
        describe_and_write_region(region_id, region_path, output_dir, paths.metadata_dir)

    print(f"Job {job_id} done. Processed {len(assigned_regions)} regions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--max_job_number", type=int, required=True)
    parser.add_argument(
        "--region-json",
        default="/home/yancui/Haiku/resources/overlap_samples_final.json",
    )
    args = parser.parse_args()
    p = TextPreprocessPaths(region_json=args.region_json)
    main(args.job_id, args.max_job_number, p)
