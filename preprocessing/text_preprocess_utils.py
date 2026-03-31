"""Shared utilities for Haiku text preprocessing/generation jobs."""

import json
import os
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextPreprocessPaths:
    root_dir: str = "/data/enable_data"
    region_json: str = "/home/yancui/Haiku/resources/overlap_samples_final.json"
    metadata_dir: str = "/project/zhihuanglab/common/datasets/enable_data/region_metadata"

    @property
    def individual_samples_dir(self) -> str:
        return os.path.join(self.root_dir, "new_individual_samples")

    @property
    def background_text_dir(self) -> str:
        return os.path.join(self.root_dir, "new_background_text")

    @property
    def caption_text_dir(self) -> str:
        return os.path.join(self.root_dir, "caption_background_text_v3")


def load_region_ids(region_json: str) -> List[str]:
    sample_dict = json.load(open(region_json))
    return list(sample_dict.keys())


def shard_items(items: List[str], job_id: int, max_job_number: int) -> List[str]:
    total = len(items)
    per_job = (total + max_job_number - 1) // max_job_number
    start_idx = job_id * per_job
    end_idx = min((job_id + 1) * per_job, total)
    return items[start_idx:end_idx]


def read_txt_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bm_summary_to_text(
    target_summary,
    z_threshold=1.0,
    percentile_threshold=80,
    always_include_topk=3,
    include_high=True,
    include_low=True,
    include_others=True,
):
    """Lightweight replacement for missing patchsum.bm_summary_to_text.

    Returns:
        text: bullet list of selected biomarkers.
        selected: dict of selected biomarker stats.
    """
    rows = []
    for name, stats in target_summary.items():
        z = float(stats.get("z", 0.0))
        p = float(stats.get("percentile", 50.0))
        rows.append((name, z, p))

    rows_sorted_abs = sorted(rows, key=lambda x: abs(x[1]), reverse=True)
    selected = {}

    for name, z, p in rows_sorted_abs[:always_include_topk]:
        selected[name] = {"z": z, "percentile": p}

    for name, z, p in rows:
        if include_high and (z >= z_threshold or p >= percentile_threshold):
            selected[name] = {"z": z, "percentile": p}
        elif include_low and (z <= -z_threshold or p <= (100 - percentile_threshold)):
            selected[name] = {"z": z, "percentile": p}
        elif include_others and name in selected:
            selected[name] = {"z": z, "percentile": p}

    selected_rows = sorted(
        [(k, v["z"], v["percentile"]) for k, v in selected.items()],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    text = "\n".join([f"- **{name}**: z-score = {z:.2f}, percentile = {p:.1f}" for name, z, p in selected_rows])
    return text, selected
