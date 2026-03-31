"""Shared sample filtering/split logic reused across notebooks."""

import json
import os

import pandas as pd

from .constants import HaikuPaths


def _read_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_exclude_ids(paths=None):
    paths = paths or HaikuPaths()
    exclude_ids = {
        "s624_c001_v001_r001_reg001",
        "ffx-50241",
        "hxs-77087",
        "UofChica-89_c001_v001_r001_reg001",
    }

    holdout_list = set(pd.read_csv(paths.holdout_csv)["ACQUISITION_ID"].tolist())
    image_skip_list = set(_read_lines(paths.image_skip_txt))
    test_ids = set(_read_lines(paths.test_regions_txt))
    liver_cancer = set(pd.read_csv(paths.liver_cancer_csv)["ACQUISITION_ID"].tolist())
    overlap_new = set(json.load(open(paths.overlap_new_json)).keys())

    exclude_ids.update(holdout_list)
    exclude_ids.update(image_skip_list)
    exclude_ids.update(test_ids)
    exclude_ids.update(liver_cancer)
    exclude_ids.update(overlap_new)
    return exclude_ids


def get_whole_region_ids(paths=None):
    paths = paths or HaikuPaths()
    all_ids = sorted(os.listdir(paths.codex_root))
    exclude_ids = build_exclude_ids(paths)
    return [rid for rid in all_ids if rid not in exclude_ids], all_ids


def get_paired_region_ids(paths=None):
    paths = paths or HaikuPaths()
    sample_dict = json.load(open(paths.paired_json))
    exclude_ids = build_exclude_ids(paths)
    return sorted([rid for rid in sample_dict.keys() if rid not in exclude_ids]), sample_dict
