"""Shared path constants for Haiku utilities."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HaikuPaths:
    project_root: str = "/home/yancui/Haiku"
    codex_root: str = "/data/enable_data/new_individual_samples"
    he_root: str = "/data/enable_data/new_he_patches"
    text_root: str = "/data/enable_data/new_text_patches"
    config_path: str = "/home/yancui/Haiku/src/configs/config.yaml"
    paired_json: str = "/home/yancui/Haiku/resources/overlap_samples_final.json"
    full_json: str = "/home/yancui/Haiku/resources/full_codex_samples.json"
    overlap_new_json: str = "/home/yancui/Haiku/preprocessing/overlap_samples_new.json"
    holdout_csv: str = "/home/yancui/tier2_acquisition_ids_huanglab (3).csv"
    image_skip_txt: str = "/data/enable_data/image_skip.txt"
    test_regions_txt: str = "/home/yancui/OmicsAnnotator/test_regions.txt"
    liver_cancer_csv: str = "/home/yancui/s4769_microe_annotations.csv"
