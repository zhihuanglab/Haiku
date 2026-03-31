import argparse
import os
import random
import re

from tqdm import tqdm

from text_preprocess_utils import TextPreprocessPaths, ensure_dir, load_region_ids, shard_items

# 1) Privacy: Keys to remove.
SENSITIVE_KEYS = [
    "position",
    "no.",
    "spot number",
    "core id",
    "core no",
    "spot",
    "patient id",
    "case id",
]

# 2) Cleanliness: Values to remove.
INVALID_VALUES = ["-", "nan", "none", "n/a", "null"]

# 3) Diversity: Intro templates.
SMART_INTROS = [
    "This region captures a sample of {t} tissue characterized by {d}.",
    "Visualized here is a {t} microenvironment presenting with {d}.",
    "This field of view displays {t} architecture associated with {d}.",
    "High-plex imaging of {t} reveals pathological features of {d}.",
    "A representative section of {t} tissue highlighting the landscape of {d}.",
]

GENERIC_INTROS = [
    "This multiplexed image captures a complex cellular microenvironment.",
    "This region of interest highlights a distinct tissue architecture.",
    "Visualized here is a high-resolution field of view from the tissue sample.",
    "This caption describes the molecular and spatial composition of the selected region.",
    "The following analysis details the protein expression profile of this tissue section.",
]

TRANSITIONS = [
    " regarding the molecular profile, the most informative biomarkers are:",
    " molecularly, the region is characterized by the following markers:",
    " in terms of protein expression, the most notable deviations from background are observed in:",
    " quantitative analysis identifies the following markers as most informative relative to background:",
    " the spatial-molecular landscape is defined by the following key markers:",
]

N_HIGH = 5
N_LOW = 5


def clean_and_parse_metadata(raw_meta_str):
    items = raw_meta_str.split(", ")
    clean_items = []
    meta_dict = {}

    for item in items:
        if ":" not in item:
            continue
        k, v = item.split(":", 1)
        k = k.strip()
        v = v.strip()

        if any(s in k.lower() for s in SENSITIVE_KEYS):
            continue
        if v.lower() in INVALID_VALUES:
            continue

        meta_dict[k.lower()] = v
        clean_items.append(f"{k}: {v}")

    return ", ".join(clean_items), meta_dict


def generate_intro_sentence(meta_dict, clean_meta_str):
    tissue = meta_dict.get("tissue_type") or meta_dict.get("tissue")
    disease = meta_dict.get("disease") or meta_dict.get("diagnosis")

    if tissue and disease:
        tmpl = random.choice(SMART_INTROS)
        intro_sentence = tmpl.format(t=tissue, d=disease)
        if clean_meta_str:
            intro_sentence += f" Additional clinical details include: {clean_meta_str}."
        return intro_sentence

    tmpl = random.choice(GENERIC_INTROS)
    if clean_meta_str:
        return f"{tmpl} Clinical metadata includes: {clean_meta_str}."
    return tmpl


def parse_and_rephrase(input_path, output_path):
    if not os.path.exists(input_path):
        return

    try:
        content = open(input_path, "r").read()
    except Exception:
        return

    meta_match = re.search(r"^Metadata: (.*)", content, re.MULTILINE)
    raw_meta = meta_match.group(1) if meta_match else ""
    clean_meta_str, meta_dict = clean_and_parse_metadata(raw_meta)

    stat_pattern = r"- \*\*(.*?)\*\*: z-score = ([\d\.-]+), percentile = ([\d\.]+)"
    stats_matches = re.findall(stat_pattern, content)
    parsed_markers = {}
    for name, z_str, perc_str in stats_matches:
        try:
            parsed_markers[name] = {"z": float(z_str), "percentile": float(perc_str), "pattern": "unknown"}
        except ValueError:
            continue

    for line in content.splitlines():
        line = line.strip()
        if not line.startswith("- ") or "z-score" in line:
            continue
        parts = line.replace("- ", "").split(":")
        if len(parts) != 2:
            continue
        m_name = parts[0].strip()
        m_pat = parts[1].strip().replace("_", " ")
        if m_name in parsed_markers:
            parsed_markers[m_name]["pattern"] = m_pat

    if not parsed_markers:
        return

    marker_list = [{"name": k, **v} for k, v in parsed_markers.items()]
    pos_markers = sorted([m for m in marker_list if m["z"] > 0], key=lambda x: x["z"], reverse=True)[:N_HIGH]
    neg_markers = sorted([m for m in marker_list if m["z"] < 0], key=lambda x: x["z"])[:N_LOW]

    if not pos_markers and not neg_markers:
        return

    intro_part = generate_intro_sentence(meta_dict, clean_meta_str)
    transition_part = random.choice(TRANSITIONS)

    parts = [f"{intro_part}{transition_part}"]
    if pos_markers:
        high = []
        for m in pos_markers:
            suffix = f" exhibiting a {m['pattern']} pattern" if m["pattern"] != "unknown" else ""
            high.append(f"{m['name']} shows relatively high expression{suffix}")
        parts.append("The region is enriched for " + "; ".join(high) + ".")
    if neg_markers:
        low = []
        for m in neg_markers:
            suffix = f" with a {m['pattern']} pattern" if m["pattern"] != "unknown" else ""
            low.append(f"{m['name']} is relatively low in expression{suffix}")
        parts.append("Conversely, it shows relative depletion of " + "; ".join(low) + ".")

    with open(output_path, "w") as f:
        f.write(" ".join(parts))


def main(job_id, max_job_number, paths):
    input_dir = paths.background_text_dir
    output_dir = paths.caption_text_dir
    ensure_dir(output_dir)

    all_regions = load_region_ids(paths.region_json)
    assigned_regions = shard_items(all_regions, job_id, max_job_number)
    print(f"Job {job_id}/{max_job_number} processing {len(assigned_regions)} regions...")

    for region_id in tqdm(assigned_regions):
        region_in_path = os.path.join(input_dir, region_id)
        if not os.path.exists(region_in_path):
            continue
        region_out_path = os.path.join(output_dir, region_id)
        ensure_dir(region_out_path)
        files = [f for f in os.listdir(region_in_path) if f.endswith(".txt")]
        for filename in files:
            parse_and_rephrase(os.path.join(region_in_path, filename), os.path.join(region_out_path, filename))

    print(f"Job {job_id} done.")


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
