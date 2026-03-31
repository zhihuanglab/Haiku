import json
import os
import pickle
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from virtues.utils.transform_utils import CropToPatchSize, GridReshape


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def custom_collate_fn_trimodal(batch):
    codex_imgs = [item["codex"] for item in batch]
    texts = torch.stack([item["text"] for item in batch])
    attention_masks = torch.stack([item["att_mask"] for item in batch])
    channels = [item["channels"] for item in batch]

    handes = [item["HandE"] for item in batch]
    if handes[0] is not None:
        handes = torch.stack(handes)
    else:
        handes = None

    region_ids = [item["region_id"] for item in batch]
    patch_ids = [item["patch_id"] for item in batch]
    raw_texts = [item["raw_text"] for item in batch] if "raw_text" in batch[0] else None

    result = {
        "codex": codex_imgs,
        "text": texts,
        "att_mask": attention_masks,
        "channels": channels,
        "region_id": region_ids,
        "patch_id": patch_ids,
        "HandE": handes,
    }
    if raw_texts is not None:
        result["raw_text"] = raw_texts
    return result


def custom_collate_fn_embedding(batch):
    codex_imgs = [item["codex_embedding"] for item in batch]
    texts = torch.stack([item["text"] for item in batch])
    attention_masks = torch.stack([item["att_mask"] for item in batch])

    handes = [item["HandE"] for item in batch]
    if all(h is not None for h in handes):
        handes = torch.stack(handes)
    else:
        handes = None

    region_ids = [item["region_id"] for item in batch]
    patch_ids = [item["patch_id"] for item in batch]
    raw_texts = [item["raw_text"] for item in batch] if "raw_text" in batch[0] else None

    result = {
        "codex_embedding": torch.stack(codex_imgs),
        "text": texts,
        "att_mask": attention_masks,
        "region_id": region_ids,
        "patch_id": patch_ids,
        "HandE": handes,
    }
    if raw_texts is not None:
        result["raw_text"] = raw_texts
    return result


def custom_collate_fn_codex(batch):
    codex_imgs = [item["codex"] for item in batch]
    channels = [item["channels"] for item in batch]
    region_ids = [item["region_id"] for item in batch]
    patch_ids = [item["patch_id"] for item in batch]
    return {
        "codex": codex_imgs,
        "channels": channels,
        "region_id": region_ids,
        "patch_id": patch_ids,
    }


# ---------------------------------------------------------------------------
# Pickle-based dataset classes (original format)
# ---------------------------------------------------------------------------

class BaseCodexTextDatasetPickVerion(Dataset):
    """Base class for loading CODEX datasets from pickle files."""

    def __init__(self, root_dir_codex, root_dir_he, root_dir_text, region_ids,
                 tokenizer, max_len=256, sample_json=None):
        self.root_dir_codex = root_dir_codex
        self.root_dir_he = root_dir_he
        self.root_dir_text = root_dir_text
        self.region_ids = region_ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._sample_json = sample_json

    def _load_sample_dict(self):
        if self._sample_json is not None:
            return json.load(open(self._sample_json))
        # Default fallback
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "resources", "full_codex_samples.json",
        )
        return json.load(open(default_path))

    def build_sample_index(self):
        sample_index = []
        sample_dict = self._load_sample_dict()
        for rid in self.region_ids:
            region_dir = os.path.join(self.root_dir_codex, rid)
            for sample_file in os.listdir(region_dir):
                if sample_file.endswith(".pkl"):
                    if sample_file.split(".")[0] in sample_dict.get(rid, []):
                        sample_id = os.path.splitext(sample_file)[0]
                        sample_index.append((rid, sample_id))
        return sample_index

    def build_sample_index_subset(self, random_seed=42, subset_prop=0.1):
        sample_index = []
        sample_dict = self._load_sample_dict()
        np.random.seed(random_seed)
        for rid in self.region_ids:
            region_dir = os.path.join(self.root_dir_codex, rid)
            for sample_file in os.listdir(region_dir):
                if sample_file.endswith(".pkl"):
                    if np.random.rand() < subset_prop:
                        if sample_file.split(".")[0] in sample_dict.get(rid, []):
                            sample_id = os.path.splitext(sample_file)[0]
                            sample_index.append((rid, sample_id))
        return sample_index

    def load_sample(self, region_id, sample_id):
        codex_file_path = os.path.join(self.root_dir_codex, region_id, f"{sample_id}.pkl")
        he_file_path = os.path.join(self.root_dir_he, region_id, f"{sample_id}.npy")
        text_file_path = os.path.join(self.root_dir_text, region_id, f"{sample_id}_backgroud.txt")
        try:
            with open(codex_file_path, "rb") as f:
                data = pickle.load(f)
        except EOFError:
            return None, []
        try:
            he_data = np.load(he_file_path)
            data["HandE"] = he_data
        except Exception:
            data["HandE"] = None
        try:
            with open(text_file_path, "r") as f:
                data["text"] = f.read().strip()
        except FileNotFoundError:
            data["text"] = ""
        return data, data["biomarker_name"]

    def text_processing(self, caption):
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return encoding["input_ids"].flatten(), encoding["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.sample_index)


class BaseCodexTextDatasetEmbedding(Dataset):
    """Base class for loading pre-computed CODEX embeddings."""

    def __init__(self, root_dir_codex, root_dir_he, root_dir_text, region_ids,
                 tokenizer, max_len=256, sample_json=None):
        self.root_dir_codex = root_dir_codex
        self.root_dir_he = root_dir_he
        self.root_dir_text = root_dir_text
        self.region_ids = region_ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._sample_json = sample_json

    def _load_sample_dict(self):
        if self._sample_json is not None:
            return json.load(open(self._sample_json))
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "resources", "full_codex_samples.json",
        )
        return json.load(open(default_path))

    def build_sample_index(self):
        sample_index = []
        sample_dict = self._load_sample_dict()
        for rid in self.region_ids:
            region_dir = os.path.join(self.root_dir_codex, rid)
            for sample_file in os.listdir(region_dir):
                if sample_file.endswith(".pkl"):
                    if sample_file.split(".")[0] in sample_dict.get(rid, []):
                        sample_id = os.path.splitext(sample_file)[0]
                        sample_index.append((rid, sample_id))
        return sample_index

    def build_sample_index_subset(self, random_seed=42, subset_prop=0.1):
        sample_index = []
        sample_dict = self._load_sample_dict()
        np.random.seed(random_seed)
        for rid in self.region_ids:
            region_dir = os.path.join(self.root_dir_codex, rid)
            for sample_file in os.listdir(region_dir):
                if sample_file.endswith(".pkl"):
                    if np.random.rand() < subset_prop:
                        if sample_file.split(".")[0] in sample_dict.get(rid, []):
                            sample_id = os.path.splitext(sample_file)[0]
                            sample_index.append((rid, sample_id))
        return sample_index

    def load_sample(self, region_id, sample_id):
        codex_file_path = os.path.join(self.root_dir_codex, region_id, f"{sample_id}.pkl")
        he_file_path = os.path.join(self.root_dir_he, region_id, f"{sample_id}.npy")
        text_file_path = os.path.join(self.root_dir_text, region_id, f"{sample_id}_backgroud.txt")
        data = {}
        try:
            with open(codex_file_path, "rb") as f:
                data["codex_embedding"] = pickle.load(f)
        except EOFError:
            return None, []
        try:
            he_data = np.load(he_file_path)
            data["HandE"] = he_data
        except Exception:
            data["HandE"] = None
        try:
            with open(text_file_path, "r") as f:
                data["text"] = f.read().strip()
        except FileNotFoundError:
            data["text"] = ""
        return data

    def text_processing(self, caption):
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return encoding["input_ids"].flatten(), encoding["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.sample_index)


class BaseCodexTextDatasetCodexOnly(Dataset):
    """Base class for loading CODEX-only datasets."""

    def __init__(self, root_dir_codex, region_ids, tokenizer, max_len=256,
                 codex_transform=None, sample_json=None):
        self.root_dir_codex = root_dir_codex
        self.region_ids = region_ids
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.codex_transform = codex_transform
        self._sample_json = sample_json
        self.sample_index = self.build_sample_index()
        self.crop_to_patchsize = CropToPatchSize(patch_size=16)
        self.to_grid = GridReshape(patch_size=16)

    def _load_sample_dict(self):
        if self._sample_json is not None:
            return json.load(open(self._sample_json))
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "resources", "full_codex_samples.json",
        )
        return json.load(open(default_path))

    def build_sample_index(self):
        sample_index = []
        sample_dict = self._load_sample_dict()
        for rid in self.region_ids:
            region_dir = os.path.join(self.root_dir_codex, rid)
            for sample_file in os.listdir(region_dir):
                if sample_file.endswith(".pkl"):
                    if sample_file.split(".")[0] in sample_dict.get(rid, []):
                        sample_id = os.path.splitext(sample_file)[0]
                        sample_index.append((rid, sample_id))
        return sample_index

    def build_sample_index_subset(self, random_seed=42, subset_prop=0.1):
        sample_index = []
        sample_dict = self._load_sample_dict()
        np.random.seed(random_seed)
        for rid in self.region_ids:
            region_dir = os.path.join(self.root_dir_codex, rid)
            for sample_file in os.listdir(region_dir):
                if sample_file.endswith(".pkl"):
                    if np.random.rand() < subset_prop:
                        if sample_file.split(".")[0] in sample_dict.get(rid, []):
                            sample_id = os.path.splitext(sample_file)[0]
                            sample_index.append((rid, sample_id))
        return sample_index

    def load_sample(self, region_id, sample_id):
        codex_file_path = os.path.join(self.root_dir_codex, region_id, f"{sample_id}.pkl")
        with open(codex_file_path, "rb") as f:
            data = pickle.load(f)
        return data, data["biomarker_name"]

    def text_processing(self, caption):
        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )
        return encoding["input_ids"].flatten(), encoding["attention_mask"].squeeze(0)

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        region_id, sample_id = self.sample_index[idx]
        dat, bms = self.load_sample(region_id, sample_id)
        codex_img = torch.tensor(dat["codex"], dtype=torch.float32)
        if self.codex_transform is not None:
            for transform in self.codex_transform:
                codex_img = transform(codex_img)
        codex_img = self.crop_to_patchsize(codex_img)
        codex_img = self.to_grid(codex_img)
        if "PGP9.5" in bms:
            bms[bms.index("PGP9.5")] = "PGP9_5"
        return {"codex": codex_img, "channels": bms, "region_id": region_id, "patch_id": sample_id}


class TrimodalDatasetViTPickeVerion(BaseCodexTextDatasetPickVerion):
    """Trimodal dataset loading CODEX + text + H&E from pickle files."""

    def __init__(self, root_dir_codex, root_dir_he, root_dir_text, region_ids,
                 tokenizer, max_len=256, codex_transform=None, he_transform=None,
                 text_captions=None, text_caption_sampling=None, subset_prop=-1,
                 return_raw=False, sample_json=None):
        super().__init__(root_dir_codex, root_dir_he, root_dir_text, region_ids,
                         tokenizer, max_len, sample_json=sample_json)
        self.codex_transform = codex_transform
        self.he_transform = he_transform
        self.text_captions = text_captions
        self.text_caption_sampling = text_caption_sampling
        self.crop_to_patchsize = CropToPatchSize(patch_size=16)
        self.to_grid = GridReshape(patch_size=16)

        if subset_prop > 0:
            self.sample_index = self.build_sample_index_subset(subset_prop=subset_prop, random_seed=42)
        else:
            self.sample_index = self.build_sample_index()

        self.return_raw = return_raw

    def __getitem__(self, idx):
        region_id, sample_id = self.sample_index[idx]
        dat, bms = self.load_sample(region_id, sample_id)

        if "PGP9.5" in bms:
            bms[bms.index("PGP9.5")] = "PGP9_5"

        codex_img = torch.tensor(dat["codex"], dtype=torch.float32)
        if self.codex_transform is not None:
            for transform in self.codex_transform:
                codex_img = transform(codex_img)
        codex_img_raw = codex_img.clone() if self.return_raw else None
        codex_img = self.crop_to_patchsize(codex_img)
        codex_img = self.to_grid(codex_img)

        input_ids, attention_mask = self.text_processing(dat["text"])

        try:
            he_img = torch.tensor(dat["HandE"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            if self.he_transform is not None:
                he_img = self.he_transform(he_img)
        except Exception:
            he_img = None

        if self.return_raw:
            return {
                "codex": codex_img_raw,
                "text": input_ids,
                "att_mask": attention_mask,
                "channels": bms,
                "region_id": region_id,
                "patch_id": sample_id,
                "HandE": he_img,
                "raw_text": dat["text"],
            }
        return {
            "codex": codex_img,
            "text": input_ids,
            "att_mask": attention_mask,
            "channels": bms,
            "region_id": region_id,
            "patch_id": sample_id,
            "HandE": he_img,
        }


class TrimodalDatasetViTEmbedding(BaseCodexTextDatasetEmbedding):
    """Trimodal dataset loading pre-computed CODEX embeddings + text + H&E."""

    def __init__(self, root_dir_codex, root_dir_he, root_dir_text, region_ids,
                 tokenizer, max_len=256, codex_transform=None, he_transform=None,
                 text_captions=None, text_caption_sampling=None, subset_prop=-1,
                 return_raw=False, sample_json=None):
        super().__init__(root_dir_codex, root_dir_he, root_dir_text, region_ids,
                         tokenizer, max_len, sample_json=sample_json)
        self.codex_transform = codex_transform
        self.he_transform = he_transform
        self.text_captions = text_captions
        self.text_caption_sampling = text_caption_sampling
        self.crop_to_patchsize = CropToPatchSize(patch_size=16)
        self.to_grid = GridReshape(patch_size=16)

        if subset_prop > 0:
            self.sample_index = self.build_sample_index_subset(subset_prop=subset_prop, random_seed=42)
        else:
            self.sample_index = self.build_sample_index()

        self.return_raw = return_raw

    def __getitem__(self, idx):
        region_id, sample_id = self.sample_index[idx]
        dat = self.load_sample(region_id, sample_id)

        codex_img = torch.tensor(dat["codex_embedding"], dtype=torch.float32)
        input_ids, attention_mask = self.text_processing(dat["text"])

        try:
            he_img = torch.tensor(dat["HandE"], dtype=torch.float32).permute(2, 0, 1) / 255.0
            if self.he_transform is not None:
                he_img = self.he_transform(he_img)
        except Exception:
            he_img = None

        return {
            "codex_embedding": codex_img,
            "text": input_ids,
            "att_mask": attention_mask,
            "region_id": region_id,
            "patch_id": sample_id,
            "HandE": he_img,
        }


# ---------------------------------------------------------------------------
# HDF5-based dataset classes
# ---------------------------------------------------------------------------

class LRUFileCache:
    def __init__(self, max_open_files=64):
        self.cache = OrderedDict()
        self.max_open_files = max_open_files

    def get_file(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        if len(self.cache) >= self.max_open_files:
            _, f = self.cache.popitem(last=False)
            f.close()
        f = h5py.File(path, "r")
        self.cache[path] = f
        return f


class BaseCodexTextDataset(Dataset):
    """Base class for HDF5-backed datasets."""

    def __init__(self, root_hdf5_dir, region_ids, tokenizer,
                 overlap_json, test_json, new_overlap_json, max_len=256):
        self.root_hdf5_dir = root_hdf5_dir
        self.region_ids = set(region_ids)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.master_dict = self._merge_jsons([overlap_json, test_json, new_overlap_json])
        self.sample_index = self.build_sample_index()

    def _merge_jsons(self, paths):
        merged = {}
        for p in paths:
            if os.path.exists(p):
                with open(p, "r") as f:
                    merged.update(json.load(f))
        return merged

    def build_sample_index(self):
        index = []
        for rid, samples in self.master_dict.items():
            if rid in self.region_ids:
                if os.path.exists(os.path.join(self.root_hdf5_dir, f"{rid}.h5")):
                    for sid in samples:
                        index.append((rid, sid))
        return index

    def build_sample_index_subset(self, subset_prop=0.1, seed=42):
        rng = np.random.default_rng(seed)
        full_index = self.build_sample_index()
        size = int(len(full_index) * subset_prop)
        return [full_index[i] for i in rng.choice(len(full_index), size, replace=False)]

    def text_processing(self, caption):
        return self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

    def __len__(self):
        return len(self.sample_index)


class TrimodalDatasetViT(BaseCodexTextDataset):
    """HDF5-backed trimodal dataset."""

    def __init__(self, root_hdf5_dir, region_ids, tokenizer,
                 overlap_json, test_json, new_overlap_json,
                 max_len=256, codex_transform=None, he_transform=None,
                 subset_prop=-1, return_raw=False):
        super().__init__(root_hdf5_dir, region_ids, tokenizer,
                         overlap_json, test_json, new_overlap_json, max_len)
        self.codex_transform = codex_transform
        self.he_transform = he_transform
        self.return_raw = return_raw
        self.crop_to_patchsize = CropToPatchSize(patch_size=16)
        self.to_grid = GridReshape(patch_size=16)
        if subset_prop > 0:
            self.sample_index = self.build_sample_index_subset(subset_prop)
        self.file_cache = None

    def __getitem__(self, idx):
        if self.file_cache is None:
            self.file_cache = LRUFileCache(max_open_files=8)

        region_id, sample_id = self.sample_index[idx]
        h5_path = os.path.join(self.root_hdf5_dir, f"{region_id}.h5")
        f = self.file_cache.get_file(h5_path)

        if sample_id not in f:
            raise KeyError(f"{sample_id} not found in {region_id}")

        grp = f[sample_id]

        # CODEX
        codex_arr = grp["codex"][:]
        codex_img = torch.from_numpy(codex_arr).float()
        channel_str = grp.attrs.get("channels", "[]")
        bms = json.loads(channel_str)
        if "PGP9.5" in bms:
            bms[bms.index("PGP9.5")] = "PGP9_5"

        if self.codex_transform:
            if isinstance(self.codex_transform, list):
                for t in self.codex_transform:
                    codex_img = t(codex_img)
            else:
                codex_img = self.codex_transform(codex_img)

        codex_img_raw = codex_img.clone() if self.return_raw else None
        codex_img = self.crop_to_patchsize(codex_img)
        codex_img = self.to_grid(codex_img)

        # Text
        raw_text = grp.attrs.get("text", "")
        encoded_text = self.text_processing(raw_text)
        input_ids = encoded_text["input_ids"].flatten()
        att_mask = encoded_text["attention_mask"].squeeze(0)

        # H&E
        he_img = None
        try:
            if "he" in grp:
                he_arr = grp["he"][:]
                he_img = torch.from_numpy(he_arr).float().permute(2, 0, 1) / 255.0
                if self.he_transform:
                    he_img = self.he_transform(he_img)
        except Exception:
            pass

        output = {
            "codex": codex_img,
            "text": input_ids,
            "att_mask": att_mask,
            "channels": bms,
            "region_id": region_id,
            "patch_id": sample_id,
            "HandE": he_img,
        }
        if self.return_raw:
            output["codex_raw"] = codex_img_raw
            output["raw_text"] = raw_text
        return output
