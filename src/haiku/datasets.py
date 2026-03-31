"""Dataset builder helpers for Haiku notebooks."""

from torchvision import transforms

from .constants import HaikuPaths


def build_transforms(cfg):
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    from utils import CustomGaussianBlurTorch, PerChannelSelfStandardization

    he_transform = None
    if cfg.dataset.he_transform:
        he_transform = transforms.Compose(
            [
                transforms.Resize(384, interpolation=3, antialias=True),
                transforms.CenterCrop((384, 384)),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ]
        )

    codex_transform = None
    if cfg.dataset.codex_transform:
        codex_transform = [PerChannelSelfStandardization(), CustomGaussianBlurTorch(kernel_size=3, sigma=1.0)]
    return codex_transform, he_transform


def build_trimodal_dataset(region_ids, tokenizer, cfg, paths=None, subset_prop=-1, return_raw=False):
    from data import TrimodalDatasetViTPickeVerion

    paths = paths or HaikuPaths()
    codex_transform, he_transform = build_transforms(cfg)
    return TrimodalDatasetViTPickeVerion(
        root_dir_codex=paths.codex_root,
        root_dir_he=paths.he_root,
        root_dir_text=paths.text_root,
        region_ids=region_ids,
        tokenizer=tokenizer,
        max_len=cfg.dataset.max_length,
        codex_transform=codex_transform,
        he_transform=he_transform,
        subset_prop=subset_prop,
        return_raw=return_raw,
    )
