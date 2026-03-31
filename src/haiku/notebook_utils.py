"""Notebook bootstrap helpers."""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def setup_notebook(project_root="/home/yancui/Haiku", cuda_visible_devices=None):
    """Configure import paths and optional CUDA pinning for notebooks."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    src_path = str(Path(project_root) / "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    return src_path


def seed_everything(seed=42):
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
