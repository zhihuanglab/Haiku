import numpy as np
from .utils import get_sliding_window_ROIs


def get_reference_patch_list(width, height,
                             patch_size=128,
                             tissue_mask=None,
                             tissue_mask_coverage_ratio=0.9):
    ROI_list = get_sliding_window_ROIs(
        '', width, height, patch_size=patch_size, tissue_mask=tissue_mask,
        tissue_mask_coverage_ratio=tissue_mask_coverage_ratio,
        num_repetition=1, stride_ratio=0.25)
    patch_list = []
    for i in np.random.choice(np.arange(len(ROI_list)), (len(ROI_list) // 4,), replace=False):
        patch_list.append(ROI_list[i]['xrange'] + ROI_list[i]['yrange'])
    return patch_list
