import numpy as np
import multiprocess as mp
from scipy.stats import percentileofscore


def batch_zscores(ref_values, query_values):
    assert query_values.shape[1] == ref_values.shape[1]
    ref_means = ref_values.mean(0, keepdims=True)
    ref_stds = ref_values.std(0, keepdims=True)
    z_scores = (query_values - ref_means) / ref_stds
    assert z_scores.shape == query_values.shape
    return z_scores


def batch_percentiles(ref_values, query_values):
    assert query_values.shape[1] == ref_values.shape[1]

    def worker_fn(inputs):
        ref_ar, query_ar = inputs
        return percentileofscore(ref_ar, query_ar)

    args = [(ref_values[:, i], query_values[:, i]) for i in range(ref_values.shape[1])]
    with mp.Pool(mp.cpu_count() // 2) as pool:
        percentiles = pool.map(worker_fn, args)
    percentiles = np.stack(percentiles, 1)
    assert percentiles.shape == query_values.shape
    return percentiles


def simple_z_score_description(z_score):
    text_desc = '(None)'
    if z_score != z_score:
        return 'NaN'

    if z_score < -3:
        text_desc = 'extremely low level'
    elif -3 <= z_score < -2:
        text_desc = 'very low level'
    elif -2 <= z_score < -1:
        text_desc = 'low level'
    elif -1 <= z_score < 1:
        text_desc = 'average level'
    elif 1 <= z_score < 2:
        text_desc = 'high level'
    elif 2 <= z_score < 3:
        text_desc = 'very high level'
    else:
        text_desc = 'extremely high level'
    return text_desc


def simple_prob_description(prob):
    text_desc = '(None)'
    if prob != prob:
        return 'NaN'

    if prob < 0.1:
        text_desc = 'very low'
    elif 0.1 <= prob < 0.25:
        text_desc = 'low'
    elif 0.25 <= prob < 0.75:
        text_desc = 'average'
    elif 0.75 <= prob < 0.9:
        text_desc = 'high'
    else:
        text_desc = 'very high'
    return text_desc


def get_sliding_window_ROIs(region_id,
                            width,
                            height,
                            patch_size=128,
                            tissue_mask=None,
                            tissue_mask_coverage_ratio=0.9,
                            num_repetition=1,
                            stride_ratio=0.5):
    """Generate batch of queries for sliding window patches for a region

    Args:
        region_id (str): region ID.
        width (int): width of the image.
        height (int): height of the image.
        patch_size (int, optional): size (in pixel) of the patch.
        tissue_mask (np.ndarray, optional): 2D binary array of tissue mask.
        tissue_mask_coverage_ratio (float, optional): threshold for excluding
            non-tissue patches.
        num_repetition (int, optional): number of repetition of sliding window
            scans, each repetition will start from a different position.
        stride_ratio (float, optional): ratio of stride to patch size.

    Returns:
        list: list of tuples (x_left, x_right, y_bottom, y_top)
    """
    stride = int(patch_size * stride_ratio)
    if tissue_mask is not None:
        assert tissue_mask.shape == (height, width)

    patch_list = []
    for _ in range(num_repetition):
        x_start = np.random.randint(0, stride)
        y_start = np.random.randint(0, stride)
        x_steps = (width - x_start - patch_size) // stride + 1
        y_steps = (height - y_start - patch_size) // stride + 1

        for i_x in range(x_steps):
            for i_y in range(y_steps):
                x_left = x_start + i_x * stride
                x_right = x_left + patch_size
                y_bottom = y_start + i_y * stride
                y_top = y_bottom + patch_size

                assert 0 <= x_left < width
                assert 0 < x_right <= width
                assert 0 <= y_bottom < height
                assert 0 < y_top <= height

                # Check for tissue coverage
                if tissue_mask is not None:
                    patch_mask = tissue_mask[y_bottom:y_top, x_left:x_right]
                    if np.sign(patch_mask).sum() < patch_mask.size * tissue_mask_coverage_ratio:
                        continue

                patch_name = "{}_{}-{}-{}-{}".format(
                    region_id, x_left, x_right, y_bottom, y_top)
                patch_list.append({
                    'region_id': region_id,
                    'center_x': (x_left + x_right) // 2,
                    'center_y': (y_bottom + y_top) // 2,
                    'xrange': (x_left, x_right),
                    'yrange': (y_bottom, y_top),
                    'patch_name': patch_name,
                })
    return patch_list

import numpy as np
from skimage.util import view_as_windows

def get_sliding_window_ROIs_skimage(region_id,
                                    width,
                                    height,
                                    patch_size=128,
                                    tissue_mask=None,
                                    tissue_mask_coverage_ratio=0.9,
                                    num_repetition=1,
                                    stride_ratio=0.5):
    """
    Same behavior as original get_sliding_window_ROIs, but uses skimage-style
    vectorization instead of nested Python loops.
    """
    stride = int(patch_size * stride_ratio)
    if stride <= 0:
        raise ValueError(f"Invalid stride {stride} for patch_size={patch_size}, stride_ratio={stride_ratio}")

    if tissue_mask is not None:
        assert tissue_mask.shape == (height, width)
        mask_bin = (tissue_mask != 0).astype(np.uint8)
    else:
        mask_bin = None

    patch_list = []
    patch_area = patch_size * patch_size

    for _ in range(num_repetition):
        # random jitter like your original function
        x_start = np.random.randint(0, stride) if width  > patch_size else 0
        y_start = np.random.randint(0, stride) if height > patch_size else 0

        x_steps = (width  - x_start - patch_size) // stride + 1
        y_steps = (height - y_start - patch_size) // stride + 1
        if x_steps <= 0 or y_steps <= 0:
            continue

        xs = x_start + np.arange(x_steps) * stride  # (x_steps,)
        ys = y_start + np.arange(y_steps) * stride  # (y_steps,)

        grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")  # (y_steps, x_steps)

        x_left   = grid_x
        y_bottom = grid_y
        x_right  = x_left + patch_size
        y_top    = y_bottom + patch_size

        # flatten
        x_left_f   = x_left.ravel()
        x_right_f  = x_right.ravel()
        y_bottom_f = y_bottom.ravel()
        y_top_f    = y_top.ravel()

        if mask_bin is not None:
            # extract all patch masks in a vectorized way via view_as_windows
            # mask_windows shape: (y_steps, x_steps, patch, patch)
            mask_windows = view_as_windows(
                mask_bin[y_start:y_start + y_steps*stride + (patch_size - stride),
                         x_start:x_start + x_steps*stride + (patch_size - stride)],
                (patch_size, patch_size),
                step=stride
            )
            # sum per window → (y_steps * x_steps,)
            tissue_counts = mask_windows.reshape(-1, patch_area).sum(axis=1)
            coverage = tissue_counts.astype(np.float32) / patch_area
            valid = coverage >= tissue_mask_coverage_ratio
        else:
            valid = np.ones_like(x_left_f, dtype=bool)

        # build dicts
        for xl, xr, yb, yt, ok in zip(x_left_f, x_right_f, y_bottom_f, y_top_f, valid):
            if not ok:
                continue
            patch_name = f"{region_id}_{xl}-{xr}-{yb}-{yt}"
            patch_list.append({
                'region_id': region_id,
                'center_x': (xl + xr) // 2,
                'center_y': (yb + yt) // 2,
                'xrange': (int(xl), int(xr)),
                'yrange': (int(yb), int(yt)),
                'patch_name': patch_name,
            })

    return patch_list
