import numpy as np
import warnings


import time

from concurrent.futures import ThreadPoolExecutor

def normalize_full_image(full_img, tissue_mask):
    """ Normalize full image using mask-based histogram threshold """
    normalized_img = []
    full_img = to_uint16(full_img)
    for channel_img in full_img:
        
    
        bg_signals = channel_img[np.where(tissue_mask == 0)]
        fg_signals = channel_img[np.where(tissue_mask == 1)]
        
        fg_signals = fg_signals[np.where(fg_signals < np.max(fg_signals))]

        # Define lower and upper limit for thresholding
        lower_lim = int(np.median(bg_signals)) if len(bg_signals > 0) else 0
        upper_lim = int(np.quantile(fg_signals, 0.99) * 1.1) if len(fg_signals > 0) else 65535
        upper_lim = min(max(upper_lim, lower_lim + 1000), 65535)


        norm_param = get_histogram_threshold(
            channel_img, const_upper=5000, const_lower=10, lower_lim=lower_lim, upper_lim=upper_lim)
        
        normalized_img.append(clip_and_normalize_patch(channel_img, norm_param))
        
        #print('finsihed normalize full image channel')
    normalized_img = np.round(np.stack(normalized_img, 0) * 255).astype('uint8')
    #print('finsihed normalize full image')
    return normalized_img

def normalize_full_image_fast(full_img, tissue_mask):
    full_img = to_uint16(full_img)

    C = full_img.shape[0]
    H, W = full_img.shape[1], full_img.shape[2]

    # Preallocate once: this avoids multi-GB temporary arrays!
    out = np.empty((C, H, W), dtype=np.uint8)

    for ci, channel_img in enumerate(full_img):

        bg_signals = channel_img[tissue_mask == 0]
        fg_signals = channel_img[tissue_mask == 1]

        if fg_signals.size > 0:
            max_fg = fg_signals.max()
            fg_signals = fg_signals[fg_signals < max_fg]

        lower_lim = int(np.median(bg_signals)) if bg_signals.size > 0 else 0
        upper_raw = np.quantile(fg_signals, 0.99) * 1.1 if fg_signals.size > 0 else 65535
        upper_lim = int(min(max(upper_raw, lower_lim + 1000), 65535))

        norm_param = get_histogram_threshold(
            channel_img,
            const_upper=5000,
            const_lower=10,
            lower_lim=lower_lim,
            upper_lim=upper_lim
        )

        # normalize this channel directly into uint8
        norm_ch = clip_and_normalize_patch(channel_img, norm_param)

        out[ci] = (norm_ch * 255).round().astype(np.uint8)
        
        #print('finsihed normalize full image channel')

    return out


def normalize_single_channel(channel_img, tissue_mask):
    bg_mask = tissue_mask == 0
    fg_mask = tissue_mask == 1

    bg_signals = channel_img[bg_mask]
    fg_signals = channel_img[fg_mask]

    if fg_signals.size > 0:
        max_fg = fg_signals.max()
        fg_signals = fg_signals[fg_signals < max_fg]
    else:
        max_fg = 0

    lower_lim = int(np.median(bg_signals)) if bg_signals.size > 0 else 0
    upper_raw = np.quantile(fg_signals, 0.99) * 1.1 if fg_signals.size > 0 else 65535
    upper_lim = int(min(max(upper_raw, lower_lim + 1000), 65535))

    norm_param = get_histogram_threshold(
        channel_img,
        const_upper=5000,
        const_lower=10,
        lower_lim=lower_lim,
        upper_lim=upper_lim
    )
    #print('finsihed histogram threshold')
    return clip_and_normalize_patch(channel_img, norm_param)

def normalize_full_image_multithread(full_img, tissue_mask, max_workers=32):
    full_img = to_uint16(full_img)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(lambda ch: normalize_single_channel(ch, tissue_mask), full_img)
    return (np.stack(list(results), axis=0) * 255).round().astype(np.uint8)

from concurrent.futures import ThreadPoolExecutor
import numpy as np

def normalize_single_channel_from_masks(channel_img, bg_mask, fg_mask):
    # bg_mask, fg_mask are boolean arrays same shape as channel_img
    bg_signals = channel_img[bg_mask]
    fg_signals = channel_img[fg_mask]

    if fg_signals.size > 0:
        max_fg = fg_signals.max()
        # drop the absolute max like your original code
        fg_signals = fg_signals[fg_signals < max_fg]
    else:
        max_fg = 0

    # lower / upper limits (same logic as before)
    lower_lim = int(np.median(bg_signals)) if bg_signals.size > 0 else 0
    if fg_signals.size > 0:
        upper_raw = np.quantile(fg_signals, 0.99) * 1.1
    else:
        upper_raw = 65535

    upper_lim = int(min(max(upper_raw, lower_lim + 1000), 65535))

    norm_param = get_histogram_threshold(
        channel_img,
        const_upper=5000,
        const_lower=10,
        lower_lim=lower_lim,
        upper_lim=upper_lim
    )
    return clip_and_normalize_patch(channel_img, norm_param)


def normalize_full_image_multithread_new(full_img, tissue_mask, max_workers=32):
    """
    full_img: [C, H, W] or [H, W, C] – whatever your to_uint16 expects
    tissue_mask: [H, W] binary mask
    """
    full_img = to_uint16(full_img)

    # Assume full_img is [C, H, W]. If it's [H, W, C], add a transpose here.
    # full_img = np.transpose(full_img, (2, 0, 1))

    # Precompute masks once
    bg_mask = (tissue_mask == 0)
    fg_mask = (tissue_mask == 1)

    def worker(ch):
        return normalize_single_channel_from_masks(ch, bg_mask, fg_mask)

    # Parallelize over channels
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker, full_img))

    normalized = (np.stack(results, axis=0) * 255).round().astype(np.uint8)
    return normalized


def to_uint16(img):
    """Casting images to uint16

    if original image is in uint8, upscale values by 256
    if original image is in int/float, clip values to 0-65535

    Args:
        img (array-like): image

    Raises:
        TypeError: when input image has an unsupported dtype

    Returns:
        numpy.ndarray: image in uint16
    """
    if img.dtype in [np.dtype('uint16')]:
        return img
    elif img.dtype in [np.dtype('uint8')]:
        return (img.astype('uint16') * 256)
    elif img.dtype in [np.dtype(int), np.dtype('int32'), np.dtype(float)]:
        if np.min(img) < 0 or np.max(img) > 65535:
            warnings.warn('Casting patch to uint16 will clip some values')
            img = np.clip(img, 0, 65535)
        return img.astype('uint16')
    else:
        raise TypeError("Cannot cast image of dtype %s" % str(img.dtype))


def get_histogram_threshold(img,
                            const_upper=5000,
                            const_lower=10,
                            lower_lim=0,
                            upper_lim=65535):
    """Calculate minimum/maximum of a single channel image

    Adapted from `autothreshold` of SpatialMap

    Args:
        img (array-like): single channel image of a full acquisition.
        const_upper (int, optional): high pass threshold. Defaults to 5000.
        const_lower (int, optional): low pass threshold. Defaults to 10.
        lower_lim (int, optional): lower limit for calculating histogram.
        upper_lim (int, optional): upper limit for calculating histogram.

    Returns:
        (int, int): minimum and maximum of the channel
    """
    assert img.dtype in [np.dtype('uint16')], \
        "Error: image must be cast to uint16 before patch normalization"
    cnts, vals = np.histogram(img, bins=range(lower_lim, upper_lim, 256))
    pixels = img.shape[0] * img.shape[1]
    upper = pixels / const_lower
    lower = pixels / const_upper
    crit = np.where((lower < cnts) & (cnts <= upper))[0]

    try:
        th_min = min(crit)
        th_max = max(crit)
        min_val, max_val = int(vals[th_min]), int(vals[th_max + 1])
    except Exception as e:
        print(e)
        min_val = lower_lim
        max_val = upper_lim
    return min_val, max_val


def clip_and_normalize_patch(single_channel_img, norm_param):
    """Clip and 0-1 normalize patch image

    Args:
        single_channel_img (array-like): single channel image.
        norm_param (tuple): lower and upper thresholds for patch normalization.

    Returns:
        numpy.ndarray: normalized single channel patch image
    """
    assert single_channel_img.dtype in [np.dtype('uint16')], \
        "Error: image must be cast to uint16 before patch normalization"
    assert 0 <= norm_param[0] < norm_param[1] <= 65535
    _im = np.clip(single_channel_img, norm_param[0], norm_param[1])
    _im = (_im - norm_param[0]) / (norm_param[1] - norm_param[0] + 1e-5)
    #print('finsihed clip and normalize patch')
    return _im


# def get_biomarker_signal_levels(biomarkers, full_img, tissue_mask):
#     bm_signals = {}
#     for channel_img, bm in zip(full_img, biomarkers):
#         bg_signals = channel_img[np.where(tissue_mask == 0)]
#         fg_signals = channel_img[np.where(tissue_mask == 1)]
#         fg_signals = fg_signals[np.where(fg_signals < np.max(fg_signals))]
#         bg_mean = np.nanmean(bg_signals)
#         bg_std = np.nanstd(bg_signals)
#         sd1_r = ((fg_signals > (bg_mean + bg_std)) * 1).sum() / fg_signals.size
#         sd2_r = ((fg_signals > (bg_mean + 2 * bg_std)) * 1).sum() / fg_signals.size
#         sd3_r = ((fg_signals > (bg_mean + 3 * bg_std)) * 1).sum() / fg_signals.size
#         avg = (np.nanmean(fg_signals) - bg_mean) / (bg_std + 1e-5)
#         if avg < 0.2 or sd1_r < 0.2:
#             print("%s\t\t:%.2f\t%.2f\t%.2f\t%.2f" % (bm, sd1_r, sd2_r, sd3_r, avg))
#         bm_signals[bm] = (sd1_r, sd2_r, sd3_r, avg)
#     return bm_signals
