import os
import numpy as np
import tifffile
from multiprocess import Pool, shared_memory
from functools import partial

from .utils import simple_z_score_description, batch_percentiles, batch_zscores


def get_full_image_for_region(region_id, channels=[], **kwargs):
    """ Download full image for a region from EM database """
    import emdatabase as emdb
    im = []
    for c in channels:
        im.append(emdb.get_channel_im(region_id, c))
    im = np.stack(im, axis=0)
    return im, channels


def get_full_image_for_region_local(region_id, channels=[], root='data', **kwargs):
    """ Read full image for a region from disk """
    im = []
    for c in channels:
        image_path = os.path.join(root, region_id, f'{c}.tif')
        assert os.path.exists(image_path), f'Image not found: {image_path}'
        im.append(tifffile.imread(image_path))
        # print(c)
        # print(im[-1].shape)
    im = np.stack(im, axis=0)
    return im, channels


def get_tissue_mask_for_region(region_id, tissue_mask_check=None, **kwargs):
    """ Calculate tissue mask for a region """
    import emimagepatch as emip
    tissue_mask_check = emip.TissueMaskCheck('cnn', [region_id]) if tissue_mask_check is None else tissue_mask_check
    full_tissue_mask = tissue_mask_check[region_id]
    return full_tissue_mask


def get_tissue_mask_for_region_local(region_id, root='data', **kwargs):
    """ Read tissue mask for a region from disk """
    # tissue_mask_path = os.path.join(root, region_id, 'tissue_mask.tif')
    tissue_mask_path = os.path.join(root, region_id, 'foreground_mask.ome.tif')
    assert os.path.exists(tissue_mask_path), f'Tissue mask not found: {tissue_mask_path}'
    full_tissue_mask = tifffile.imread(tissue_mask_path)
    # print(full_tissue_mask.dtype)
    # print(full_tissue_mask.shape)
    # print(full_tissue_mask)
    return full_tissue_mask


def get_roi(xyrange, full_img, tissue_mask=None):
    """ Extract region of interest from full image based on xy range """
    if tissue_mask is not None:
        assert full_img.shape[-2:] == tissue_mask.shape[-2:]

    x0, x1, y0, y1 = xyrange
    region_img = full_img[..., y0:y1, x0:x1]
    if tissue_mask is not None:
        region_tissue_mask = tissue_mask[..., y0:y1, x0:x1]
        return region_img, region_tissue_mask
    else:
        return region_img


def calculate_roi_signal_average(xyrange, full_slice, tissue_mask):
    """ Calculate average signal intensity in the region of interest """
    region_slice, region_tissue_mask = get_roi(
        xyrange, full_slice, tissue_mask=tissue_mask)

    region_signal_average = np.mean(region_slice[region_tissue_mask > 0])
    return region_signal_average


# def calculate_roi_signal_moran_I(xyrange, full_slice, tissue_mask):
#     from libpysal.weights import lat2W
#     from esda.moran import Moran
#     region_slice, region_tissue_mask = get_roi(
#         xyrange, full_slice, tissue_mask=tissue_mask)
#     if ((region_tissue_mask > 0).sum() / region_tissue_mask.size) < 0.95:
#         return None, None
#     w = lat2W(region_slice.shape[0], region_slice.shape[1])
#     mi = Moran(region_slice, w)
#     return mi.I, mi.p_norm


def generate_raw_bm_description_for_target_patches(xyranges,
                                                   reference_patches,
                                                   full_ref_image,
                                                   full_tissue_mask,
                                                   channel_biomarkers,
                                                   return_raw=False):
    """ Generate tabular description of raw (image-based) biomarker expression
    levels of target patches

    Args:
        xyranges (list): list of tuples (x_left, x_right, y_bottom, y_top) for
            target ROIs
        reference_patches (list): list of tuples (x_left, x_right, y_bottom, y_top)
            for reference ROIs
        full_ref_image (np.ndarray): full image of reference ROIs, shape (C, H, W)
        full_tissue_mask (np.ndarray): full tissue mask, shape (H, W)
        channel_biomarkers (list): list of biomarker names, length C
        return_raw (bool, optional): if to return raw z-scores or text description

    Returns:
        np.ndarray or list: raw z-scores or text description
    """
    if isinstance(xyranges, tuple) and len(xyranges) == 4 and all(isinstance(i, int) for i in xyranges):
        # Single patch
        xyranges = [xyranges]
    assert full_ref_image.shape[0] == len(channel_biomarkers)
    assert full_ref_image.shape[-2:] == full_tissue_mask.shape[-2:]

    shm_im = shared_memory.SharedMemory(create=True, size=full_ref_image.nbytes)
    shared_im = np.ndarray(full_ref_image.shape, dtype=full_ref_image.dtype, buffer=shm_im.buf)
    np.copyto(shared_im, full_ref_image)

    shm_tm = shared_memory.SharedMemory(create=True, size=full_tissue_mask.nbytes)
    shared_tm = np.ndarray(full_tissue_mask.shape, dtype=full_tissue_mask.dtype, buffer=shm_tm.buf)
    np.copyto(shared_tm, full_tissue_mask)

    def worker_fn(im_details, tm_details, xyr):
        im_name, im_dtype, im_shape = im_details
        tm_name, tm_dtype, tm_shape = tm_details
        _shm_im = shared_memory.SharedMemory(im_name)
        _full_ref_image = np.ndarray(im_shape, dtype=im_dtype, buffer=_shm_im.buf)

        _shm_tm = shared_memory.SharedMemory(tm_name)
        _full_tissue_mask = np.ndarray(tm_shape, dtype=tm_dtype, buffer=_shm_tm.buf)

        signals_avg = [calculate_roi_signal_average(xyr, full_slice, _full_tissue_mask)
                       for full_slice in _full_ref_image]
        _shm_im.close()
        _shm_tm.close()
        return signals_avg

    worker_func_partial = partial(
        worker_fn,
        (shm_im.name, full_ref_image.dtype, full_ref_image.shape),
        (shm_tm.name, full_tissue_mask.dtype, full_tissue_mask.shape))

    pool = Pool(8)
    reference_signals = pool.map(worker_func_partial, reference_patches)
    reference_signals = np.array(reference_signals)
    query_signals = pool.map(worker_func_partial, xyranges)
    query_signals = np.array(query_signals)
    pool.close()
    shm_im.unlink()
    shm_tm.unlink()

    zscores = batch_zscores(reference_signals, query_signals)
    percentiles = batch_percentiles(reference_signals, query_signals)

    if return_raw:
        return zscores
    else:
        text_descs = []
        for i in range(len(xyranges)):
            text_desc = '|Biomarker Name|Z-Score|Percentile|Description|\n| --- | --- | --- | --- |\n'
            for j, bm in enumerate(channel_biomarkers):
                z = zscores[i, j]
                z_desc = simple_z_score_description(z)
                percentile = percentiles[i, j]
                text_desc += '|{}|{:.1f}|{:.0f}|{}|\n'.format(bm, z, percentile, z_desc)
            text_descs.append(text_desc)
        return text_descs


def generate_raw_bm_description_for_target_patches_single_thread(xyranges,
                                                                 reference_patches,
                                                                 full_ref_image,
                                                                 full_tissue_mask,
                                                                 channel_biomarkers,
                                                                 return_raw=False):
    """ Generate tabular description of raw (image-based) biomarker expression
    levels of target patches (single-thread version for debugging)

    Args:
        xyranges (list): list of tuples (x_left, x_right, y_bottom, y_top) for
            target ROIs
        reference_patches (list): list of tuples (x_left, x_right, y_bottom, y_top)
            for reference ROIs
        full_ref_image (np.ndarray): full image of reference ROIs, shape (C, H, W)
        full_tissue_mask (np.ndarray): full tissue mask, shape (H, W)
        channel_biomarkers (list): list of biomarker names, length C
        return_raw (bool, optional): if to return raw z-scores or text description

    Returns:
        np.ndarray or list: raw z-scores or text description
    """
    if isinstance(xyranges, tuple) and len(xyranges) == 4 and all(isinstance(i, int) for i in xyranges):
        # Single patch
        xyranges = [xyranges]
    assert full_ref_image.shape[0] == len(channel_biomarkers)
    assert full_ref_image.shape[-2:] == full_tissue_mask.shape[-2:]

    def worker_fn(xyr):
        signals_avg = [calculate_roi_signal_average(xyr, full_slice, full_tissue_mask) for full_slice in full_ref_image]
        return signals_avg

    reference_signals = np.array([worker_fn(xyr) for xyr in reference_patches])
    query_signals = np.array([worker_fn(xyr) for xyr in xyranges])

    zscores = batch_zscores(reference_signals, query_signals)
    percentiles = batch_percentiles(reference_signals, query_signals)

    if return_raw:
        return zscores
    else:
        text_descs = []
        for i in range(len(xyranges)):
            text_desc = '|Biomarker Name|Z-Score|Percentile|Description|\n| --- | --- | --- | --- |\n'
            for j, bm in enumerate(channel_biomarkers):
                z = zscores[i, j]
                z_desc = simple_z_score_description(z)
                percentile = percentiles[i, j]
                text_desc += '|{}|{:.1f}|{:.0f}|{}|\n'.format(bm, z, percentile, z_desc)
            text_descs.append(text_desc)
        return text_descs
