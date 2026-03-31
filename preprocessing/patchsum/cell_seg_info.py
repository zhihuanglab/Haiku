import os
import numpy as np
import pandas as pd
import tifffile

from .emmorphology_copy import get_morphology_from_mask
from .utils import simple_z_score_description, batch_percentiles, batch_zscores


def get_cell_segmentation_df(region_id,
                             segmentation_version=1,
                             **kwargs):
    """ Get cell segmentation dataframe for a region """
    import emdatabase as emdb
    cell_seg_df = emdb.get_cell_segmentation_output_for_acquisition_id(
        region_id, segmentation_version, conn=emdb.connect_sf())
    return cell_seg_df


def get_cell_segmentation_df_local(region_id, root='data', **kwargs):
    """ Read cell segmentation dataframe for a region from disk """
    seg_path = os.path.join(root, f"{region_id}.cell_data.csv")
    assert os.path.exists(seg_path), f'Cell segmentation file not found: {seg_path}'
    cell_seg_df = pd.read_csv(seg_path)
    return cell_seg_df


def get_cell_morphology_df(region_id,
                           segmentation_version=1,
                           biomarker_expression_version=1,
                           **kwargs):
    """ Calculate cell morphology based on segmentation task, downloaded from EM database """
    import emdatabase as emdb
    seg_mask = emdb.get_seg_mask(region_id,
                                 segmentation_version=segmentation_version,
                                 biomarker_expression_version=biomarker_expression_version,
                                 mask='cell')
    cell_morphology_df = get_morphology_from_mask(seg_mask)
    return cell_morphology_df


def get_cell_morphology_df_local(region_id, root='data', **kwargs):
    """ Calculate cell morphology based on segmentation task, read from disk """
    
    seg_mask_tif_path = os.path.join(root, region_id, 'segmentation_mask.tif')
    # seg_mask_npy_path = os.path.join(root, region_id, 'segmentation_mask.npy')
    seg_mask_npy_path = os.path.join(root, f"{region_id}.cell_mask.npy")
    
    if os.path.exists(seg_mask_tif_path):
        seg_mask = tifffile.imread(seg_mask_tif_path)
    elif os.path.exists(seg_mask_npy_path):
        seg_mask = np.load(seg_mask_npy_path)
    else:
        raise FileNotFoundError(f'Segmentation mask file not found: {seg_mask_tif_path} or {seg_mask_npy_path}')
    
    cell_morphology_df = get_morphology_from_mask(seg_mask)
    return cell_morphology_df

def get_cells_within_roi(xyrange, seg_df):
    """ Get Cell IDs within a specified ROI """
    x0, x1, y0, y1 = xyrange
    sub_df = seg_df[(seg_df['X'] >= x0) * (seg_df['X'] <= x1) * (seg_df['Y'] >= y0) * (seg_df['Y'] <= y1)]
    return list(sub_df['CELL_ID'])


def subset_feature_df_to_roi(xyrange, feature_df, seg_df):
    """ Subset feature dataframe to a specified ROI """
    subset_cells = get_cells_within_roi(xyrange, seg_df)
    sub_feature_df = feature_df[feature_df['CELL_ID'].isin(subset_cells)]
    return sub_feature_df


def calculate_roi_cell_density(xyrange, cell_seg_df, tissue_mask, um_per_pixel=0.3775):
    """ Calculate cell density within a specified ROI """
    sub_seg_df = subset_feature_df_to_roi(xyrange, cell_seg_df, cell_seg_df)
    n_cells = sub_seg_df.shape[0]

    x0, x1, y0, y1 = xyrange
    region_tissue_mask = tissue_mask[..., y0:y1, x0:x1]
    area_size = (region_tissue_mask > 0).sum() * ((um_per_pixel * 0.001) ** 2)

    density = (n_cells / area_size) / 1000  # Thousand cells per mm^2
    return density


def calculate_roi_cell_morphology(xyrange, cell_morphology_df, cell_seg_df, um_per_pixel=0.3775):
    """ Calculate cell morphology within a specified ROI """
    sub_morphology_df = subset_feature_df_to_roi(xyrange, cell_morphology_df, cell_seg_df)
    size = np.nanmean(sub_morphology_df['Size.Area']) * um_per_pixel ** 2  # um^2
    circularity = np.nanmean(sub_morphology_df['Shape.Circularity'])
    eccentricity = np.nanmean(sub_morphology_df['Shape.Eccentricity'])
    return size, circularity, eccentricity


def generate_cell_seg_description_for_target_patches(xyranges,
                                                     reference_patches,
                                                     cell_seg_df,
                                                     cell_morphology_df,
                                                     full_tissue_mask,
                                                     um_per_pixel=0.3775):
    """ Generate tabular description of cell segmentation and morphology features

    Args:
        xyranges (list): list of tuples (x_left, x_right, y_bottom, y_top) for
            target ROIs
        reference_patches (list): list of tuples (x_left, x_right, y_bottom, y_top)
            for reference ROIs
        cell_seg_df (pd.DataFrame): cell segmentation dataframe
        cell_morphology_df (pd.DataFrame): cell morphology dataframe
        full_tissue_mask (np.ndarray): full tissue mask, shape (H, W)
        um_per_pixel (float, optional): microns per pixel

    Returns:
        list: text description
    """
    if isinstance(xyranges, tuple) and len(xyranges) == 4 and all(isinstance(i, int) for i in xyranges):
        # Single patch
        xyranges = [xyranges]

    def worker_fn(xyr):
        density = calculate_roi_cell_density(
            xyr, cell_seg_df, full_tissue_mask, um_per_pixel=um_per_pixel)
        size, circularity, eccentricity = calculate_roi_cell_morphology(
            xyr, cell_morphology_df, cell_seg_df)
        return density, size, circularity, eccentricity

    cell_feature_names = ['Cell Density (k/mm^2)', 'Cell Size (um^2)', 'Cell Circularity', 'Cell Eccentricity']
    reference_cell_features = [worker_fn(xyr) for xyr in reference_patches]
    # Avoid NaNs in reference
    reference_cell_features = np.array([f for f in reference_cell_features if np.all(np.array(f) == np.array(f))])
    query_cell_features = np.array([worker_fn(xyr) for xyr in xyranges])

    zscores = batch_zscores(reference_cell_features, query_cell_features)
    percentiles = batch_percentiles(reference_cell_features, query_cell_features)

    text_descs = []
    for i in range(len(xyranges)):
        text_desc = '|Feature Name|Value|Z-Score|Percentile|Description|\n| --- | --- | --- | --- | --- |\n'
        for j, name in enumerate(cell_feature_names):
            val = query_cell_features[i, j]
            z = zscores[i, j]
            z_desc = simple_z_score_description(z)
            percentile = percentiles[i, j]
            text_desc += '|{}|{:.2g}|{:.1f}|{:.0f}|{}|\n'.format(
                name, val, z, percentile, z_desc)
        text_descs.append(text_desc)
    return text_descs
