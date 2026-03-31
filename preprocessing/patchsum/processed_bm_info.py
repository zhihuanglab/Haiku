import os
import numpy as np
import pandas as pd

from .cell_seg_info import subset_feature_df_to_roi
from .utils import simple_z_score_description, simple_prob_description, batch_percentiles, batch_zscores


def get_cell_biomarker_expression_df(region_id,
                                     biomarker_expression_version=1,
                                     channels=[],
                                     **kwargs):
    """ Download cell biomarker expression table for a region from EM database """
    import emdatabase as emdb
    cell_bm_df = emdb.get_cell_biomarker_expression_for_acquisition_id(
        region_id, biomarker_expression_version, conn=emdb.connect_sf())
    cell_bm_df = cell_bm_df.set_index('CELL_ID')
    if len(channels) > 0:
        cell_bm_df = cell_bm_df[channels]

    cell_bm_df = np.arcsinh(cell_bm_df / (5 * np.quantile(cell_bm_df, 0.2, axis=0) + 1e-5))
    cell_bm_df = (cell_bm_df - cell_bm_df.mean(0)) / cell_bm_df.std(0)
    cell_bm_df = cell_bm_df.reset_index()
    return cell_bm_df


def get_cell_biomarker_expression_df_local(region_id,
                                           root='data',
                                           channels=[],
                                           **kwargs):
    """ Read cell biomarker expression table for a region from disk """
    table_file_path = os.path.join(root, f'{region_id}.expression.csv')
    assert os.path.exists(table_file_path), f'Cell biomarker expression table not found: {table_file_path}'
    cell_bm_df = pd.read_csv(table_file_path)
    cell_bm_df = cell_bm_df.set_index('CELL_ID')
    
    # Check for name_mapping in kwargs and rename columns if provided
    name_mapping = kwargs.get('name_mapping', {})
    if name_mapping:
        cell_bm_df.rename(columns=name_mapping, inplace=True)

    if len(channels) > 0:
        cell_bm_df = cell_bm_df[channels]

    cell_bm_df = np.arcsinh(cell_bm_df / (5 * np.quantile(cell_bm_df, 0.2, axis=0) + 1e-5))
    cell_bm_df = (cell_bm_df - cell_bm_df.mean(0)) / cell_bm_df.std(0)
    cell_bm_df = cell_bm_df.reset_index()
    return cell_bm_df


def calculate_roi_cell_signal_average(xyrange, cell_bm_df, cell_seg_df):
    """ Calculate average cell signal intensity in the region of interest """
    sub_bm_df = subset_feature_df_to_roi(xyrange, cell_bm_df, cell_seg_df)
    cell_signal_average = sub_bm_df.mean()
    return cell_signal_average


def generate_cell_bm_description_for_target_patches(xyranges,
                                                    reference_patches,
                                                    cell_seg_df,
                                                    cell_bm_df,
                                                    return_raw=False):
    """ Generate tabular description of cell-based biomarker expression
    levels of target patches

    Args:
        xyranges (list): list of tuples (x_left, x_right, y_bottom, y_top) for
            target ROIs
        reference_patches (list): list of tuples (x_left, x_right, y_bottom, y_top)
            for reference ROIs
        cell_seg_df (pd.DataFrame): cell segmentation table for the full region
        cell_bm_df (pd.DataFrame): cell biomarker expression table for the full region
        return_raw (bool, optional): if to return raw z-scores or text description

    Returns:
        np.ndarray or list: raw z-scores or text description
    """
    if isinstance(xyranges, tuple) and len(xyranges) == 4 and all(isinstance(i, int) for i in xyranges):
        # Single patch
        xyranges = [xyranges]
    channel_biomarkers = [bm for bm in cell_bm_df.columns if bm != 'CELL_ID']

    def worker_fn(xyr):
        cell_signal_average = calculate_roi_cell_signal_average(xyr, cell_bm_df, cell_seg_df)
        ar = [cell_signal_average[bm] for bm in channel_biomarkers]
        return ar

    reference_cell_features = np.array([worker_fn(xyr) for xyr in reference_patches])
    query_cell_features = np.array([worker_fn(xyr) for xyr in xyranges])

    # Fill nan values
    medians = np.nanmedian(reference_cell_features, axis=0)
    for i in range(len(medians)):
        reference_cell_features[np.isnan(reference_cell_features[:, i]), i] = medians[i]
        query_cell_features[np.isnan(query_cell_features[:, i]), i] = medians[i]

    zscores = batch_zscores(reference_cell_features, query_cell_features)
    percentiles = batch_percentiles(reference_cell_features, query_cell_features)

    if return_raw:
        return zscores
    else:
        text_descs = []
        for i in range(len(xyranges)):
            text_desc = '|Biomarker|Z-Score|Percentile|Description|\n| --- | --- | --- | --- |\n'
            for j, name in enumerate(channel_biomarkers):
                z = zscores[i, j]
                z_desc = simple_z_score_description(z)
                percentile = percentiles[i, j]
                text_desc += '|{}|{:.1f}|{:.0f}|{}|\n'.format(
                    name, z, percentile, z_desc)
            text_descs.append(text_desc)
        return text_descs


def generate_continuous_feature_description_for_target_patches(xyranges,
                                                               reference_patches,
                                                               cell_seg_df,
                                                               continous_signal_df,
                                                               feature_names=[],
                                                               value_type='strength'):
    """ Generate tabular description of generic continuous features

    Args:
        xyranges (list): list of tuples (x_left, x_right, y_bottom, y_top) for
            target ROIs
        reference_patches (list): list of tuples (x_left, x_right, y_bottom, y_top)
            for reference ROIs
        cell_seg_df (pd.DataFrame): cell segmentation table for the full region
        continuous_signal_df (pd.DataFrame): table of generic continuous features for the full region
        feature_names (list, optional): list of feature names
        value_type (str, optional): 'strength' or 'prob'

    Returns:
        np.ndarray or list: raw z-scores or text description
    """
    if isinstance(xyranges, tuple) and len(xyranges) == 4 and all(isinstance(i, int) for i in xyranges):
        # Single patch
        xyranges = [xyranges]

    feature_cols = [col for col in continous_signal_df.columns if col != 'CELL_ID']
    feature_names = feature_cols if len(feature_names) == 0 else feature_names
    assert len(feature_names) == len(feature_cols)

    def worker_fn(xyr):
        cell_signal_average = calculate_roi_cell_signal_average(xyr, continous_signal_df, cell_seg_df)
        ar = [cell_signal_average[col] for col in feature_cols]
        return ar
    query_cell_features = np.array([worker_fn(xyr) for xyr in xyranges])

    if value_type == 'strength':
        reference_cell_features = np.array([worker_fn(xyr) for xyr in reference_patches])
        zscores = batch_zscores(reference_cell_features, query_cell_features)
        percentiles = batch_percentiles(reference_cell_features, query_cell_features)

        text_descs = []
        for i in range(len(xyranges)):
            text_desc = '|Feature Name|Z-Score|Percentile|Description|\n| --- | --- | --- | --- |\n'
            for j in range(len(feature_cols)):
                z = zscores[i, j]
                z_desc = simple_z_score_description(z)
                percentile = percentiles[i, j]
                text_desc += '|{}|{:.1f}|{:.0f}|{}|\n'.format(
                    feature_names[j], z, percentile, z_desc)
            text_descs.append(text_desc)
    elif value_type == 'prob':
        text_descs = []
        for i in range(len(xyranges)):
            text_desc = '|Feature Name|Value|Probability Level|\n| --- | --- | --- |\n'
            for j in range(len(feature_cols)):
                prob = query_cell_features[i, j]
                prob_desc = simple_prob_description(prob)
                text_desc += '|{}|{:.1f}|{}|\n'.format(feature_names[j], prob, prob_desc)
            text_descs.append(text_desc)
    return text_descs
