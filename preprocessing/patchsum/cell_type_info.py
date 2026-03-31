import os
import numpy as np
import pandas as pd
from collections import Counter

from .cell_seg_info import subset_feature_df_to_roi


def get_cell_type_df(region_id, annotation_id=None, **kwargs):
    """ Get cell type dataframe for a region """
    import emdatabase as emdb
    cell_type_df = emdb.get_cell_classification_output_for_acquisition_id(
        region_id, emdb.connect_sf(), annotation_id=annotation_id, with_annotation_label=False)
    cell_type_mapping = emdb.get_cluster_id_label_mapping(annotation_id, emdb.connect_sf())
    cell_type_mapping = dict(zip(cell_type_mapping['ANNOTATION_VALUE'], cell_type_mapping['ANNOTATION_LABEL']))
    return cell_type_df, cell_type_mapping


def get_cell_type_df_local(region_id, root='data', **kwargs):
    """ Read cell type dataframe for a region from disk """
    cell_type_path = os.path.join(root, f"{region_id}.cell_types.csv")
    assert os.path.exists(cell_type_path), f'Cell type file not found: {cell_type_path}'
    cell_type_df = pd.read_csv(cell_type_path)
    ### fill nan with unannotated ###
    cell_type_df = cell_type_df.fillna({'VALUE': -1, "ANNOTATION_LABEL": 'unannotated'})
    cell_type_mapping = dict(zip(cell_type_df['VALUE'], cell_type_df['ANNOTATION_LABEL']))
    return cell_type_df, cell_type_mapping

def get_cell_type_from_csv(region_id, cell_label_df_path):
    cell_type_df = pd.read_csv(cell_label_df_path)
    # filter for region ID #
    cell_type_df = cell_type_df[cell_type_df['REGION_ID'] == region_id]
    cell_type_mapping = dict(zip(cell_type_df['VALUE'], cell_type_df['ANNOTATION_LABEL']))
    return cell_type_df, cell_type_mapping

def get_signature_biomarkers(cell_type_df, cell_bm_df):
    """ Get signature biomarkers for each cell type """
    cts = set(cell_type_df['VALUE'])
    signature_biomarkers = {}
    for ct in cts:
        cell_ids = cell_type_df[cell_type_df['VALUE'] == ct]['CELL_ID']
        bms = dict(cell_bm_df[cell_bm_df['CELL_ID'].isin(cell_ids)].mean())
        top_biomarkers = sorted([k for k in bms.keys() if k != 'CELL_ID'], key=lambda x: bms[x], reverse=True)[:2]
        signature_biomarkers[ct] = top_biomarkers
    return signature_biomarkers


def calculate_cell_type_composition(xyrange, cell_type_df, cell_seg_df, unique_cell_types):
    """ Calculate cell type composition within a specified ROI """
    sub_ct_df = subset_feature_df_to_roi(xyrange, cell_type_df, cell_seg_df)

    ct_count = Counter(sub_ct_df['VALUE'])
    comp_vec = np.zeros(len(unique_cell_types))
    for k, v in ct_count.items():
        comp_vec[unique_cell_types.index(k)] = v
    comp_vec = comp_vec / (comp_vec.sum() + 1e-5)
    return comp_vec


def generate_cell_type_description_for_target_patches(xyranges,
                                                      cell_type_df,
                                                      cell_type_mapping,
                                                      cell_seg_df,
                                                      cell_bm_df,
                                                      n_key_cell_types=3):
    """ Generate tabular description of cell type composition

    Args:
        xyranges (list): list of tuples (x_left, x_right, y_bottom, y_top) for
            target ROIs
        cell_type_df (pd.DataFrame): cell type dataframe
        cell_type_mapping (dict): cell type mapping
        cell_seg_df (pd.DataFrame): cell segmentation dataframe
        cell_bm_df (pd.DataFrame): cell biomarker expression dataframe
        n_key_cell_types (int): number of key cell types to include, 3 to 5

    Returns:
        list: text description
    """
    signature_biomarkers = get_signature_biomarkers(cell_type_df, cell_bm_df)
    unique_cell_types = sorted(cell_type_mapping.keys())
    print(unique_cell_types)

    full_range = (cell_seg_df['X'].min(), cell_seg_df['X'].max(), cell_seg_df['Y'].min(), cell_seg_df['Y'].max())
    avg_comp = calculate_cell_type_composition(full_range, cell_type_df, cell_seg_df, unique_cell_types)

    text_descs = []
    for xyr in xyranges:
        text_desc = '|Cell Type|Composition (percent)|Enrichment|Signature Biomarkers|\n| --- | --- | --- | --- |\n'
        roi_comp = calculate_cell_type_composition(xyr, cell_type_df, cell_seg_df, unique_cell_types)
        enrichment = roi_comp / (avg_comp + 1e-5)

        key_ct_inds = list(np.argsort(-roi_comp)[:n_key_cell_types])
        key_ct_inds = [i for i in key_ct_inds if roi_comp[i] > 0.05]
        for i in np.argsort(-enrichment)[:n_key_cell_types]:
            if i not in key_ct_inds and enrichment[i] > 1.5 and roi_comp[i] > 0.05:
                key_ct_inds.append(i)

        for ct_ind in key_ct_inds:
            ct_name = cell_type_mapping[unique_cell_types[ct_ind]]
            ct_comp = roi_comp[ct_ind] * 100
            enr = enrichment[ct_ind]
            sig_bms = signature_biomarkers[unique_cell_types[ct_ind]]
            text_desc += '|{}|{:.0f}|{:.1f}|{}|\n'.format(ct_name, ct_comp, enr, ', '.join(sig_bms))
        text_descs.append(text_desc)
    return text_descs
