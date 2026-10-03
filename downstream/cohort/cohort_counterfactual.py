"""Population-scale per-patient counterfactual inference (Methods 4.5.8).

For every patient of a cohort whose recorded value of the edited field is the
factual value (lung: survival status Deceased) we build two metadata-only
captions from that patient's OWN record:

    x0  factual         survival_status: dead,  survival: m,          "... Deceased ..."
    x1  counterfactual  survival_status: Alive, survival: round(2.4 m), "... Alive ..."

Every other field, and the H&E embedding of every query patch, is held fixed.
Each query patch of the patient is retrieved against the mIF atlas with the
fused score  s = w_he * cos(h, g) + (1 - w_he) * cos(t, g)  (top-K), and the
score-weighted prevalence of each marker among the retrieved neighbours that
measured it is read out,

    nu_{i,m} = sum_j s_ij L_jm / sum_j s_ij V_jm .

The per-patient effect theta_{p,m} is the mean over the patient's patches of
nu^(1) - nu^(0).  Statistics are computed on theta (see cohort_stats.py).

Everything here starts from PRECOMPUTED patch embeddings (H&E and mIF, as
written by examples/extract_haiku_multimodal_embeddings.py) and a precomputed
per-patch marker-positivity matrix; only the two captions per patient are
encoded on the fly, with the public Haiku text encoder
(`models.Haiku.from_pretrained`).
"""
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Caption construction mirrors the training captions / the single-patient case
# studies (notebooks 11 and 12).
SENSITIVE_FIELDS = {'position', 'no.', 'spot number', 'core id', 'core no', 'spot',
                    'patient id', 'case id'}
INVALID_VALUES = {'-', 'nan', 'none', 'n/a', 'null', ''}
ALIVE = {'alive', 'survival'}
DEAD = {'dead', 'deceased'}


# ===================================================================== atlas
def load_atlas(embeddings_dir, label_matrix_path):
    """Load the precomputed atlas.

    embeddings_dir must contain he_embedding.pt, codex_embedding.pt,
    region_label.pt and region_id_mapping.json ({region_id: integer label}).
    label_matrix_path is an .npz with
        L        (N_patches x N_markers) float, binary positivity, NaN where the
                 marker was not measured in that patch's acquisition
        markers  (N_markers,) marker names
    whose rows are aligned with the embedding rows.
    """
    embeddings_dir = str(embeddings_dir)
    mapping = json.load(open(os.path.join(embeddings_dir, 'region_id_mapping.json')))
    label_of = {k: int(v) for k, v in mapping.items()}
    he = torch.load(os.path.join(embeddings_dir, 'he_embedding.pt'), map_location='cpu')
    codex = torch.load(os.path.join(embeddings_dir, 'codex_embedding.pt'), map_location='cpu')
    rl = torch.load(os.path.join(embeddings_dir, 'region_label.pt'),
                    map_location='cpu').reshape(-1).long()
    z = np.load(str(label_matrix_path), allow_pickle=True)
    L = z['L'].astype(np.float32)
    markers = [str(m) for m in z['markers']]
    if L.shape[0] != he.shape[0]:
        raise ValueError(f'label matrix has {L.shape[0]} rows but the embeddings '
                         f'have {he.shape[0]}; they must be row-aligned')
    return {'he': he, 'codex': codex, 'region_label': rl,
            'label_of': label_of, 'region_of': {v: k for k, v in label_of.items()},
            'L': L, 'markers': markers}


def build_label_matrix(patch_order, label_dir, markers=None, out_path=None):
    """Build the (N_patches x N_markers) positivity matrix from per-patch JSONs.

    patch_order: list of (region_id, patch_id) in the SAME order as the
    embedding rows (the dataset's `sample_index` used when the embeddings were
    extracted).  Each patch has <label_dir>/<region_id>/<patch_id>_backgroud.json
    mapping marker -> 0/1; markers absent from that JSON stay NaN.
    """
    if markers is None:
        ms = set()
        for rid, pid in patch_order[:2000]:
            f = os.path.join(label_dir, rid, f'{pid}_backgroud.json')
            if os.path.exists(f):
                ms |= set(json.load(open(f)).keys())
        markers = sorted(ms)
    idx = {m: i for i, m in enumerate(markers)}
    L = np.full((len(patch_order), len(markers)), np.nan, dtype=np.float32)
    for i, (rid, pid) in enumerate(patch_order):
        f = os.path.join(label_dir, rid, f'{pid}_backgroud.json')
        if not os.path.exists(f):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in d.items():
            j = idx.get(k)
            if j is not None:
                try:
                    L[i, j] = float(v)
                except (TypeError, ValueError):
                    pass
    if out_path is not None:
        np.savez_compressed(out_path, L=L, markers=np.array(markers),
                            region_ids=np.array([r for r, _ in patch_order]),
                            patch_ids=np.array([p for _, p in patch_order]))
    return L, list(markers)


# ================================================================== metadata
def read_region_metadata(metadata_dir, region_ids):
    """{region_id: {field: value}} from <metadata_dir>/<region>.metadata.csv."""
    meta = {}
    for a in region_ids:
        meta[a] = {}
        for cand in (a, a.split('_')[0]):
            p = os.path.join(str(metadata_dir), f'{cand}.metadata.csv')
            if os.path.exists(p):
                d = pd.read_csv(p)
                meta[a] = {str(k).strip(): str(v).strip()
                           for k, v in zip(d['FEATURE_NAME'], d['FEATURE_VALUE'])}
                break
    return meta


def caption(m, override=None):
    """Metadata-only caption in the training-caption format."""
    m = dict(m)
    if override:
        m.update(override)
    tis = m.get('tissue_type', 'tissue')
    dis = m.get('disease', 'disease')
    parts = []
    for k, v in m.items():
        if k.lower() in SENSITIVE_FIELDS:
            continue
        if v is None or str(v).strip().lower() in INVALID_VALUES:
            continue
        parts.append(f'{k}: {v}')
    return (f'A representative section of {tis} tissue highlighting the landscape '
            f'of {dis}. Additional clinical details include: ' + ', '.join(parts) + '.')


def survival_edit(m, ratio=2.4, field='survival_status', factual='dead',
                  counterfactual='Alive'):
    """Patient-anchored Deceased -> Alive edit of the three survival fields.

      survival_status : factual             -> counterfactual
      survival        : the patient's own m -> round(m * ratio)
      'survival status' free text: 'Deceased'/'dead' -> 'Alive', and m -> m'

    ratio = 2.4 is the single-patient edit's own ratio (25 -> 60 months), so
    the index patient of notebook 12 receives exactly the published edit.
    Returns the two override dicts (factual, counterfactual).
    """
    mo = str(m.get('survival', '')).strip()
    try:
        m0 = float(mo)
        m1 = int(round(m0 * ratio))
    except ValueError:
        m0 = m1 = None
    free = str(m.get('survival status', '')).strip()
    ova = {field: factual}
    ovb = {field: counterfactual}
    if m0 is not None:
        ova['survival'] = mo
        ovb['survival'] = str(m1)
    if free:
        ova['survival status'] = free
        fb = re.sub(r'(?i)\b(deceased|dead)\b', 'Alive', free)
        if m0 is not None:
            fb = re.sub(r'\b%s\b' % re.escape(mo), str(m1), fb)
        ovb['survival status'] = fb
    return ova, ovb


def select_cohort(meta, tissue='Lung', field='survival_status', factual_values=DEAD):
    """Regions of `tissue` whose recorded `field` is a factual value, sorted."""
    fv = {v.lower() for v in factual_values}
    return sorted(a for a, m in meta.items()
                  if m.get('tissue_type', '') == tissue
                  and str(m.get(field, '')).strip().lower() in fv)


# ============================================================== text encoder
class TextEncoder:
    """Encode captions with the Haiku text tower (models.Haiku.from_pretrained)."""

    def __init__(self, model_path='zhihuanglab/Haiku', device='cuda', max_length=200):
        from models import Haiku          # <repo>/src must be on sys.path
        self.model, self.tokenizer, _ = Haiku.from_pretrained(model_path, device=device)
        self.model.eval()
        self.device = device
        self.max_length = max_length

    def n_tokens(self, text):
        return int(self.tokenizer(text, return_tensors='pt')['input_ids'].shape[1])

    @torch.no_grad()
    def __call__(self, texts):
        enc = self.tokenizer(list(texts), padding='max_length', truncation=True,
                             max_length=self.max_length, return_tensors='pt')
        out = self.model.get_features_single_modality(
            {'text': enc['input_ids'].to(self.device),
             'att_mask': enc['attention_mask'].to(self.device)}, modality='text')
        return out.float().cpu()


# ================================================================= retrieval
class FusedReadout:
    """Fused H&E + text retrieval against the mIF atlas, read out as
    score-weighted marker prevalence.

    weight_mode 'raw'     : weights are the fused scores s_ij themselves
                            (Methods 4.5.8, the published configuration)
                'softmax' : weights softmax(s_i.) over the K neighbours
    mask_self             : drop the query's own region from the gallery
                            (False in the published configuration)
    restrict_tissue       : keep only gallery patches of this tissue name
                            (None in the published configuration)
    """

    def __init__(self, atlas, w_he=0.6, top_k=50, weight_mode='raw',
                 mask_self=False, restrict_tissue=None, region_tissue=None,
                 device='cuda'):
        self.a = atlas
        self.w_he, self.top_k = float(w_he), int(top_k)
        self.weight_mode, self.mask_self = weight_mode, bool(mask_self)
        self.device = device
        self.g = F.normalize(atlas['codex'].float(), dim=1).to(device)
        self.rl = atlas['region_label'].to(device)
        L = atlas['L']
        self.Lg = torch.as_tensor(np.nan_to_num(L, nan=0.0), device=device)
        self.Vg = torch.as_tensor((~np.isnan(L)).astype(np.float32), device=device)
        self.keep = None
        if restrict_tissue is not None:
            tis = np.array([region_tissue.get(atlas['region_of'][int(l)], '')
                            for l in atlas['region_label'].tolist()])
            self.keep = torch.as_tensor(tis == restrict_tissue, device=device)
        self.gallery_tissue = None
        if region_tissue is not None:
            self.gallery_tissue = np.array(
                [region_tissue.get(atlas['region_of'][int(l)], '')
                 for l in atlas['region_label'].tolist()])

    @torch.no_grad()
    def __call__(self, he_rows, text_emb, query_label):
        h = F.normalize(self.a['he'][he_rows].float(), dim=1).to(self.device)
        t = F.normalize(text_emb.float(), dim=1).to(self.device).expand(len(he_rows), -1)
        sim = self.w_he * (h @ self.g.T) + (1 - self.w_he) * (t @ self.g.T)
        if self.mask_self:
            sim[:, self.rl == query_label] = -torch.inf
        pool = sim.shape[1]
        if self.keep is not None:
            sim[:, ~self.keep] = -torch.inf
            pool = int(self.keep.sum())
        s, i = torch.topk(sim, min(self.top_k, pool - 1), dim=1)
        w = s if self.weight_mode == 'raw' else torch.softmax(s, dim=1)
        num = torch.einsum('qk,qkm->qm', w, self.Lg[i])
        den = torch.einsum('qk,qkm->qm', w, self.Vg[i])
        prev = torch.where(den > 0, num / den, torch.full_like(num, float('nan')))
        return prev.cpu().numpy(), i.cpu().numpy()


# ===================================================================== run
def run_cohort(atlas, meta, cohort, encoder, readout, ratio=2.4,
               field='survival_status', factual='dead', counterfactual='Alive',
               min_patches=5, max_length=200, verbose=True):
    """Per-patient survival edit for every region in `cohort`.

    Returns a dict with
      effect          (P x M) theta_{p,m}: mean over patches of nu^(1) - nu^(0)
      control,
      counterfactual  (P x M) patch-mean prevalence under each caption
      patients, markers, n_patches, captions, provenance (DataFrame), skipped
    """
    rl = atlas['region_label']
    markers = atlas['markers']
    eff, ctrl, pert, ids, npatch, caps, prov, skipped = [], [], [], [], [], [], [], []
    for ci, acq in enumerate(cohort):
        lab = atlas['label_of'][acq]
        rows = (rl == lab).nonzero(as_tuple=True)[0]
        if len(rows) < min_patches:
            skipped.append((acq, f'fewer than {min_patches} patches'))
            continue
        m = meta[acq]
        ova, ovb = survival_edit(m, ratio, field, factual, counterfactual)
        cap_a, cap_b = caption(m, ova), caption(m, ovb)
        if max(encoder.n_tokens(cap_a), encoder.n_tokens(cap_b)) > max_length:
            skipped.append((acq, 'caption exceeds the text encoder length'))
            continue
        emb = encoder([cap_a, cap_b])
        pa, ia = readout(rows, emb[0:1], lab)
        pb, ib = readout(rows, emb[1:2], lab)
        with warnings.catch_warnings():          # all-NaN marker columns -> NaN
            warnings.simplefilter('ignore', RuntimeWarning)
            eff.append(np.nanmean(pb - pa, axis=0))
            ctrl.append(np.nanmean(pa, axis=0))
            pert.append(np.nanmean(pb, axis=0))
        ids.append(acq); npatch.append(len(rows)); caps.append((cap_a, cap_b))
        rec = {'region': acq, 'n_patches': int(len(rows))}
        for arm, idx in (('ctrl', ia), ('pert', ib)):
            labs = rl.numpy()[idx.reshape(-1)]
            rec[f'{arm}_self_frac'] = float((labs == lab).mean())
            rec[f'{arm}_n_regions'] = int(len(np.unique(labs)))
            if readout.gallery_tissue is not None:
                rec[f'{arm}_same_tissue_frac'] = float(
                    (readout.gallery_tissue[idx.reshape(-1)] == m.get('tissue_type', '')).mean())
        prov.append(rec)
        if verbose and (ci + 1) % 10 == 0:
            print(f'  {ci + 1}/{len(cohort)} patients', flush=True)
    return {'effect': np.array(eff), 'control': np.array(ctrl),
            'counterfactual': np.array(pert), 'patients': np.array(ids),
            'markers': np.array(markers), 'n_patches': np.array(npatch),
            'captions': caps, 'provenance': pd.DataFrame(prov), 'skipped': skipped}


def save_effects(res, path):
    np.savez(path, effect=res['effect'], control=res['control'],
             counterfactual=res['counterfactual'], patients=res['patients'],
             markers=res['markers'], n_patches=res['n_patches'])


def load_effects(path):
    """Effect matrix as a DataFrame (patients x markers)."""
    z = np.load(str(path), allow_pickle=True)
    return pd.DataFrame(z['effect'], index=[str(x) for x in z['patients']],
                        columns=[str(x) for x in z['markers']])


def effects_frame(res):
    return pd.DataFrame(res['effect'], index=[str(x) for x in res['patients']],
                        columns=[str(x) for x in res['markers']])
