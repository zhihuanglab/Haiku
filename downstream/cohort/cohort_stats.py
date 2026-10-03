"""Patient-level statistics on the per-patient effect matrix theta (P x M).

patient_level_tests    one-sample two-sided Wilcoxon signed-rank across patients
                       per marker, paired d_z, patient bootstrap CI (B = 2000)
bh_fdr                 Benjamini-Hochberg q values
marker_qc              drop channels measured in < 90% of patients and channels
                       whose per-patient dispersion exceeds 3x the panel median
standardize            per-marker z-score across patients (NaN -> marker mean)
cluster_order          average-linkage clustering on correlation distance
grade_classifier       grade 3 vs grade 2 l1-logistic regression, nested CV,
                       pooled out-of-fold AUROC and label-permutation P

No patch-level P value is computed anywhere: the patient is the unit.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


# ============================================================ per-marker tests
def bh_fdr(p):
    """Benjamini-Hochberg adjusted q values (NaN-free input)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0, 1)
    return out


def patient_level_tests(E, min_patients=10, n_boot=2000, seed=0):
    """Per-marker patient-level statistics on theta.

    E: DataFrame (patients x markers).  For every marker with at least
    `min_patients` finite values: cohort mean, paired d_z = mean / sd (ddof=1),
    patient-bootstrap 95% CI of the mean (B resamples with replacement, one
    shared RNG stream walked over the markers in column order), and the
    two-sided one-sample Wilcoxon signed-rank P against zero.  q_BH is over
    the markers tested here; recompute it with `bh_fdr` after any further QC.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for mk in E.columns:
        d = E[mk].values.astype(float)
        d = d[np.isfinite(d)]
        if len(d) < min_patients:
            continue
        try:
            p = stats.wilcoxon(d, alternative='two-sided').pvalue
        except ValueError:
            p = np.nan
        sd = d.std(ddof=1)
        bs = [rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)]
        rows.append({'marker': mk, 'n_patients': int(len(d)),
                     'mean_delta': float(d.mean()),
                     'cohens_dz': float(d.mean() / sd) if sd > 0 else np.nan,
                     'ci_lo': float(np.percentile(bs, 2.5)),
                     'ci_hi': float(np.percentile(bs, 97.5)),
                     'p_wilcoxon': float(p)})
    res = pd.DataFrame(rows)
    res['q_BH'] = bh_fdr(res['p_wilcoxon'].values)
    return res


# ======================================================================= QC
def marker_qc(E, min_frac=0.9, sd_factor=3.0):
    """Channel QC on the effect matrix.

    Keeps markers with a finite effect in >= int(min_frac * P) patients, then
    drops markers whose across-patient SD (ddof=0) exceeds sd_factor x the
    median SD of the panel (measurement artefacts; CD16 and CD19 for lung).
    Returns (E_kept, dropped) with dropped a {marker: reason} dict.
    """
    dropped = {}
    keep = E.dropna(axis=1, thresh=int(min_frac * len(E)))
    for m in E.columns.difference(keep.columns):
        dropped[m] = f'finite in < {min_frac:.0%} of patients'
    sd = keep.std(ddof=0)
    bad = sd > sd_factor * sd.median()
    for m in keep.columns[bad]:
        dropped[m] = f'SD {sd[m]:.4f} > {sd_factor:g} x median {sd.median():.4f}'
    return keep.loc[:, ~bad], dropped


def standardize(E):
    """z-score each marker across patients; undefined entries -> marker mean."""
    E = E.fillna(E.mean())
    return (E - E.mean()) / E.std(ddof=0)


def cluster_order(X):
    """Average-linkage, correlation-distance ordering of rows and columns.

    Returns (patient_linkage, marker_linkage, patient_order, marker_order).
    """
    X = np.asarray(X)
    Zp = linkage(pdist(X, metric='correlation'), method='average')
    Zm = linkage(pdist(X.T, metric='correlation'), method='average')
    po = dendrogram(Zp, no_plot=True)['leaves']
    mo = dendrogram(Zm, no_plot=True)['leaves']
    return Zp, Zm, po, mo


# ============================================================ grade classifier
def _fit_l1(C, X, y, random_state=None):
    """liblinear l1-logistic.  sklearn >= 1.8 deprecates `penalty` and warns
    about l1_ratio, but still fits the l1 model; those warnings are silenced.
    liblinear visits coordinates in a random order, so `random_state` (None =
    numpy's global RNG) can move the fit slightly; see grade_classifier."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        warnings.simplefilter('ignore', UserWarning)
        return LogisticRegression(penalty='l1', solver='liblinear', C=C,
                                  max_iter=5000, random_state=random_state).fit(X, y)


def grade_classifier(X, y, Cs=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
                     outer_folds=5, inner_folds=3, n_perm=2000, seed=0,
                     feature_names=None, liblinear_seed=None):
    """l1-penalised logistic regression under nested cross-validation.

    Outer stratified `outer_folds`-fold split (each patient held out once);
    inside each training split an inner stratified `inner_folds`-fold search
    picks C from Cs by mean inner AUROC.  AUROC is computed ONCE on the pooled
    out-of-fold predictions; its null comes from permuting the labels n_perm
    times against those fixed predictions (P_perm = fraction >= observed).
    Coefficients are averaged over the outer folds (descriptive only).

    `seed` fixes the CV splits and the permutations; `liblinear_seed` fixes
    liblinear's coordinate order (None = numpy's global RNG, as in the original
    analysis script).  Near-ties between neighbouring C values in the inner
    search make the pooled AUROC move by ~0.01 with the liblinear seed.
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    oof = np.full(len(y), np.nan)
    coefs, chosen = [], []
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y):
        best_C, best_auc = None, -1.0
        for C in Cs:
            sc = []
            inner = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
            for i2, j2 in inner.split(X[tr], y[tr]):
                mdl = _fit_l1(C, X[tr][i2], y[tr][i2], liblinear_seed)
                try:
                    sc.append(roc_auc_score(y[tr][j2], mdl.predict_proba(X[tr][j2])[:, 1]))
                except ValueError:
                    pass
            if sc and np.mean(sc) > best_auc:
                best_C, best_auc = C, float(np.mean(sc))
        C = best_C if best_C is not None else 0.5
        chosen.append(C)
        mdl = _fit_l1(C, X[tr], y[tr], liblinear_seed)
        oof[te] = mdl.predict_proba(X[te])[:, 1]
        coefs.append(mdl.coef_[0])
    auc = roc_auc_score(y, oof)
    fpr, tpr, _ = roc_curve(y, oof)
    rng = np.random.RandomState(seed)
    null = np.array([roc_auc_score(rng.permutation(y), oof) for _ in range(n_perm)])
    coef = np.mean(coefs, axis=0)
    if feature_names is not None:
        coef = pd.Series(coef, index=list(feature_names))
    p_perm = float((null >= auc).mean()) if n_perm > 0 else np.nan
    return {'auroc': float(auc), 'p_perm': p_perm,
            'null': null, 'oof': oof, 'y': y, 'fpr': fpr, 'tpr': tpr,
            'coef': coef, 'C_chosen': chosen}
