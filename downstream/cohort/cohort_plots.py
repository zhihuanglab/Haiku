"""Figures for the cohort counterfactual notebook (Fig. 6e-h, Supp. Fig. S8)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram

UP1, DN1, UP2, DN2, NS = '#B3123A', '#12457F', '#E88AA0', '#8FB4DC', '#C9C9C9'


def _stars(q):
    return '***' if q < 1e-3 else '**' if q < 1e-2 else '*' if q < 0.10 else ''


def plot_marker_shifts(T, n_patients, only_sig=False, q_col='q'):
    """Cohort mean shift with 95% patient-bootstrap CI (left) and paired d_z
    (right).  only_sig=True gives the Fig. 6e view (q < 0.10), False the
    all-marker Supplementary Fig. S8 view."""
    T = T.copy()
    if only_sig:
        T = T[T[q_col] < 0.10]
    T = T.sort_values('mean_delta')
    q, d = T[q_col].values, T.mean_delta.values
    t1, t2 = q < 0.05, (q >= 0.05) & (q < 0.10)
    col = np.where(t1 & (d > 0), UP1, np.where(t1, DN1,
                   np.where(t2 & (d > 0), UP2, np.where(t2, DN2, NS))))
    y = np.arange(len(T))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 0.22 * len(T) + 1.6), sharey=True,
                                 gridspec_kw={'width_ratios': [1.4, 1]})
    a1.hlines(y, T.ci_lo, T.ci_hi, color=col, lw=2)
    a1.scatter(d, y, c=col, s=26, zorder=3, edgecolors='white', linewidths=.5)
    a1.axvline(0, color='k', lw=1, ls='--')
    a1.set_yticks(y)
    a1.set_yticklabels([f'{m} {_stars(v)}' for m, v in zip(T.marker, q)], fontsize=8)
    a1.set_xlabel('shift in prevalence (Alive - Deceased)\nmean, 95% patient bootstrap CI')
    a2.barh(y, T.cohens_dz, color=col, edgecolor='k', lw=.4)
    for v in (-0.8, -0.5, 0.5, 0.8):
        a2.axvline(v, color='#888', lw=.8, ls=':')
    a2.axvline(0, color='k', lw=1)
    a2.set_xlabel('paired Cohen\'s $d_z$')
    for a in (a1, a2):
        a.spines[['top', 'right']].set_visible(False)
    sig = q < 0.10
    fig.suptitle(f'{len(T)} markers, n = {n_patients} patients; '
                 f'q<0.10: {int(sig.sum())} ({int((d[sig] > 0).sum())} up, '
                 f'{int((d[sig] < 0).sum())} down), q<0.05: {int(t1.sum())}',
                 fontsize=10, x=0.02, ha='left')
    fig.tight_layout()
    return fig


def plot_clustermap(X, markers, Zp, po, mo):
    """Standardized per-patient shift vectors, ordered by the dendrograms."""
    fig = plt.figure(figsize=(10, 9))
    gs = GridSpec(3, 1, height_ratios=[1.3, 8, .35], hspace=.05, figure=fig)
    aT = fig.add_subplot(gs[0])
    dendrogram(Zp, ax=aT, no_labels=True, color_threshold=0,
               above_threshold_color='#444')
    aT.axis('off')
    aT.set_title('Per-patient shift vectors (z-score per marker)', loc='left')
    aH = fig.add_subplot(gs[1])
    H = np.asarray(X)[np.ix_(po, mo)].T
    v = np.nanpercentile(np.abs(H), 98)
    im = aH.imshow(H, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v,
                   interpolation='nearest')
    aH.set_yticks(range(len(mo)))
    aH.set_yticklabels([markers[i] for i in mo], fontsize=6)
    aH.yaxis.tick_right()
    aH.set_xticks([])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[2]), orientation='horizontal')
    cb.set_label(f'shift (z-score); columns = {H.shape[1]} patients')
    return fig


def plot_grade_classifier(g, n_top=7):
    """Pooled out-of-fold ROC and fold-averaged Lasso coefficients."""
    y = g['y']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.plot(g['fpr'], g['tpr'], lw=2.4, color='#2C6FB5',
            label=f"AUROC {g['auroc']:.3f}\n$P_{{perm}}$ "
                  + (f"< {1 / len(g['null']):.0e}" if g['p_perm'] == 0
                     else f"= {g['p_perm']:.3f}"))
    a1.plot([0, 1], [0, 1], ls='--', color='#888')
    a1.set_xlabel('false positive rate')
    a1.set_ylabel('true positive rate')
    a1.set_title(f'grade 3 vs 2, nested CV (n={len(y)}: {int(y.sum())} G3, '
                 f'{int((1 - y).sum())} G2)', loc='left', fontsize=10)
    a1.legend(frameon=False, loc='lower right')
    cs = g['coef']
    nz = cs[cs.abs() > 1e-8].sort_values()
    top = pd.concat([nz.head(n_top), nz.tail(n_top)])
    top = top[~top.index.duplicated()]
    a2.barh(np.arange(len(top)), top.values, edgecolor='k', lw=.6,
            color=['#2C6FB5' if v < 0 else '#D1495B' for v in top.values])
    a2.set_yticks(np.arange(len(top)))
    a2.set_yticklabels(top.index, fontsize=9)
    a2.axvline(0, color='k', lw=1)
    a2.set_xlabel('mean Lasso coefficient (+ grade 3 / - grade 2)')
    a2.set_title(f'{len(nz)}/{len(cs)} non-zero, median C = '
                 f"{np.median(g['C_chosen']):.2f}", loc='left', fontsize=10)
    for a in (a1, a2):
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig
