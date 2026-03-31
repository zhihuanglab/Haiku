"""Shared evaluation utilities for Haiku example notebooks.

Provides reusable functions for kNN classification, zero-shot evaluation,
linear probing, MIL classification, and standard metric computation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings(path, device="cpu"):
    """Load a .pt embedding tensor with device mapping."""
    return torch.load(path, map_location=device)


# ---------------------------------------------------------------------------
# kNN classification
# ---------------------------------------------------------------------------

def knn_classify(query_emb, gallery_emb, gallery_labels, k=5, metric="cosine"):
    """Cosine-similarity kNN classifier.

    Args:
        query_emb: (N, D) tensor of query embeddings.
        gallery_emb: (M, D) tensor of gallery embeddings.
        gallery_labels: (M,) tensor of integer labels.
        k: number of nearest neighbours.
        metric: "cosine" or "dot".

    Returns:
        predictions: (N,) tensor of predicted labels (majority vote).
        scores: (N, num_classes) tensor of class vote proportions.
    """
    query_emb = F.normalize(query_emb.float(), p=2, dim=-1)
    gallery_emb = F.normalize(gallery_emb.float(), p=2, dim=-1)

    sim = query_emb @ gallery_emb.T
    topk_vals, topk_idx = sim.topk(k, dim=1)
    topk_labels = gallery_labels[topk_idx]

    num_classes = int(gallery_labels.max().item()) + 1
    scores = torch.zeros(query_emb.size(0), num_classes, device=query_emb.device)
    for c in range(num_classes):
        scores[:, c] = (topk_labels == c).float().sum(dim=1) / k

    predictions = scores.argmax(dim=1)
    return predictions, scores


# ---------------------------------------------------------------------------
# Zero-shot classification
# ---------------------------------------------------------------------------

def zero_shot_classify(image_emb, text_emb, labels=None):
    """Zero-shot classification via image-text cosine similarity.

    Args:
        image_emb: (N, D) image embeddings.
        text_emb: (C, D) text embeddings (one per class).
        labels: optional list of class names.

    Returns:
        predictions: (N,) tensor of predicted class indices.
        similarities: (N, C) cosine similarity matrix.
    """
    image_emb = F.normalize(image_emb.float(), p=2, dim=-1)
    text_emb = F.normalize(text_emb.float(), p=2, dim=-1)
    similarities = image_emb @ text_emb.T
    predictions = similarities.argmax(dim=1)
    return predictions, similarities


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_score=None):
    """Compute standard classification metrics.

    Args:
        y_true: ground truth labels (array-like).
        y_pred: predicted labels (array-like).
        y_score: predicted probabilities or scores (array-like), optional.

    Returns:
        dict with accuracy, f1_macro, f1_micro, and optionally auroc, auprc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
    }

    if y_score is not None:
        y_score = np.asarray(y_score)
        try:
            if y_score.ndim == 1 or y_score.shape[1] == 1:
                metrics["auroc"] = roc_auc_score(y_true, y_score.ravel())
                metrics["auprc"] = average_precision_score(y_true, y_score.ravel())
            else:
                metrics["auroc"] = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        except ValueError:
            pass

    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_barplot(results_df, metric_col, group_col="model", ax=None, title=None):
    """Standard grouped barplot for comparing models on a metric.

    Args:
        results_df: DataFrame with columns [group_col, metric_col, ...].
        metric_col: column name of the metric to plot.
        group_col: column name for grouping (x-axis).
        ax: optional matplotlib Axes.
        title: optional plot title.

    Returns:
        matplotlib Axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    groups = results_df[group_col].unique()
    values = [results_df[results_df[group_col] == g][metric_col].values[0] for g in groups]
    ax.bar(range(len(groups)), values, tick_label=groups)
    ax.set_ylabel(metric_col)
    if title:
        ax.set_title(title)
    return ax


def plot_roc_pr(y_true, y_score, ax_roc=None, ax_pr=None, label=None):
    """Plot ROC and PR curves.

    Args:
        y_true: binary ground truth labels.
        y_score: predicted probabilities.
        ax_roc: optional Axes for ROC curve.
        ax_pr: optional Axes for PR curve.
        label: legend label.

    Returns:
        (ax_roc, ax_pr) tuple.
    """
    import matplotlib.pyplot as plt

    if ax_roc is None or ax_pr is None:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    ax_roc.plot(fpr, tpr, label=f"{label} (AUC={auc_val:.3f})" if label else f"AUC={auc_val:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax_roc.set_xlabel("FPR")
    ax_roc.set_ylabel("TPR")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap_val = average_precision_score(y_true, y_score)
    ax_pr.plot(recall, precision, label=f"{label} (AP={ap_val:.3f})" if label else f"AP={ap_val:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR Curve")
    ax_pr.legend()

    return ax_roc, ax_pr


# ---------------------------------------------------------------------------
# Linear probing
# ---------------------------------------------------------------------------

def run_linear_probe(X_train, y_train, X_test, y_test, max_iter=1000, C=1.0):
    """Run logistic regression linear probe.

    Args:
        X_train, X_test: feature arrays (N, D).
        y_train, y_test: label arrays (N,).
        max_iter: max iterations for solver.
        C: inverse regularization strength.

    Returns:
        dict with accuracy, f1_macro, f1_micro.
    """
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    clf = LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs", multi_class="auto")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred)


# ---------------------------------------------------------------------------
# MIL classifier
# ---------------------------------------------------------------------------

class MILClassifier(nn.Module):
    """Attention-pooling Multiple Instance Learning classifier."""

    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.25):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (N_instances, D) tensor of instance embeddings for one bag.

        Returns:
            logits: (num_classes,) tensor.
            attention_weights: (N_instances, 1) tensor.
        """
        attn_scores = self.attention(x)
        attn_weights = F.softmax(attn_scores, dim=0)
        bag_repr = (attn_weights * x).sum(dim=0)
        logits = self.classifier(bag_repr)
        return logits, attn_weights


def train_mil_fold(
    train_bags, train_labels, val_bags, val_labels,
    input_dim, num_classes=2, epochs=50, lr=1e-3, device="cpu",
):
    """Train a MIL classifier for one cross-validation fold.

    Args:
        train_bags: list of (N_i, D) tensors.
        train_labels: list of integer labels.
        val_bags: list of (N_i, D) tensors.
        val_labels: list of integer labels.
        input_dim: embedding dimension D.
        num_classes: number of output classes.
        epochs: training epochs.
        lr: learning rate.
        device: torch device string.

    Returns:
        model: trained MILClassifier.
        val_metrics: dict with accuracy, f1_macro, etc.
    """
    model = MILClassifier(input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        indices = np.random.permutation(len(train_bags))
        for i in indices:
            bag = train_bags[i].to(device)
            label = torch.tensor(train_labels[i], dtype=torch.long, device=device)
            logits, _ = model(bag)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    preds, scores_list = [], []
    with torch.no_grad():
        for bag in val_bags:
            logits, _ = model(bag.to(device))
            prob = F.softmax(logits, dim=0)
            preds.append(logits.argmax().item())
            scores_list.append(prob.cpu().numpy())

    val_metrics = compute_metrics(val_labels, preds)
    return model, val_metrics
