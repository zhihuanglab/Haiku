import torch
import torch.nn as nn


class MarkerEmbedding(nn.Module):
    """Flexible marker embedding module.

    Supports ESM embeddings (non-trainable), zero-init learnable embeddings
    for known non-protein markers (e.g. DAPI, PanCK, HistoneH3p), and
    dynamic zero-init embeddings for unseen markers.

    Args:
        esm_embeddings: Precomputed ESM embeddings, e.g. {"CD16": tensor(...)}.
        known_markers: List of known non-protein marker names.
        embedding_dim: Dimension of the embedding space.
        model_dim: Dimension to project embeddings into.
    """

    def __init__(self, esm_embeddings, known_markers=None, embedding_dim=1152, model_dim=768):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim

        self.esm_embeddings = nn.ParameterDict(
            {name: nn.Parameter(emb, requires_grad=False) for name, emb in esm_embeddings.items()}
        )

        if known_markers is None:
            known_markers = ["DAPI", "PanCK", "HistoneH3p"]
        self.learnable_embeddings = nn.ParameterDict(
            {marker: nn.Parameter(torch.zeros(embedding_dim)) for marker in known_markers}
        )

        self.dynamic_embeddings = nn.ParameterDict()
        self.proj = nn.Linear(embedding_dim, model_dim)

    def add_esm_embedding(self, marker_name, embedding):
        """Dynamically add or update a precomputed ESM embedding."""
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embedding.shape[0]}"
            )
        self.esm_embeddings[marker_name] = nn.Parameter(embedding, requires_grad=False)

    def _get_single_embedding(self, marker_name):
        if marker_name in self.esm_embeddings:
            return self.esm_embeddings[marker_name]
        if marker_name in self.learnable_embeddings:
            return self.learnable_embeddings[marker_name]
        if marker_name not in self.dynamic_embeddings:
            self.dynamic_embeddings[marker_name] = nn.Parameter(
                torch.zeros(self.embedding_dim), requires_grad=False
            ).to(next(self.parameters()).device)
        return self.dynamic_embeddings[marker_name]

    def forward(self, marker_lists):
        """
        Args:
            marker_lists: list of list of str, e.g. [["CD16", "DAPI"], ["PanCK"]]

        Returns:
            List of tensors, each of shape [num_markers, model_dim].
        """
        batch_embeddings = []

        for ch_list in marker_lists:
            row_embeddings = []
            for marker in ch_list:
                emb = self._get_single_embedding(marker)
                row_embeddings.append(emb)

            if len(row_embeddings) > 0:
                row_embeddings_tensor = torch.stack(row_embeddings, dim=0)
                row_embeddings_tensor = self.proj(row_embeddings_tensor)
            else:
                row_embeddings_tensor = torch.zeros(0, self.model_dim, device=next(self.parameters()).device)

            batch_embeddings.append(row_embeddings_tensor)

        return batch_embeddings
