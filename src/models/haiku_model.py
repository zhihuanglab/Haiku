import torch
import torch.nn as nn

from . import encoders


class Haiku(nn.Module):
    """Trimodal (CODEX + H&E + text) contrastive learning model.

    Encodes CODEX multiplexed imaging, H&E histology, and free-text
    descriptions into a shared embedding space via independent projection
    heads.
    """

    def __init__(
        self,
        hf_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        codex_dim=512,
        text_dim=768,
        he_dim=768,
        projection_dim=512,
        shared_projection=False,
        marker_embedding=None,
        freeze_bert_layers=False,
        tune_bert_layers=None,
        freeze_he_encoder=False,
        freeze_codex_encoder=True,
        freeze_he_layers=False,
        tune_he_layers=None,
        pretrained_weights_path="/project/zhihuanglab/yancui/full_mae_train_expt/full_dataset_0521_20250521_162114/checkpoints/model_8.pt",
    ):
        """
        Args:
            hf_model: Hugging Face model name for the BERT-based text encoder.
            codex_dim: Output dimension of the CODEX encoder.
            text_dim: Output dimension of the text encoder.
            he_dim: Output dimension of the H&E encoder.
            projection_dim: Output dimension of the projection heads.
            shared_projection: If True, use the same projection for all modalities.
            marker_embedding: Optional embedding module for marker/channel input.
            freeze_bert_layers: If True, freezes all BERT layers.
            tune_bert_layers: Indices of BERT layers to fine-tune (if not frozen).
            freeze_he_encoder: If True, freezes the H&E image encoder.
            freeze_codex_encoder: If True, freezes the CODEX image encoder.
            freeze_he_layers: If True, freezes specific H&E layers.
            tune_he_layers: Indices of H&E layers to fine-tune.
            pretrained_weights_path: Path to pretrained VirTues encoder weights.
        """
        super().__init__()

        self.text_encoder = encoders.TextEncoder(hf_model, freeze_bert_layers, tune_bert_layers)

        self.codex_encoder = encoders.CODEXEncoder(
            codex_dim=codex_dim,
            marker_embedding=marker_embedding,
            pretrained_weights_path=pretrained_weights_path,
        )
        for param in self.codex_encoder.parameters():
            param.requires_grad = not freeze_codex_encoder

        self.he_encoder = encoders.MuskEncoder(freeze_he_layers, tune_he_layers)

        if shared_projection:
            if codex_dim == text_dim == he_dim:
                shared = nn.Linear(codex_dim, projection_dim)
                self.codex_projection = shared
                self.text_projection = shared
                self.he_projection = shared
            else:
                shared = nn.Linear(text_dim, projection_dim)
                self.codex_projection = nn.Sequential(
                    nn.Linear(codex_dim, text_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.LayerNorm(text_dim),
                    shared,
                )
                self.he_projection = nn.Sequential(
                    nn.Linear(he_dim, text_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.LayerNorm(text_dim),
                    shared,
                )
                self.text_projection = shared
        else:
            self.codex_projection = nn.Sequential(
                nn.Linear(codex_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
                nn.BatchNorm1d(projection_dim),
            )
            self.text_projection = nn.Sequential(
                nn.Linear(text_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
                nn.BatchNorm1d(projection_dim),
            )
            self.he_projection = nn.Sequential(
                nn.Linear(he_dim, projection_dim),
                nn.ReLU(),
                nn.Linear(projection_dim, projection_dim),
                nn.BatchNorm1d(projection_dim),
            )

        self._initialize_weights()

    def _initialize_weights(self):
        """Applies Xavier initialization to the projection layers only."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and any(
                proj in name for proj in ["he_projection", "codex_projection", "text_projection"]
            ):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def get_features_single_modality(self, x, modality="he"):
        self.eval()

        if not isinstance(modality, str) or modality not in ["he", "codex", "text"]:
            raise ValueError("Modality must be one of ['he', 'codex', 'text']")

        encoder = getattr(self, f"{modality}_encoder")
        projection = getattr(self, f"{modality}_projection")

        if modality == "text":
            out = projection(encoder(x["text"], x["att_mask"]))
        elif modality == "codex":
            if "codex_embedding" in x:
                out = projection(x["codex_embedding"])
            else:
                out = projection(encoder(x["codex"], x["channels"]))
        elif modality == "he":
            out = projection(encoder(x["HandE"]))
        else:
            raise ValueError(f"Unknown modality: {modality}")

        return out

    def forward(self, data):
        if "codex_embedding" in data:
            codex_features = data["codex_embedding"]
        else:
            codex_features = self.codex_encoder(data["codex"], data["channels"])
        he_features = self.he_encoder(data["HandE"])
        text_features = self.text_encoder(data["text"], data["att_mask"])

        codex_features = self.codex_projection(codex_features)
        he_features = self.he_projection(he_features)
        text_features = self.text_projection(text_features)

        return {"codex": codex_features, "text": text_features, "HandE": he_features}
