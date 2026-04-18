import json
import os
from pathlib import Path

import torch
import torch.nn as nn

from . import encoders
from .embedding_module import MarkerEmbedding


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
        skip_pretrained=False,
        bert_config=None,
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

        self.text_encoder = encoders.TextEncoder(
            hf_model,
            freeze_bert_layers,
            tune_bert_layers,
            skip_pretrained=skip_pretrained,
            bert_config=bert_config,
        )

        self.codex_encoder = encoders.CODEXEncoder(
            codex_dim=codex_dim,
            marker_embedding=marker_embedding,
            pretrained_weights_path=None if skip_pretrained else pretrained_weights_path,
        )
        for param in self.codex_encoder.parameters():
            param.requires_grad = not freeze_codex_encoder

        self.he_encoder = encoders.MuskEncoder(
            freeze_he_layers, tune_he_layers, skip_pretrained=skip_pretrained
        )

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

    @classmethod
    def from_pretrained(cls, repo_id_or_path, device="cpu", token=None, cache_dir=None):
        """Load a Haiku model from a local directory or a Hugging Face Hub repo.

        The repo is expected to contain:
            - config.json
            - haiku_state_dict.pt
            - tokenizer/  (BiomedBERT tokenizer + config.json)
            - esm_embeddings/*.pt  (optional; state_dict already holds them)
            - vocab.pkl  (optional, for downstream use)

        Returns:
            (model, tokenizer, marker_embedding)
        """
        from transformers import BertTokenizer

        local = Path(repo_id_or_path)
        if not local.exists():
            from huggingface_hub import snapshot_download

            local = Path(
                snapshot_download(
                    repo_id=str(repo_id_or_path),
                    token=token,
                    cache_dir=cache_dir,
                )
            )

        with open(local / "config.json") as f:
            cfg = json.load(f)

        tokenizer_dir = local / "tokenizer"
        tokenizer = BertTokenizer.from_pretrained(str(tokenizer_dir))

        esm_marker_names = cfg["esm_marker_names"]
        known_markers = cfg["known_markers"]
        embedding_dim = cfg["embedding_dim"]
        marker_model_dim = cfg["marker_model_dim"]

        esm_placeholder = {name: torch.zeros(embedding_dim) for name in esm_marker_names}

        esm_dir = local / "esm_embeddings"
        if esm_dir.is_dir():
            for pt_file in esm_dir.glob("*.pt"):
                try:
                    esm_placeholder[pt_file.stem] = torch.load(
                        str(pt_file), map_location="cpu", weights_only=False
                    )
                except Exception:
                    pass

        marker_embedding = MarkerEmbedding(
            esm_placeholder,
            known_markers=known_markers,
            embedding_dim=embedding_dim,
            model_dim=marker_model_dim,
        )

        model = cls(
            hf_model=cfg["hf_model"],
            codex_dim=cfg["codex_dim"],
            text_dim=cfg["text_dim"],
            he_dim=cfg["he_dim"],
            projection_dim=cfg["projection_dim"],
            shared_projection=cfg["shared_projection"],
            marker_embedding=marker_embedding,
            freeze_bert_layers=cfg.get("freeze_bert_layers", True),
            tune_bert_layers=cfg.get("tune_bert_layers", [10, 11]),
            freeze_he_encoder=cfg.get("freeze_he_encoder", True),
            freeze_codex_encoder=cfg.get("freeze_codex_encoder", True),
            pretrained_weights_path=None,
            skip_pretrained=True,
            bert_config=str(tokenizer_dir),
        )

        state_dict = torch.load(
            str(local / "haiku_state_dict.pt"), map_location="cpu", weights_only=False
        )
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0:
            print(f"[Haiku.from_pretrained] missing keys: {len(missing)} (showing first 5): {missing[:5]}")
        if len(unexpected) > 0:
            print(f"[Haiku.from_pretrained] unexpected keys: {len(unexpected)} (showing first 5): {unexpected[:5]}")

        model.to(device)
        return model, tokenizer, marker_embedding

    def save_pretrained(self, save_dir, config):
        """Save model weights + config for later from_pretrained loading.

        The caller is responsible for also copying the tokenizer directory
        and optional extras (esm_embeddings/, vocab.pkl) into save_dir.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), save_dir / "haiku_state_dict.pt")
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
