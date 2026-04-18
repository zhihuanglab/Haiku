import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, BertModel, BertConfig

from virtues.models.virtues.mae import VirTuesMAE, VirTuesEncoder

from musk import utils, modeling
from timm.models import create_model


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class TextEncoder(nn.Module):
    def __init__(
        self,
        hf_model="nomic-ai/nomic-embed-text-v1",
        freeze_bert_layers=False,
        tune_bert_layers=None,
        skip_pretrained=False,
        bert_config=None,
    ):
        """
        Args:
            hf_model: The Hugging Face model identifier for the text encoder.
            freeze_bert_layers: If True and tune_bert_layers is set, freezes all
                layers except the specified ones.
            tune_bert_layers: List of BERT layer indices to keep trainable.
            skip_pretrained: If True, build the BERT architecture from config
                without downloading pretrained weights (weights are expected to
                come from a subsequent state_dict load).
            bert_config: Optional BertConfig or path to a local config.json;
                used only when skip_pretrained=True.
        """
        super().__init__()
        if skip_pretrained:
            if bert_config is None:
                cfg = BertConfig.from_pretrained(hf_model)
            elif isinstance(bert_config, BertConfig):
                cfg = bert_config
            else:
                cfg = BertConfig.from_pretrained(str(bert_config))
            self.text_encoder = BertModel(cfg)
        else:
            self.text_encoder = BertModel.from_pretrained(hf_model)

        if freeze_bert_layers and tune_bert_layers is not None:
            self._freeze_bert_layers(tune_bert_layers)
        else:
            for param in self.text_encoder.parameters():
                param.requires_grad = True

    def _freeze_bert_layers(self, tune_bert_layers):
        for name, param in self.text_encoder.named_parameters():
            if any(f"layer.{i}" in name for i in tune_bert_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x, mask):
        out = self.text_encoder(x, mask)
        embeddings = mean_pooling(out, mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class PLIPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.plip_encoder = CLIPModel.from_pretrained("vinid/plip").vision_model.train()

    def forward(self, x):
        last_hidden_state = self.plip_encoder(x).last_hidden_state
        last_hidden_state = self.plip_encoder.post_layernorm(last_hidden_state)
        out = last_hidden_state[:, 0, :]
        return out


class MuskEncoder(nn.Module):
    def __init__(self, freeze_bert_layers=False, tune_bert_layers=None, skip_pretrained=False):
        super().__init__()
        model = create_model("musk_large_patch16_384")
        if not skip_pretrained:
            utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, "model|module", "")
        self.encoder = model

        if freeze_bert_layers and tune_bert_layers is not None:
            self._freeze_bert_layers(tune_bert_layers)
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def _freeze_bert_layers(self, tune_bert_layers):
        for name, param in self.encoder.named_parameters():
            if (any(f"layers.{i}" in name for i in tune_bert_layers)) or ("vision_head" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        return self.encoder(
            image=x,
            text_description=None,
            padding_mask=None,
            out_norm=True,
            with_head=True,
        )[0]


class CODEXEncoder(nn.Module):
    def __init__(
        self,
        codex_dim=512,
        marker_embedding=None,
        pretrained_weights_path=None,
    ):
        super().__init__()

        self.encoder = VirTuesEncoder(
            protein_emb=marker_embedding,
            patch_size=16,
            model_dim=codex_dim,
            feedforward_dim=1024,
            encoder_pattern="hvhvhvhv",
            num_encoder_heads=8,
            dropout=0.0,
            pos_emb="rope",
        )

        if pretrained_weights_path is not None:
            state_dict, _, _ = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)

            encoder_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("encoder.encoder."):
                    encoder_state_dict[k.replace("encoder.", "", 1)] = v
                elif k.startswith("encoder.protein_emb."):
                    encoder_state_dict[k.replace("encoder.", "", 1)] = v
                elif k.startswith("encoder.patch_encoder.") or k.startswith("encoder.protein_encoder."):
                    encoder_state_dict[k.replace("encoder.", "", 1)] = v
                elif k in ["encoder.patch_summary_token", "encoder.masked_token"]:
                    encoder_state_dict[k.replace("encoder.", "", 1)] = v

            self.encoder.load_state_dict(encoder_state_dict, strict=False)

    def forward(self, img, channels, mask=None):
        """
        Args:
            img: list of tensors C_i x H x W x D
            channels: list of tensors C_i
            mask: optional list of C_i x H x W masks

        Returns:
            B x D tensor of CLS tokens
        """
        _, patch_summaries = self.encoder.forward_list(img, channels, mask=mask)
        cls_token = [ps.mean(dim=(0, 1)) for ps in patch_summaries]
        return torch.stack(cls_token)
