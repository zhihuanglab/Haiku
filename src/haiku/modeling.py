"""Reusable model/tokenizer setup used across Haiku notebooks."""

import os
import pickle

import torch
from omegaconf import OmegaConf
from transformers import BertTokenizer

from .constants import HaikuPaths


def load_cfg(paths=None):
    paths = paths or HaikuPaths()
    return OmegaConf.load(paths.config_path)


def load_vocab(vocab_path="/data/enable_data/new_individual_samples/final_biomarker_list.pkl"):
    vocab = pickle.load(open(vocab_path, "rb"))
    vocab[vocab == "PGP9.5"] = "PGP9_5"
    for i in range(len(vocab)):
        if "." in vocab[i]:
            vocab[i] = vocab[i].replace(".", "_")
    return vocab


def build_tokenizer(cfg):
    return BertTokenizer.from_pretrained(cfg.model.text_model)


def load_esm_embeddings(
    esm_dir="/project/zhihuanglab/common/datasets/enable_data/esm_embeddings",
):
    esm_embeddings = {}
    if not os.path.isdir(esm_dir):
        return esm_embeddings
    for pt_file in os.listdir(esm_dir):
        if not pt_file.endswith(".pt"):
            continue
        marker = pt_file.replace(".pt", "")
        try:
            esm_embeddings[marker] = torch.load(os.path.join(esm_dir, pt_file), map_location="cpu")
        except Exception:
            pass
    return esm_embeddings


def build_haiku_model(cfg, vocab, checkpoint_path=None):
    from models import Haiku, MarkerEmbedding

    esm_embeddings = load_esm_embeddings()
    known_markers = [m for m in vocab if m not in esm_embeddings]
    if not esm_embeddings:
        esm_embeddings = {f"marker_{i}": torch.randn(1152) for i in range(10)}

    marker_embedding = MarkerEmbedding(
        esm_embeddings=esm_embeddings,
        known_markers=known_markers,
        embedding_dim=1152,
        model_dim=cfg.model.codex_dim,
    )

    model = Haiku(
        hf_model=cfg.model.text_model,
        codex_dim=cfg.model.codex_dim,
        text_dim=cfg.model.text_dim,
        he_dim=cfg.model.he_dim,
        projection_dim=cfg.model.projection_dim,
        shared_projection=cfg.model.shared_projection,
        marker_embedding=marker_embedding,
        freeze_bert_layers=True,
        tune_bert_layers=[10, 11],
        freeze_he_encoder=cfg.model.freeze_he_encoder,
        freeze_codex_encoder=cfg.model.freeze_codex_encoder,
        pretrained_weights_path=cfg.model.codex_encoder_weights_path,
    )
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    return model
