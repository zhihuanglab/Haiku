import torch
import os
import pandas as pd

def load_esm_dict(conf):
    """
    Loads a dictionary mapping protein IDs to their index in the ESM embeddings matrix.
    Args:
        conf: configuration object
    Returns: 
        esm_dict: dictionary mapping protein IDs to their index in the ESM embeddings matrix
    """
    esm_model = conf.esm.name
    esm_dict = {}
    esm_folder = os.path.join(conf.esm.encoding_dir, esm_model)
    if not os.path.exists(esm_folder):
        raise ValueError(f"ESM model {esm_model} not found in {conf.esm.encoding_dir}")
    files = os.listdir(esm_folder)
    files = list(sorted(files))
    index = 0
    for file in files:
        if file.endswith(".pt"):
            protein_id = file.removesuffix(".pt")
            esm_dict[protein_id] = index
            index += 1
    return esm_dict

def load_esm_embeddings(conf):
    """
    Loads ESM embeddings as a matrix.
    Args:
        conf: configuration object
    Returns:
        embeddings: tensor of shape (num_proteins, embedding_dim)
    """
    esm_model = conf.esm.name
    esm_folder = os.path.join(conf.esm.encoding_dir, esm_model)
    files = os.listdir(esm_folder)
    files = sorted(files)
    embeddings = []
    for file in files:
        if file.endswith(".pt"):
            embeddings.append(torch.load(os.path.join(esm_folder, file)))
    embeddings = torch.stack(embeddings, dim=0)
    return embeddings