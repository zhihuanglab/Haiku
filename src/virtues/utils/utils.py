import random
import numpy as np
import torch
import os
import wandb
import pickle

def set_seed(seed):
    """
    Fixes seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def create_folder_if_not_exists(path : str):
    """
    Creates a folder if it does not exist
    """
    if not os.path.exists(path):
        os.makedirs(path)

def save_config(conf):
    """
    Saves the config file for a given checkpoint name as a pickeled dictionnary.
    """
    with open(f'{conf.experiment.dir}/{conf.experiment.name}/config.pkl', 'wb') as f:
        pickle.dump(conf, f)


def load_config(conf):
    """
    Loads a config file for given  name. If no file is found, returns an empty dictionary.
    """
    assert os.path.exists(f'{conf.experiment.dir}/{conf.experiment.name}/config.pkl'), f"No config file found for {conf.experiment.name}."

    with open(f'{conf.experiment.dir}/{conf.experiment.name}/config.pkl', 'rb') as f:
        config = pickle.load(f)
    return config

def setup_wandb_and_config(conf):
    """
    Sets up wandb and saves the config file.
    """

    assert conf.experiment.disable_wandb in ["online", "offline", "disabled"], f"Received {conf.experiment.disable_wandb} for disable_wandb. Must be one of ['online', 'offline', 'disabled']"

    project = conf.experiment.wandb_project
    if conf.training.resume:
        with open(f'{conf.experiment.dir}/{conf.experiment.name}/config.pkl', 'rb') as f:
            config = pickle.load(f)

        assert config.experiment.wandb_run_id is not None, "No run_id found"
        wandb.init(project=project, entity=conf.experiment.wandb_entity, mode=conf.experiment.disable_wandb, dir=f'{conf.experiment.dir}/{conf.experiment.name}/wandb', id=config.experiment.wandb_run_id, resume="must")

    else:
        wandb.init(name=conf.experiment.wandb_name, project=project, entity=conf.experiment.wandb_entity, mode=conf.experiment.disable_wandb, dir=f'{conf.experiment.dir}/{conf.experiment.name}/wandb')
        if conf.experiment.disable_wandb == "disabled":
            conf.experiment.wandb_run_id = "debug-run"
        else:
            conf.experiment.wandb_run_id = wandb.run.id
        save_config(conf)

    return conf