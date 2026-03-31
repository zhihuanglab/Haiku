import numpy as np
import torch
import random

def constant_schedule(value, num_epochs):
    """
    Creates a constant value schedule.
    Args:
        value: the value of the schedule
        num_epochs: the number of epochs
    """
    return np.ones(num_epochs) * value

def cosine_scheduler_with_linear_warmup(base_value, final_value, num_epochs, warmup_epochs=0, start_warmup_value=0):
    """
    Creates a cosine schedule with linear warmup.
    Args:
        base_value: the base value of the schedule (after warmup)
        final_value: the final value of the schedule
        num_epochs: the number of epochs
        warmup_epochs: the number of warmup epochs
        start_warmup_value: the value at the start of the warmup
    """
    warmup = start_warmup_value + (np.arange(warmup_epochs) + 1) * (base_value - start_warmup_value) / (warmup_epochs + 1)
    decay_steps = np.pi * np.arange(num_epochs-warmup_epochs) / (num_epochs - warmup_epochs)
    cosine_decay = final_value  + 0.5 * (base_value - final_value) * (1 + np.cos(decay_steps))
    schedule = np.concatenate((warmup, cosine_decay))
    return schedule

def get_params_groups(model):
    """
    Gets parameter groups of the model for the optimizer. Separates parameters that require regularization from those that do not (bias and LayerNorm terms).
    Args:
        model (torch.nn.Module): the model
    """
    reg = []
    not_reg = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            reg.append(param)
        else:
            not_reg.append(param)
    
    return [{'params': reg}, {'params': not_reg, 'weight_decay': 0.}]

def save_model(conf, model, optimizer, epoch):
    """
    Saves the model to a file.
    Args:
        model (torch.nn.Module): the model to be saved
        name (str): the name of the file
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_to_save = model.module
    else:
        model_to_save = model

    state_dict = {k: v.cpu() for k, v in model_to_save.state_dict().items()}

    torch.save((state_dict, optimizer.state_dict(), epoch), f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints/model_{epoch}.pt')
    torch.save(
                {'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'np_rng_state': np.random.get_state(),
                'random_state': random.getstate()},
                f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints/rng_state_{epoch}.pt'
    )
    
def load_model_state(conf):
    """
    Loads a model from a file.
    Args:
        name (str): the name of the file
    Returns:
        The state dictionary of the model
    """
    state_dict, optimizer_state_dict, epoch = torch.load(f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints/model.pt')
    return state_dict