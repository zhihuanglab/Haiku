import os
from utils.utils import setup_wandb_and_config, set_seed
from utils.training_utils import cosine_scheduler_with_linear_warmup, get_params_groups, save_model, load_model_state
from dataset.imc_base import IMCDataset, UnionIMCDataset, MAEDataset
from utils.esm_utils import load_esm_embeddings
from models.virtues.helpers import custom_collate_fn, MAELoss, log_results_to_disk
from models.virtues.mae import VirTuesMAE
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from time import time
import wandb
from loguru import logger
from omegaconf import OmegaConf
from dataset.imc_dataset import get_imc_dataset, get_union_imc_datasets

def train_mae(conf):
    """
    Trains a VirTues encoder-decoder pair using masked autoencoding objective.
    Args:
        conf: OmegaConf object with all the configuration
    """
    if conf.dataset.union_list is None:
        imc_dataset = get_imc_dataset(conf, filter_channel_names=conf.dataset.filter_channels)
    else:
        imc_dataset = get_union_imc_datasets(conf, conf.dataset.union_list, filter_channel_names=None)

    esm_embeddings = load_esm_embeddings(conf)
    logger.info(f'Loaded ESM embeddings of shape {esm_embeddings.shape}')
    mae_model = VirTuesMAE(
        protein_emb=esm_embeddings,
        patch_size=conf.image_info.patch_size,
        model_dim=conf.model.dim,
        feedforward_dim=conf.model.feedforward_dim,
        encoder_pattern=conf.model.encoder_pattern,
        num_encoder_heads=conf.model.num_encoder_heads,
        mae_decoder_pattern=conf.model.decoder_pattern,
        mae_num_decoder_heads=conf.model.num_decoder_heads,
        mae_num_hidden_layers_head=conf.model.num_decoder_hidden_layers,
        dropout=conf.training.dropout,
        pos_emb=conf.model.pos_emb
    )

    logger.info(f'Model has {sum(p.numel() for p in mae_model.parameters() if p.requires_grad)} trainable parameters')
    lr_scheduler = cosine_scheduler_with_linear_warmup(conf.training.lr, conf.training.lr_end, conf.training.epochs, warmup_epochs=conf.training.warmup_epochs, start_warmup_value=0)
    wd_scheduler = cosine_scheduler_with_linear_warmup(conf.training.weight_decay, conf.training.weight_decay_end, conf.training.epochs, warmup_epochs=0)

    grad_scaler = GradScaler(enabled=conf.training.fp16)
    optimizer = torch.optim.AdamW(params=get_params_groups(mae_model))

    start_epoch = 0
    if conf.training.resume:
        logger.info(f"Resuming training from checkpoint at {conf.experiment.dir}/{conf.experiment.name}/checkpoints")
        mode_state_dict, optimizer_state_dict, epoch = torch.load(f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints/model.pt')
        mae_model.load_state_dict(mode_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        start_epoch = epoch + 1
        rng_state, cuda_rng_state, np_rng_state, random_state = torch.load(f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints/rng_state.pt')
        torch.set_rng_state(rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(random_state)

    mae_model.cuda()

    train_dataset = MAEDataset(conf, imc_dataset, split='train')
    test_dataset = MAEDataset(conf, imc_dataset, split='test')

    logger.info(f'Train dataset has {len(train_dataset)} samples')
    logger.info(f'Test dataset has {len(test_dataset)} samples')
    
    train_dataloader = DataLoader(train_dataset, batch_size=conf.training.batch_size, collate_fn=custom_collate_fn, pin_memory=True, shuffle=True, num_workers=conf.training.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.training.batch_size, collate_fn=custom_collate_fn, pin_memory=True, shuffle=False, num_workers=conf.training.num_workers)

    logger.info("Starting training")
    
    for epoch in range(start_epoch, conf.training.epochs):
        logger.info(f"Starting Epoch {epoch}")
        start = time()
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_scheduler[epoch]
            if i == 0:
                param_group['weight_decay'] = wd_scheduler[epoch]
        
        train_one_epoch(conf, mae_model, train_dataloader, optimizer, grad_scaler, epoch)
        evaluate_mse(conf, mae_model, test_dataloader, epoch)
        logger.info(f"Epoch {epoch} took {time() - start} seconds")

        if epoch > 0 and epoch % 100 == 0:
            save_model(conf, mae_model, optimizer, epoch)
            mae_model.cuda()

    logger.info("Finished training. Running final evaluations next.")
    mae_model.eval()
    save_model(conf, mae_model, optimizer, epoch)
    mae_model.cuda()

    logger.info("Finished experiment")


def train_one_epoch(conf, model, train_dataloader, optimizer, grad_scaler, epoch):
    model.train()

    mae_loss_fn = MAELoss(predict_all=conf.training.predict_all, alpha_fft=conf.training.alpha_fft)
    list_metrics = []    
    
    optimizer.zero_grad()
  
    for iteration, (img, channel_ids, mask) in enumerate(tqdm(train_dataloader)):
        img = [i.cuda() for i in img]
        channel_ids = [c.cuda() for c in channel_ids]
        mask = [m.cuda() for m in mask]

        with torch.cuda.amp.autocast(enabled=conf.training.fp16):
            out = model.forward_list(img, channel_ids, mask)

        target_img = torch.concat(img, dim=0)
        mask = torch.concat(mask, dim=0)

        loss_total, mae_metrics = mae_loss_fn(out, target_img, mask)
        list_metrics.append(mae_metrics)

        loss_total = loss_total / conf.training.grad_accumulation
        
        grad_scaler.scale(loss_total).backward()
        if iteration % conf.training.grad_accumulation == 0 or iteration == len(train_dataloader) - 1:
            if conf.training.clip_grad:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.training.clip_grad)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

    avg_metrics = {"train_"+k: sum([m[k] for m in list_metrics]) / len(list_metrics) for k in list_metrics[0].keys()}
    avg_metrics["epoch"] = epoch
    logger.info(avg_metrics)
    wandb.log(avg_metrics)
    log_results_to_disk(avg_metrics, conf)

def evaluate_mse(conf, model, test_dataloader, epoch):
    model.eval()
    mae_loss_fn = MAELoss(predict_all=conf.training.predict_all, alpha_fft=conf.training.alpha_fft)
    list_metrics = []

    for (img, channel_ids, mask) in test_dataloader:
        img = [i.cuda() for i in img]
        channel_ids = [c.cuda() for c in channel_ids]
        mask = [m.cuda() for m in mask]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=conf.training.fp16):
                out = model.forward_list(img, channel_ids, mask)
            target_img = torch.concat(img, dim=0)
            mask = torch.concat(mask, dim=0)
            loss_total, mae_metrics = mae_loss_fn(out, target_img, mask)
        
        list_metrics.append(mae_metrics)


    avg_metrics = {"test_"+k: sum([m[k] for m in list_metrics]) / len(list_metrics) for k in list_metrics[0].keys()}
    avg_metrics["epoch"] = epoch
    wandb.log(avg_metrics)
    log_results_to_disk(avg_metrics, conf)
    
if __name__ == "__main__":

    conf = OmegaConf.load("configs/base_config.yaml")

    cli_conf = OmegaConf.from_cli()

    if hasattr(cli_conf, 'base') and hasattr(cli_conf.base, 'additional_config') and cli_conf.base.additional_config is not None:
        additional_conf = OmegaConf.load(cli_conf.base.additional_config)
        conf = OmegaConf.merge(conf, additional_conf)

    conf = OmegaConf.merge(conf, cli_conf)

    if conf.dataset.union_list is not None and conf.dataset.filter_channels is not None:
        raise ValueError("Cannot use multiple datasets with filter channels at the same time")

    os.makedirs(conf.experiment.dir, exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}', exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}/checkpoints', exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}/wandb', exist_ok=True)
    os.makedirs(f'{conf.experiment.dir}/{conf.experiment.name}/logs', exist_ok=True)

    logger.add(f'{conf.experiment.dir}/{conf.experiment.name}/logs/train.log')
    
    logger.info(f"Starting experiment {conf.experiment.name}")

    conf = setup_wandb_and_config(conf)

    if conf.dataset.filter_channels is not None and isinstance(conf.dataset.filter_channels, str):
        conf.dataset.filter_channels = [conf.dataset.filter_channels]
    logger.info(OmegaConf.to_yaml(conf))
    if conf.dataset.filter_channels is not None:
        logger.info(f'Using {len(conf.dataset.filter_channels)} genes')
        logger.info(f'Gene set: {conf.dataset.filter_channels}')
    
    set_seed(conf.training.seed)
    train_mae(conf)
