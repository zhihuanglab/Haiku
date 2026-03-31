import torch
import torch.nn as nn
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
import json
import os
from loguru import logger

def build_selfattention_bias(split_mask, use_true_as_query=True):
    """
    split_mask: ... x S tensor of bools
    If cross_attention, False will be used as queries and True as keys and values.
    """
    if use_true_as_query:
        seq_lens = split_mask.sum(-1)
    else:
        seq_lens = (~split_mask).sum(-1)
    seq_lens = seq_lens.flatten().tolist()
    seq_lens = [x for x in seq_lens if x != 0] # Filter-out zero length sequences for self-attention
    attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seq_lens) 
    return attn_bias

def build_selfattention_bias_channel_concat(split_mask, tokens_per_sequence, use_true_as_query=True):
    """
    This builds a BlockDiagonalMask when in the final dimension of the input tensor, the channels
    of different samples are concatenated and should not attend to each other.
    split_mask: ... x S tensor of bools
    tokens_per_sequence: number of channels per sample summing up to S i.e. S_i
    """
    split_masks = split_mask.split(tokens_per_sequence, dim=-1) # list of ... x S_i tensors 
    if use_true_as_query:
        seq_lens = [split_mask.sum(-1) for split_mask in split_masks]
    else:
        seq_lens = [(~split_mask).sum(-1) for split_mask in split_masks]
    seq_lens = torch.stack(seq_lens, dim=-1)
    seq_lens = seq_lens.flatten().tolist()
    seq_lens = [x for x in seq_lens if x != 0] # Filter-out zero length sequences for self-attention
    attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=seq_lens)
    return attn_bias


def build_crossattention_bias(split_mask, use_true_as_query=True):

    if use_true_as_query:
        q_seq_lens = split_mask.sum(-1)
        kv_seq_lens = (~split_mask).sum(-1)
    else:
        q_seq_lens = (~split_mask).sum(-1)
        kv_seq_lens = split_mask.sum(-1)
    q_seq_lens = q_seq_lens.flatten().tolist()
    kv_seq_lens = kv_seq_lens.flatten().tolist()
    if 0 in q_seq_lens or 0 in kv_seq_lens:
        raise ValueError("Cross attention requires at least one 'False' element per sequence")
    attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=q_seq_lens, kv_seqlen=kv_seq_lens)
    return attn_bias

def split_batch(x, split_mask):
    """
    x: ... S x D tensor
    split_mask: ... x S tensor of bools
    Returns x_true, x_false 1 x S_total x D
    """
    x_true = x[split_mask].unsqueeze(0)
    x_false = x[~split_mask].unsqueeze(0)

    return x_true, x_false

def merge_batch(x_true, x_false, split_mask):
    """
    x_true, x_false. 1 x S x D tensor
    split_mask: ... tensor of bools
    """
    target = torch.empty(*split_mask.size(), x_true.size(-1), dtype=x_true.dtype, device=x_true.device)
    target[split_mask] = x_true.squeeze(0)
    target[~split_mask] = x_false.squeeze(0)

    return target
    
def custom_collate_fn(batch):
    img, channel_ids, mask = zip(*batch)
    return img, channel_ids, mask


class MAELoss(nn.Module):

    def __init__(self, predict_all=True, alpha_fft=0.0):
        super().__init__()
        self.predict_all = predict_all
        self.alpha_fft = alpha_fft

    def forward(self, recon, target, mask):
        """
        recon: ...xD
        target: ...xD
        mask: ...
        """
        recon_fft = torch.fft.fft(recon, dim=-1).abs()
        target_fft = torch.fft.fft(target, dim=-1).abs()

        if self.predict_all:
            loss_mse = torch.mean((recon - target) ** 2)
            loss_fft = torch.mean((recon_fft - torch.fft.fft(target, dim=-1).abs()) ** 2)
            loss_total = (1.0-self.alpha_fft) * loss_mse + self.alpha_fft * loss_fft
            loss_masked = torch.sum(mask[...,None]*(recon - target) ** 2) / (mask.sum().clamp(min=1.0) * recon.shape[-1])
        else:
            loss_mse = torch.sum(mask[...,None] * (target - recon)**2) / (mask.sum().clamp(min=1.0) * recon.shape[-1])
            loss_fft = torch.sum(mask[...,None] * torch.abs(target_fft - recon_fft)) / (mask.sum().clamp(min=1.0) * recon.shape[-1])
            loss_total = (1.0 - self.alpha_fft) * loss_mse + self.alpha_fft * loss_fft
            loss_masked = loss_mse
        
        metrics = {
            "loss_mae": loss_total.item(),
            "loss_mae_mse": loss_mse.item(),
            "loss_mae_fft": loss_fft.item(),
            "loss_mae_masked": loss_masked.item()
        }
        
        return loss_total, metrics

def load_results_from_disk(conf):

    if os.path.isfile(f'{conf.experiment.dir}/{conf.experiment.name}/results.json'):
        with open(f'{conf.experiment.dir}/{conf.experiment.name}/results.json', "r") as f:
            results = json.load(f)
            return results
    else:
        logger.info(f"Warning: No results file found for {conf.experiment.name}. Returning empty dictionary.")
        return dict()

def log_results_to_disk(results, conf):
    old_results = load_results_from_disk(conf)
    old_results.update(results)
    with open(f'{conf.experiment.dir}/{conf.experiment.name}/results.json', "w") as f:
        json.dump(old_results, f, indent=4)
    

