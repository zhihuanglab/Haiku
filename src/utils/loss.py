import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import numpy as np

class PairwiseCLIPLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, constant_temperature: bool = True, region_weight: float = 0.5):
        super(PairwiseCLIPLoss, self).__init__()
        # self.temperature = temperature
        self.log_temp = nn.Parameter(torch.tensor([np.log(1/temperature)]))
        if constant_temperature:
            self.log_temp.requires_grad = False
        self.region_weight = region_weight  # Weight for negative pairs from the same region (should be < 1.0)

    def forward(self, emb_dict: dict, region_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            embeddings_dict (dict): A dictionary where keys are modality names and values are 
                                    the corresponding embeddings (Tensor) of shape (batch_size, embedding_dim).
            region_ids (torch.Tensor, optional): Tensor of shape (batch_size,) containing region IDs for each sample.
                                               If provided, will apply weighting to negative pairs from the same region.
        """
        # All pairwise combos
        modality_pairs = list(combinations(emb_dict.keys(), 2))
        total_loss = 0.0
        num_pairs = len(modality_pairs)

        pairwise_losses = {}  # Dictionary to store individual pairwise losses
        # temperature = torch.exp(self.log_temp)
        temp = 1 / self.log_temp.exp()
        temp = temp.clamp(min=0.001) 

        for (modality_a, modality_b) in modality_pairs:
            
            emb_a = emb_dict[modality_a]
            emb_b = emb_dict[modality_b]

            emb_a = F.normalize(emb_a, p=2, dim=-1)
            emb_b = F.normalize(emb_b, p=2, dim=-1)

            # CLIP is bi-directional
            logits_a_to_b = emb_a @ emb_b.t() / temp
            logits_b_to_a = emb_b @ emb_a.t() / temp

            # Labels are indices of the ground-truth matches
            batch_size = emb_a.size(0)
            labels = torch.arange(batch_size, dtype=torch.long, device=emb_a.device)

            # Apply region-based weighting if region_ids are provided
            if region_ids is not None:
                # Create a mask for pairs from the same region (but different samples)
                region_mask_a_to_b = self._create_region_mask(region_ids, batch_size, emb_a.device)
                region_mask_b_to_a = region_mask_a_to_b.t()  # Transpose for the other direction
                
                # Apply weighted cross entropy loss
                loss_a_to_b = self._weighted_cross_entropy(logits_a_to_b, labels, region_mask_a_to_b)
                loss_b_to_a = self._weighted_cross_entropy(logits_b_to_a, labels, region_mask_b_to_a)
            else:
                # Standard cross entropy without weighting
                loss_a_to_b = F.cross_entropy(logits_a_to_b, labels)
                loss_b_to_a = F.cross_entropy(logits_b_to_a, labels)

            pair_loss = (loss_a_to_b + loss_b_to_a) / 2

            # Add the average of the two losses to the total loss
            total_loss += pair_loss

            # Store individual pairwise losses
            pairwise_losses[(modality_a, modality_b)] = pair_loss

        # Average again?
        final_loss = total_loss / num_pairs if num_pairs > 0 else total_loss

        # Optionally, return pairwise_losses if needed
        return final_loss, pairwise_losses
    
    def _create_region_mask(self, region_ids, batch_size, device):
        """
        Create a mask for pairs from the same region (but different samples).
        
        Args:
            region_ids: Tensor of shape (batch_size,) containing region IDs
            batch_size: Size of the batch
            device: Device to create the mask on
            
        Returns:
            A boolean mask of shape (batch_size, batch_size) where True indicates
            pairs from the same region (excluding self-pairs on the diagonal)
        """
        # Ensure region_ids is the right shape and type
        if region_ids.shape[0] != batch_size:
            print(f"Warning: region_ids shape {region_ids.shape[0]} doesn't match batch size {batch_size}")
            # If we have more region_ids than batch_size (e.g., from gathering in multi-GPU), truncate
            if region_ids.shape[0] > batch_size:
                region_ids = region_ids[:batch_size]
            # If we have fewer region_ids than batch_size (shouldn't happen), pad with unique values
            else:
                padding = torch.arange(batch_size - region_ids.shape[0], device=device) + region_ids.max() + 1
                region_ids = torch.cat([region_ids, padding])
        
        # Expand region_ids to compare all pairs
        region_ids_expanded_i = region_ids.unsqueeze(1).expand(batch_size, batch_size)
        region_ids_expanded_j = region_ids.unsqueeze(0).expand(batch_size, batch_size)
        
        # Create mask where True indicates same region
        same_region_mask = (region_ids_expanded_i == region_ids_expanded_j)
        
        # Remove diagonal (self-pairs) from the mask
        diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        same_region_mask = same_region_mask & ~diagonal_mask
        
        return same_region_mask
    
    def _weighted_cross_entropy(self, logits, labels, region_mask):
        """
        Apply weighted cross entropy loss where negative pairs from the same region have lower weights.
        
        Args:
            logits: Tensor of shape (batch_size, batch_size) containing similarity scores
            labels: Tensor of shape (batch_size,) containing ground truth indices
            region_mask: Boolean mask of shape (batch_size, batch_size) where True indicates pairs from the same region
            
        Returns:
            Weighted cross entropy loss
        """
        batch_size = logits.size(0)
        
        # Create a weight matrix initialized with ones
        weights = torch.ones_like(logits)
        
        # Apply lower weights to negative pairs from the same region
        # We only modify weights for non-diagonal elements that are from the same region
        weights[region_mask] = self.region_weight
        
        # Keep diagonal weights as 1 (these are the positive pairs)
        diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        weights[diagonal_mask] = 1.0
        
        # Apply softmax with the weights
        exp_logits = torch.exp(logits) * weights
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        
        # Compute mean of the negative log-likelihood for the positive pairs
        nll_loss = -log_prob[torch.arange(batch_size, device=logits.device), labels]
        return nll_loss.mean()

class OneVersusAllLoss(nn.Module):
    def forward(self, emb_dict: dict) -> torch.Tensor:
        """
        Args:
            emb_dict (dict): A dictionary where keys are modality names and values are 
                             the corresponding embeddings (Tensor) of shape (batch_size, embedding_dim).
                             It must contain exactly three modalities.
        """
        modalities = list(emb_dict.keys())
        assert len(modalities) == 3, "Exactly three modalities are required."

        emb_list = [F.normalize(emb_dict[mod], p=2, dim=-1) for mod in modalities]
        batch_size = emb_list[0].size(0)
        device = emb_list[0].device
        indices = torch.arange(batch_size, device=device)

        total_loss = 0.0

        for m_idx, emb_m in enumerate(emb_list):
            # Create negative embeddings by concatenating other modalities
            other_idxs = [idx for idx in range(3) if idx != m_idx]
            neg_embs = torch.cat([emb_list[idx] for idx in other_idxs], dim=0)  # (2*batch_size, embedding_dim)
            
            # Compute logits only against other modalities
            logits = emb_m @ neg_embs.t() / self.temperature  # (batch_size, 2*batch_size)
            log_probs = F.log_softmax(logits, dim=1)

            # Create positive mask
            positive_mask = torch.zeros(batch_size, 2*batch_size, device=device, dtype=torch.bool)
            for o_idx, other_idx in enumerate(other_idxs):
                positive_mask[indices, indices + o_idx * batch_size] = True

            # Compute loss
            num_positives = len(other_idxs)
            loss = - (log_probs * positive_mask.float()).sum(dim=1) / num_positives
            loss = loss.mean()
            total_loss += loss

        # Average over modalities
        final_loss = total_loss / 3.0

        return final_loss


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations

class TriModalSigLIPCentroidLoss(nn.Module):
    """
    Sigmoid/BCE tri-modal objective with:
      - (A2J) Anchor-to-Joint (multi-positive) across the other two modalities
      - (CENT) Centroid contrast: each modality vs per-sample centroid
      - (VAR) Variance hinge to keep modalities near centroid
      - (UNI) Uniformity on centroids to avoid collapse

    Returns: total_loss, pairwise_losses  (so you can call: loss, pairwise = criterion(embeddings))

    Notes:
      * pairwise_losses uses SigLIP/BCE on each unordered pair, averaged over both directions (m->n and n->m).
      * By default, pairwise losses are for monitoring only; set lam_a2a>0 to include them in the total loss.
    """
    def __init__(
        self,
        temperature: float = 0.07,
        constant_temperature: bool = True,
        pos_weight: float = 2.0,
        delta: float = 0.2,
        alpha: float = 2.0,
        lam_a2j: float = 1.0,
        lam_cent: float = 1.0,
        lam_var: float = 0.05,
        lam_uni: float = 0.05,
        lam_a2a: float = 0.1,   # set >0 if you want pairwise BCE to contribute to total loss
        margin: float = 0.0
    ):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([np.log(1/temperature)], dtype=torch.float32))
        if constant_temperature:
            self.log_temp.requires_grad = False

        self.register_buffer("pos_weight_buffer", torch.tensor(pos_weight, dtype=torch.float32))
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.lam_a2j = float(lam_a2j)
        self.lam_cent = float(lam_cent)
        self.lam_var = float(lam_var)
        self.lam_uni = float(lam_uni)
        self.lam_a2a = float(lam_a2a)
        self.margin = float(margin)

    @property
    def temperature(self):
        # Convert back to temperature; clamp just in case.
        temp = 1.0 / self.log_temp.exp()
        return torch.clamp(temp, min=1e-3, max=1e3)

    def _bce(self, logits, targets, pos_weight):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")

    def forward(self, emb_dict: dict):
        """
        emb_dict: dict with exactly 3 modalities, each tensor (B, d). Unnormalized is OK; we L2-normalize inside.
        Returns:
            total_loss: scalar tensor
            pairwise_losses: dict {('modA','modB'): scalar_tensor} for monitoring (SigLIP/BCE, averaged m->n and n->m)
        """
        mods = list(emb_dict.keys())
        assert len(mods) == 3, "Exactly three modalities are required."

        # L2-normalize embeddings
        Z = {m: F.normalize(emb_dict[m], p=2, dim=-1) for m in mods}
        B = Z[mods[0]].shape[0]
        device = Z[mods[0]].device
        temp = self.temperature
        pos_w = self.pos_weight_buffer.to(device)
        eye = torch.eye(B, device=device, dtype=Z[mods[0]].dtype)

        # ---------- Centroids ----------
        C = torch.stack([Z[m] for m in mods], dim=0).mean(dim=0)  # (B,d)
        C = F.normalize(C, dim=-1)

        # ---------- (A2J) Anchor-to-Joint (multi-positive BCE) ----------
        a2j_terms = []
        for m in mods:
            blocks, labels = [], []
            for n in mods:
                if n == m:
                    continue
                S = (Z[m] @ Z[n].T) / temp  # (B,B)
                if self.margin != 0.0:
                    S = S.clone()
                    S.diagonal().add_(self.margin)
                blocks.append(S)
                labels.append(eye)  # positives on diagonal
            S_cat = torch.cat(blocks, dim=1)  # (B, 2B)
            Y_cat = torch.cat(labels, dim=1)  # (B, 2B)
            L = self._bce(S_cat, Y_cat, pos_weight=pos_w).mean()
            a2j_terms.append(L)
        L_a2j = torch.stack(a2j_terms).mean()

        # ---------- (CENT) Modality -> Centroid BCE ----------
        cent_terms = []
        for m in mods:
            Smc = (Z[m] @ C.T) / temp  # (B,B)
            if self.margin != 0.0:
                Smc = Smc.clone()
                Smc.diagonal().add_(self.margin)
            L = self._bce(Smc, eye, pos_weight=pos_w).mean()
            cent_terms.append(L)
        L_cent = torch.stack(cent_terms).mean()

        # ---------- (VAR) Variance hinge ----------
        var_terms = []
        for m in mods:
            dist = (Z[m] - C).norm(dim=1)  # (B,)
            var_terms.append(torch.clamp(dist - self.delta, min=0.0).pow(2).mean())
        L_var = torch.stack(var_terms).mean()

        # ---------- (UNI) Uniformity on centroids ----------
        D2 = torch.cdist(C, C, p=2).pow(2)                       # (B,B)
        mask = ~torch.eye(B, dtype=torch.bool, device=device)    # exclude diag
        if mask.any():
            L_uni = torch.log(torch.mean(torch.exp(self.alpha * D2[mask])))
        else:
            L_uni = torch.tensor(0.0, device=device, dtype=C.dtype)

        # ---------- Pairwise monitoring (and optional contribution) ----------
        # SigLIP/BCE per unordered pair, averaged over both directions.
        pairwise_losses = {}
        for a, b in combinations(mods, 2):
            Sab = (Z[a] @ Z[b].T) / temp
            Sba = (Z[b] @ Z[a].T) / temp
            if self.margin != 0.0:
                Sab = Sab.clone(); Sab.diagonal().add_(self.margin)
                Sba = Sba.clone(); Sba.diagonal().add_(self.margin)
            Lab = self._bce(Sab, eye, pos_weight=pos_w).mean()
            Lba = self._bce(Sba, eye, pos_weight=pos_w).mean()
            pair_loss = 0.5 * (Lab + Lba)
            pairwise_losses[(a, b)] = pair_loss

        # Optional: include average pairwise loss in total (default 0.0)
        if self.lam_a2a > 0.0:
            L_pairs_mean = torch.stack(list(pairwise_losses.values())).mean()
        else:
            L_pairs_mean = torch.tensor(0.0, device=device, dtype=C.dtype)

        # ---------- Total ----------
        total_loss = (
            self.lam_a2j * L_a2j +
            self.lam_cent * L_cent +
            self.lam_var * L_var +
            self.lam_uni * L_uni +
            self.lam_a2a * L_pairs_mean
        )

        return total_loss, pairwise_losses
