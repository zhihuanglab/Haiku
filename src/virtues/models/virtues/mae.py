
from re import X
import torch
import torch.nn as nn
from einops import rearrange
from virtues.models.virtues.layers import LearnablePositionalEmbedding2D, PositionalEmbedding2D, cTransformerEncoderLayer
from virtues.models.virtues.helpers import build_selfattention_bias,  split_batch, merge_batch, build_selfattention_bias_channel_concat
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

class VirTuesEncoder(nn.Module):

    def __init__(self,
                protein_emb,
                patch_size=24,
                model_dim=512,
                feedforward_dim=1024,
                encoder_pattern="hwhw",
                num_encoder_heads=8,
                dropout=0.0,
                pos_emb="rope",
                ):
        super().__init__()

        #self.register_buffer("protein_emb", protein_emb, persistent=False)
        self.protein_emb = protein_emb
        self.patch_summary_token = nn.Parameter(torch.randn(model_dim)/model_dim**2)
        self.masked_token = nn.Parameter(torch.randn(model_dim)/model_dim**2)

        self.patch_encoder = nn.Linear(patch_size**2, model_dim)
        self.protein_encoder = nn.Linear(protein_emb.model_dim, model_dim)     
        
        if pos_emb == "learnable":
            self.positional_embedding = LearnablePositionalEmbedding2D(model_dim, max_pos=100)
        elif pos_emb == "absolute_beginning":
            self.positional_embedding = PositionalEmbedding2D(model_dim=model_dim, max_width_or_height=100)
        else:
            self.positional_embedding = None

        enc_layers = []
        for pattern in encoder_pattern:
            if pattern == "|" or pattern == "v":
                enc_layers.append(MarkerAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            elif pattern == "-" or pattern == "h":
                enc_layers.append(ChannelAttentionEncoderBlock(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            elif pattern == "f":
                enc_layers.append(FullAttentionEncoderLayer(model_dim, num_encoder_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            else:
                raise ValueError("encoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention)")
        
        self.encoder = nn.ModuleList(enc_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, channel_ids, mask=None):
        """
        x: B x C x H x W x D
        channel_ids: B x C
        Returns:
        x: B x C x H x W x D
        ps: B x H x W x D
        """
        B, C, H, W, _ = x.shape
        C = C+1
        pos = torch.stack(torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"), dim=-1) # H x W x 2
        pos = pos.expand(B, C, H, W, 2)
        pos = rearrange(pos, "b c h w d -> b c (h w) d")

        x = self.patch_encoder(x)

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.masked_token.expand(x.shape), x)
            mask = rearrange(mask, "b c h w -> b c (h w)")
        x = rearrange(x, "b c h w d -> b c (h w) d")

        proteins = self.protein_emb(channel_ids) # B x C x P
        proteins = torch.concat(proteins, dim=0)
        #proteins = self.protein_encoder(proteins) # B x C x D
        proteins = proteins.unsqueeze(2).expand(*x.shape)
        x = x + proteins

        x = torch.concat([self.patch_summary_token.expand(B, 1, H*W, x.shape[-1]), x], dim=1)
        if mask is not None:
            mask = torch.concat([torch.zeros(B, 1, H*W, dtype=torch.bool, device=mask.device), mask], dim=1)

        if self.positional_embedding is not None:
            x = self.positional_embedding(x, pos)

        for layer in self.encoder:
            if mask is None:
                x = layer.forward(x, pos)
            else:
                x = layer.forward_masked(x, pos, mask)

        x = rearrange(x, "b c (h w) d -> b c h w d", h=H, w=W)
        patch_summ = x[:, 0]
        x = x[:, 1:]
        return x, patch_summ

    def forward_list(self, x, channel_ids, mask=None):
        """
        x: list of tensors C_i x H x W x D
        channel_ids: list of tensors C_i
        mask: list of tensors C_i x H x W or None
        Returns:
        x: list of tensors C_i x H x W x D
        ps: list of tensors H x W x D
        """
        H, W, D = x[0].shape[1], x[0].shape[2], x[0].shape[3]
        channels_per_sample = [len(channels) for channels in channel_ids]
        #print(x.shape)

        x = torch.concat(x, dim=0) # sum(C_i) x H x W x D
        #channel_ids = torch.concat(channel_ids, dim=0) # sum(C_i)
        if mask is not None:
            mask = torch.concat(mask, dim=0) # sum(C_i) x H x W

        sum_C = x.shape[0]
        sum_C = sum_C + len(channels_per_sample)

        pos = torch.stack(torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"), dim=-1) # H x W x 2
        pos = pos.expand(sum_C, H, W, 2)
        pos = rearrange(pos, "c h w d -> c (h w) d")

        x = self.patch_encoder(x)

        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.masked_token.expand(x.shape), x)
            mask = rearrange(mask, "c h w -> c (h w)")

        x = rearrange(x, "c h w d -> c (h w) d")

        proteins = self.protein_emb(channel_ids) # sum_C x P
        proteins = torch.concat(proteins, dim=0)
        #proteins = self.protein_encoder(proteins) # sum_C x D
        proteins = proteins.unsqueeze(1).expand(*x.shape)
        x = x + proteins

        x = torch.split(x, channels_per_sample, dim=0)
        x = [torch.concat([self.patch_summary_token.expand(1, H*W, x_i.shape[-1]), x_i], dim=0) for x_i in x]
        x = torch.concat(x, dim=0)
        if mask is not None:
            mask = torch.split(mask, channels_per_sample, dim=0)
            mask = [torch.concat([torch.zeros(1, H*W, dtype=torch.bool, device=mask_i.device), mask_i], dim=0) for mask_i in mask]
            mask = torch.concat(mask, dim=0)

        channels_per_sample = [c + 1 for c in channels_per_sample]

        if self.positional_embedding is not None:
            x = self.positional_embedding(x, pos)

        for layer in self.encoder:
            if mask is None:
                x = layer.forward_cc(x, pos, channels_per_sample)
            else:
                x = layer.forward_cc_masked(x, pos, mask, channels_per_sample)
        
        x = rearrange(x, "c (h w) d -> c h w d", h=H, w=W)
        x = torch.split(x, channels_per_sample, dim=0) # list of tensors C_i x H x W x D
        
        ps = [x_i[0] for x_i in x]
        x = [x_i[1:] for x_i in x]

        return x, ps

class VirTuesDecoderMAE(nn.Module):

    def __init__(self,
                patch_size=24,
                model_dim=512,
                feedforward_dim=1024,
                pattern="hwhw",
                num_heads=8,
                num_hidden_layers_head=0,
                dropout=0.0,
                pos_emb="rope"
                ):
        super().__init__()
        
        decoder_layers = []
        if num_hidden_layers_head > 0:
            for _ in range(num_hidden_layers_head -1):
                decoder_layers.append(nn.Linear(model_dim, model_dim))
                decoder_layers.append(nn.GELU())
        decoder_layers.append(nn.Linear(model_dim, patch_size**2))
        self.decoder_mlp = nn.Sequential(*decoder_layers)
        
        dec_layers = []
        for pattern in pattern:
            if pattern == "|" or pattern == "v":
                dec_layers.append(MarkerAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            elif pattern == "-" or pattern == "h":
                dec_layers.append(ChannelAttentionEncoderBlock(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            elif pattern == "f":
                dec_layers.append(FullAttentionEncoderLayer(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=pos_emb))
            else:
                raise ValueError("decoder_pattern should contain either 'v' (for IntraCellAttention) or 'h' (IntraChannelAttention) or 'f' (FullAttention)")
        self.decoder = nn.ModuleList(dec_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, ps):
        """
        x: B x C x H x W x D
        ps: B x H x W x D
        Returns : B x C x H x W x D
        """
        B, C, H, W, D = x.shape
        ps = ps[:, None, None, :, :, :].expand(B, C, 1, H, W, D)
        x = x[:,:, None, :, :, :]
        x = torch.concat([ps, x], dim=2) # B x C x 2 x H x W x D
        x = rearrange(x, "b c e h w d -> (b c) e (h w) d")

        pos = torch.stack(torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"), dim=-1) # H x W x 2
        pos = pos.expand(B*C, 2, H, W, 2)
        pos = rearrange(pos, "b e h w d -> b e (h w) d")

        for layer in self.decoder:
            x = layer.forward(x, pos)

        x = x[:, 1] # (B C) S D
        x = self.decoder_mlp(x)
        x = rearrange(x, "(b c) (h w) d -> b c h w d", b=B, c=C, h=H, w=W)
        return x 

    def forward_list(self, x, ps):
        """
        x: list of tensors C_i x H x W x D
        ps: list of tensors H x W x D
        Returns: 
        x: sum(C_i) x H x W x D
        """
        H, W, D = x[0].shape[1], x[0].shape[2], x[0].shape[3]
        x = torch.concat([
            torch.concat([
                ps_i[None, None, :, :, :].expand(x_i.shape[0], 1, H, W, D),
                x_i[:, None , :, :, :],
            ], dim=1)            
            for x_i, ps_i in zip(x, ps)
        ], dim=0) # sum(C_i) x 2 x H x W x D
        x = rearrange(x, "c e h w d -> (c e) (h w) d")

        pos = torch.stack(torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij"), dim=-1) # H x W x 2
        pos = pos.expand(x.shape[0], H, W, 2)
        pos = rearrange(pos, "c h w d -> c (h w) d")

        channels = x.shape[0] // 2
        channels_per_sample = [2]*channels

        for layer in self.decoder:
            x = layer.forward_cc(x, pos, channels_per_sample) # x: sum(C_i)*2 x S x D

        x = x[1::2] # sum(C_i) x S x D
        x = self.decoder_mlp(x)
        x = rearrange(x, "c (h w) d -> c h w d", h=H, w=W)
        return x

class VirTuesMAE(nn.Module):

    def __init__(self,
                protein_emb,
                patch_size=24,
                model_dim=512,
                feedforward_dim=1024,
                encoder_pattern="hwhw",
                num_encoder_heads=8,
                mae_decoder_pattern="hwhw",
                mae_num_decoder_heads=8,
                mae_num_hidden_layers_head=0,
                dropout=0.0,
                pos_emb="rope",
                ):
        super().__init__()
        self.encoder = VirTuesEncoder(protein_emb, patch_size, model_dim, feedforward_dim, encoder_pattern, num_encoder_heads, dropout, pos_emb)
        self.mae_decoder = VirTuesDecoderMAE(patch_size, model_dim, feedforward_dim, mae_decoder_pattern, mae_num_decoder_heads, mae_num_hidden_layers_head, dropout, pos_emb)
        
    def embed(self, x, channel_ids, mask=None, return_dict=False, place_on_cpu=False):
        x, ps = self.encoder.forward(x, channel_ids, mask=mask) # B x C x H x W x D, B x H x W x D
        cls_token = ps.mean(dim=(1,2)) # B x D
        if return_dict:
            results = [{
                "cls_token": cls_token,
                "patch_summary_token": ps,
            }]
            if place_on_cpu:
                for res in results:
                    for key in res.keys():
                        res[key] = res[key].cpu()
            return results
        else:
            return cls_token, ps
       
    def forward(self, x, channel_ids, mask=None):
        """
        x: list of tensors B x C x H x W x D
        channel_ids: list of tensors B x C
        mask: list of tensors B x C x H x W or None
        Returns:
        cls_tokens: B x D
        x: B x C x H x W x D
        """
        x, ps = self.encoder.forward(x, channel_ids, mask=mask)
        recon = self.mae_decoder.forward(x, ps) # B x C x H x W x D
        return recon

    def forward_list(self, x, channel_ids, mask=None, return_only_dino=False):
        """
        x: list of tensors C_i x H x W x D
        channel_ids: list of tensors C_i
        mask: list of tensors C_i x H x W or None
        Returns:
        cls_tokens: B x D
        x: sum(C_i) x H x W x D        
        """
        x, ps = self.encoder.forward_list(x, channel_ids, mask=mask) # list C_i x H x W x D,  list of H x W x D
        recon = self.mae_decoder.forward_list(x, ps) # sum(C_i) x H x W x D
        return recon
    
    def reconstruct(self, x, channel_ids, mask=None, inject_mask_token_before_decoder=None):
        x, ps = self.encoder.forward(x, channel_ids, mask=mask)
        if inject_mask_token_before_decoder is not None:
            x = inject_mask_token_before_decoder.expand(x.shape)
        recon = self.mae_decoder.forward(x, ps) # B x C x H x W x D
        return recon


class ChannelAttentionEncoderBlock(nn.Module):

    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope"):
        super().__init__()
        self.encoder_layer = cTransformerEncoderLayer(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb)

    def forward(self, x, pos):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B C) S D")
        pos = rearrange(pos, "B C S D -> (B C) S D")
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "(B C) S D -> B C S D", B=B)
        return x
    
    def forward_masked(self, x, pos, mask):
        """
        x: BxCxSxD
        mask: BxCxS True indicating a masked token.
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B C) S D")
        pos = rearrange(pos, "B C S D -> (B C) S D")
        mask = rearrange(mask, "B C S -> (B C) S")

        x_true, x_false = split_batch(x, mask)
        pos_true, pos_false = split_batch(pos, mask)
        attn_bias = build_selfattention_bias(mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x = merge_batch(x_true, x_false, mask)

        x = rearrange(x, "(B C) S D -> B C S D", B=B)
        return x
    
    def forward_cc(self, x, pos, channels_per_sample):
        """
        x: CxSxD
        pos: CxSx2
        """
        x = self.encoder_layer(src=x, src_pos=pos)
        return x

    def forward_cc_masked(self, x, pos, mask, channels_per_sample):
        """
        x: CxSxD
        pos: CxSx2
        mask: CxS True indicating a masked token.
        """
        x_true, x_false = split_batch(x, mask)
        pos_true, pos_false = split_batch(pos, mask)
        attn_bias = build_selfattention_bias(mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x = merge_batch(x_true, x_false, mask)
        return x

class MarkerAttentionEncoderBlock(nn.Module):

    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope"):
        super().__init__()
        self.encoder_layer = cTransformerEncoderLayer(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb)

    def forward(self, x, pos):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B S) C D")
        pos = rearrange(pos, "B C S D -> (B S) C D")
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "(B S) C D -> B C S D", B=B)
        return x

    def forward_masked(self, x, pos, mask):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> (B S) C D")
        pos = rearrange(pos, "B C S D -> (B S) C D")
        mask = rearrange(mask, "B C S -> (B S) C")

        x_true, x_false = split_batch(x, mask)
        pos_true, pos_false = split_batch(pos, mask)
        attn_bias = build_selfattention_bias(mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x = merge_batch(x_true, x_false, mask)

        x = rearrange(x, "(B S) C D -> B C S D", B=B)
        return x
    
    def forward_cc(self, x, pos, channels_per_sample):
        """
        x: CxSxD
        pos: CxSx2
        """
        S = x.shape[1]
        q_seq_lens = channels_per_sample * S
        x = rearrange(x, "C S D -> (S C) D").unsqueeze(0)
        pos = rearrange(pos, "C S D -> (S C) D").unsqueeze(0)
        attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=q_seq_lens)
        x = self.encoder_layer(src=x, src_pos=pos, mask=attn_bias)
        x = x.squeeze(0)
        x = rearrange(x, "(S C) D -> C S D", S=S)
        return x
    
    def forward_cc_masked(self, x, pos, mask, channels_per_sample):
        """
        x: CxSxD
        pos: CxSx2
        """
        x = rearrange(x, "C S D -> S C D")
        pos = rearrange(pos, "C S D -> S C D")
        mask = rearrange(mask, "C S -> S C")

        x_true, x_false = split_batch(x, mask)
        pos_true, pos_false = split_batch(pos, mask)
        attn_bias = build_selfattention_bias_channel_concat(mask, channels_per_sample, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x = merge_batch(x_true, x_false, mask)

        x = rearrange(x, "S C D -> C S D")
        return x

class FullAttentionEncoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads, feedforward_dim, dropout, inbuilt_pos_emb="rope"):
        super().__init__()
        self.encoder_layer = cTransformerEncoderLayer(model_dim, num_heads, feedforward_dim, dropout=dropout, inbuilt_pos_emb=inbuilt_pos_emb)

    def forward(self, x, pos):
        """
        x: BxCxSxD
        pos: BxCxSx2
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> B (C S) D")
        pos = rearrange(pos, "B C S D -> B (C S) D")
        x = self.encoder_layer(src=x, src_pos=pos)
        x = rearrange(x, "B (C S) D -> B C S D", C=C)
        return x
    
    def forward_masked(self, x, pos, mask):
        """
        x: BxCxSxD
        pos: BxCxSx2
        mask: BxCxS
        """
        B, C, S, D = x.shape
        x = rearrange(x, "B C S D -> B (C S) D")
        pos = rearrange(pos, "B C S D -> B (C S) D")
        mask = rearrange(mask, "B C S -> B (C S)")

        x_true, x_false = split_batch(x, mask)
        pos_true, pos_false = split_batch(pos, mask)
        attn_bias = build_selfattention_bias(mask, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x = merge_batch(x_true, x_false, mask)

        x = rearrange(x, "B (C S) D -> B C S D", C=C)
        return x

    def forward_cc(self, x, pos, channels_per_sample):
        """
        x: CxSxD
        pos: CxSx2
        """
        S = x.shape[1]
        q_seq_lens = [c * S for c in channels_per_sample]
        x = rearrange(x, "C S D -> (C S) D").unsqueeze(0)
        pos = rearrange(pos, "C S D -> (C S) D").unsqueeze(0)
        attn_bias = BlockDiagonalMask.from_seqlens(q_seqlen=q_seq_lens)
        x = self.encoder_layer(src=x, src_pos=pos, mask=attn_bias)
        x = x.squeeze(0)
        x = rearrange(x, "(C S) D -> C S D", S=S)

        return x
    
    def forward_cc_masked(self, x, pos, mask, channels_per_sample):
        """
        x: CxSxD
        pos: CxSx2
        mask: CxS
        """
        S = x.shape[1]    
        x = rearrange(x, "C S D -> (C S) D")
        pos = rearrange(pos, "C S D -> (C S) D")
        mask = rearrange(mask, "C S -> (C S)")

        tokens_per_sample = [c * S for c in channels_per_sample]
        
        x_true, x_false = split_batch(x, mask)
        pos_true, pos_false = split_batch(pos, mask)
        attn_bias = build_selfattention_bias_channel_concat(mask, tokens_per_sample, use_true_as_query=False)

        x_false = self.encoder_layer(src=x_false, src_pos=pos_false, mask=attn_bias)
        x = merge_batch(x_true, x_false, mask)

        x = rearrange(x, "(C S) D -> C S D", S=S)
        return x
