
import torch
import torch.nn as nn
from torch.nn import functional as F

from xformers.ops import memory_efficient_attention

from torch import LongTensor

def build_activation(act_fcn):
    return {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "selu": nn.SELU(),
        "silu": nn.SiLU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid()
    }.get(act_fcn)


def build_feedforward(d_model, dim_feedforward, activation="gelu", dropout=0.1):
    return nn.Sequential(
        nn.Linear(d_model, dim_feedforward),
        build_activation(activation),
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, d_model),
    )

def linear_block(in_dim, out_dim, dropout_rate=0.05):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout_rate)
    )

class PositionalEmbedding2D(nn.Module):

    def __init__(self, model_dim : int, max_width_or_height : int = 1200, temperature : float = 10000.):
        super(PositionalEmbedding2D, self).__init__()

        assert model_dim % 4 == 0, 'Embedding dimension must be multiple of 4 for 2D positional embedding'

        dim_pe = model_dim // 2

        possible_positions = torch.arange(max_width_or_height, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_pe , 2, dtype=torch.float32) * - (torch.log(torch.tensor(temperature)) / dim_pe))
        pos = possible_positions * div_term
        sin = torch.sin(pos)
        cos = torch.cos(pos)

        self.register_buffer('positional_embeddings', torch.zeros(max_width_or_height, dim_pe))

        self.positional_embeddings[:, 0::2] = sin
        self.positional_embeddings[:, 1::2] = cos

    def forward(self, x, positions : LongTensor):
        """
        Computes positional embeddings corresponding to 2D input positions
        Args:
            x: (..., model_dim) tensor
            positions: (..., 2) tensor tensor of 2D positions
        Returns:
            (..., model_dim) tensor of positional embeddings
        """
        rows = positions.select(dim=-1, index=0)
        cols = positions.select(dim=-1, index=1)

        row_pos_emb = self.positional_embeddings[rows]
        col_pos_emb = self.positional_embeddings[cols]

        pos_emb = torch.cat([row_pos_emb, col_pos_emb], dim=-1)
        return x + pos_emb    


class RotaryPositionalEmbedding1D(nn.Module):

    def __init__(self, model_dim : int, max_seq_length : int = 1200, temperature : float = 10000.):
        super(RotaryPositionalEmbedding1D, self).__init__()

        assert model_dim % 2 == 0, 'Embedding dimension must be multiple of 2 for 1D positional embedding'
        self.model_dim = model_dim

        possible_positions = torch.arange(max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim , 2, dtype=torch.float32) * - (torch.log(torch.tensor(temperature)) / model_dim))
        pos = possible_positions * div_term
        sin = torch.sin(pos)
        sin = torch.concat([sin, sin], dim=-1)
        self.register_buffer('sin', sin)
        cos = torch.cos(pos)
        cos = torch.concat([cos, cos], dim=-1)
        self.register_buffer('cos', cos)

    def invert_negate(self, x):
        return torch.cat([-x[...,self.model_dim // 2:], x[...,:self.model_dim // 2]], dim=-1)

    def forward(self, x, pos):
        """
        Applies rotary positional encoding to input tensor
        Args:
            x: (..., model_dim) tensor
            pos: (..., ) tensor of positions
        """
        x = x * self.cos[pos] +  self.invert_negate(x) * self.sin[pos]
        return x

class RotaryPositionalEmbedding2D(nn.Module):

    def __init__(self, model_dim : int, max_pos : int = 1200, temperature : float = 10000.):
        super(RotaryPositionalEmbedding2D, self).__init__()

        assert model_dim % 4 == 0, 'Embedding dimension must be multiple of 4 for 2D positional embedding'
        self.model_dim = model_dim
        self.rope1d = RotaryPositionalEmbedding1D(model_dim // 2, max_pos, temperature)

    def forward(self, x, pos):
        """
        Applies 2D rotary positional encoding to input tensor
        Args:
            x: (..., model_dim) tensor
            pos: (..., 2) tensor of 2D positions
        """
        d = self.model_dim // 2

        x1 = x[..., :d]
        x2 = x[..., d:]

        x1 = self.rope1d(x1, pos.select(dim=-1, index=0))
        x2 = self.rope1d(x2, pos.select(dim=-1, index=1))

        return torch.cat([x1, x2], dim=-1)
    
class LearnablePositionalEmbedding2D(nn.Module):

    def __init__(self, model_dim, max_pos=100):
        super(LearnablePositionalEmbedding2D, self).__init__()
        self.pos_embeddings = nn.Parameter(torch.randn(max_pos, max_pos, model_dim) / model_dim**2)

    def forward(self, x, pos):
        """
        Applies learnable positional embedding to input tensor
        Args:
            x: (..., model_dim) tensor
            pos: (..., 2) tensor of 2D positions
        """
        to_add = self.pos_embeddings[pos[...,0], pos[...,1]]
        return x + to_add

class cMHAwithPosEmb(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, inbuilt_pos_emb="absolute"):
        
        super().__init__()

        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.dropout = dropout

        if inbuilt_pos_emb == "absolute":
            self.pos_emb = PositionalEmbedding2D(model_dim=self.embed_dim)
            self.pos_after_linear = False
            self.pos_before_linear = True
        elif inbuilt_pos_emb == "rope":
            self.pos_emb = RotaryPositionalEmbedding2D(model_dim=self.head_dim)
            self.pos_after_linear = True
            self.pos_before_linear = False
        elif inbuilt_pos_emb == "learnable":
            self.pos_after_linear = False
            self.pos_before_linear = False
        elif inbuilt_pos_emb == "absolute_beginning":
            self.pos_after_linear = False
            self.pos_before_linear = False
        else:
            raise ValueError("pos_embedding must be 'absolute' or 'rope' or 'learnable' or 'absolute_beginning'")
    
    def forward(self, query, key, value, query_pos=None, key_pos=None, mask=None, key_padding_mask=None):
        """
        Args:
            q: (B, L, embed_dim)
            k: (B, S, embed_dim)
            v: (B, S, embed_dim)
            q_pos: (B, L, 2) 2D positions of query
            k_pos: (B, S, 2) 2D positions of key
        """
        assert mask is None or key_padding_mask is None, "mask and key_padding_mask cannot be provided at the same time"

        bs = query.size(0)
        src_length = key.size(1)
        target_length = query.size(1)

        if self.pos_before_linear:
            if query_pos is not None and key_pos is not None:
                query = self.pos_emb(query, query_pos)
                key = self.pos_emb(key, key_pos)

        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bs, src_length), "key_padding_mask shape must be (B, S)"
            key_padding_mask = key_padding_mask.view(bs, 1, src_length)# (B, 1, S)
            if key_padding_mask.dtype == torch.bool:
                key_padding_mask = torch.zeros_like(key_padding_mask, dtype=torch.float32).masked_fill(key_padding_mask, float("-inf"))
                attn_mask = key_padding_mask #(B, 1, S)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, target_length, -1) # (B, num_heads, L, S)

            query = query.reshape(bs, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, L, head_dim)
            key = key.reshape(bs, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, S, head_dim)
            value = value.reshape(bs, -1, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, S, head_dim)

            if self.pos_after_linear:
                if query_pos is not None and key_pos is not None:
                    query_pos = query_pos.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # (B, num_heads, L, 2)
                    key_pos = key_pos.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # (B, num_heads, S, 2)

                    query = self.pos_emb(query, query_pos)
                    key = self.pos_emb(key, key_pos)

            attn_output = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=self.dropout) # (B, num_heads, L, head_dim)
            attn_output = attn_output.transpose(1, 2).reshape(bs, -1, self.num_heads * self.head_dim) # (B, L, embed_dim)
            return self.W_o(attn_output) # (B, L, embed_dim)

        else:
            query = query.reshape(bs, -1, self.num_heads, self.head_dim) # (B, L, num_heads,head_dim)
            key = key.reshape(bs, -1, self.num_heads, self.head_dim) # (B, S, num_heads, head_dim)
            value = value.reshape(bs, -1, self.num_heads, self.head_dim) # (B, S, num_heads, head_dim)

            if self.pos_after_linear:
                if query_pos is not None and key_pos is not None:
                    query_pos = query_pos.unsqueeze(2).expand(-1, -1, self.num_heads, -1) # (B, L, num_heads, 2)
                    key_pos = key_pos.unsqueeze(2).expand(-1, -1, self.num_heads, -1) # (B, S, num_heads, 2)
                    query = self.pos_emb(query, query_pos)
                    key = self.pos_emb(key, key_pos)

            if torch.is_autocast_enabled():
                query, key, value = query.half(), key.half(), value.half()
            attn_output = memory_efficient_attention(query, key, value, attn_bias=mask, p=self.dropout) # (B, L, num_heads, 2)
            attn_output = attn_output.reshape(bs, -1, self.num_heads * self.head_dim) # (B, L, embed_dim)
            return self.W_o(attn_output) # (B, L, embed_dim)


class cTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward, dropout, activation="gelu", bias=True, inbuilt_pos_emb="absolute"):
        super(cTransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([cTransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout=dropout, activation=activation, bias=bias, inbuilt_pos_emb=inbuilt_pos_emb) for _ in range(num_layers)])

    def forward(self, src, src_pos=None, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (B, S, d_model) source sequence
            attn_mask: (B, S, S) boolean mask or float mask
            key_padding_mask: (B, S) boolean mask or float mask
        """   
        for layer in self.layers:
            src = layer(src, src_pos=src_pos, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src

class cTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation = "gelu", bias=True, inbuilt_pos_emb="absolute"):
        super(cTransformerEncoderLayer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.multi_head_attention = cMHAwithPosEmb(embed_dim=d_model, num_heads=nhead, dropout=dropout, bias=bias, inbuilt_pos_emb=inbuilt_pos_emb)
        self.feedforward = build_feedforward(d_model, dim_feedforward, activation, dropout)

        self.layernorm1 = nn.LayerNorm(d_model, bias=bias)
        self.layernorm2 = nn.LayerNorm(d_model, bias=bias)
        
    def forward(self, src, src_pos=None, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: (B, S, d_model) source sequence
            attn_mask: (B, S, S) boolean mask or float mask
            key_padding_mask: (B, S) boolean mask or float mask
        """
        
        """ Post LN Attention 
        src = self.layernorm1(src + self.multi_head_attention(query=src, key=src, value=src, query_pos=src_pos, key_pos=src_pos, mask=mask, key_padding_mask=src_key_padding_mask))
        src = self.layernorm2(src + self.feedforward(src))
        """
        # Pre-LN MHA
        lsrc = self.layernorm1(src)
        src = src + self.multi_head_attention(query=lsrc, key=lsrc, value=lsrc, query_pos=src_pos, key_pos=src_pos, mask=mask, key_padding_mask=src_key_padding_mask)
        lsrc = self.layernorm2(src)
        src = src + self.feedforward(lsrc)

        return src
 