import math
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_forecasting
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import GatedResidualNetwork as _GRN
from pytorch_forecasting.models.timexer.sub_modules import AttentionLayer, FullAttention, TriangularCausalMask
from pytorch_forecasting.models.tide.sub_modules import _ResidualBlock as _RB

# Time series Encoder

class _TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        future_cov_dim: int,
        temporal_hidden_size_future: int,
        temporal_width_future: int,
        static_cov_dim: int,
        output_chunk_length: int,
        input_chunk_length: int,
        num_encoder_layers: int,
        hidden_size: int,
        embed_dim: int,
        use_layer_norm: bool,
        dropout: float,
    ):
        super().__init__()
        # save args
        self.output_dim = output_dim
        self.future_cov_dim = future_cov_dim
        self.temporal_width_future = temporal_width_future
        self.static_cov_dim = static_cov_dim
        self.output_chunk_length = output_chunk_length  # H
        self.input_chunk_length = input_chunk_length    # L
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        # future cov projection
        if future_cov_dim > 0 and temporal_width_future > 0:
            self.future_cov_projection = _RB(
                input_dim=future_cov_dim,
                output_dim=temporal_width_future,
                hidden_size=temporal_hidden_size_future,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            cov_dim = temporal_width_future
        else:
            self.future_cov_projection = None
            cov_dim = future_cov_dim
        # encoder MLP stack
        encoder_input_dim = (
            input_chunk_length * output_dim +
            (input_chunk_length + output_chunk_length) * cov_dim +
            static_cov_dim
        )
        layers = [
            _RB(
                input_dim=encoder_input_dim,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
        ]
        for _ in range(num_encoder_layers - 1):
            layers.append(
                _RB(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
            )
        self.encoder_mlp = nn.Sequential(*layers)
        # embedding projection into per-step embeddings
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_size, output_chunk_length * embed_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(embed_dim) if use_layer_norm else None
        # positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, output_chunk_length, embed_dim))

    def forward(self, x_in):
        x_past, x_future_cov, x_static = x_in
        B = x_past.size(0)
        # slice past targets
        x_hist = x_past[:, :, : self.output_dim]  # (B,L,output_dim)
        # dynamic future covariates
        if self.future_cov_dim > 0 and x_future_cov is not None:
            hist_cov = x_past[:, :, -self.future_cov_dim:]
            dyn = torch.cat([hist_cov, x_future_cov], dim=1)  # (B, L+H, D_f)
            if self.future_cov_projection is not None:
                dyn = self.future_cov_projection(dyn)         # (B,L+H,cov_dim)
        else:
            dyn = None
        # flatten inputs
        parts = [x_hist, dyn, x_static]
        flat = [t.flatten(start_dim=1) for t in parts if t is not None]
        enc_in = torch.cat(flat, dim=1)                    # (B, encoder_input_dim)
        # encode
        hidden = self.encoder_mlp(enc_in)                  # (B, hidden_size)
        # project to sequence of embeddings
        emb_flat = self.embedding_head(hidden)             # (B, H*embed_dim)
        embeddings = emb_flat.view(B, self.output_chunk_length, self.embed_dim)
        # add positional encoding
        embeddings = embeddings + self.pos_encoding
        if self.layer_norm is not None:
            embeddings = self.layer_norm(embeddings)
        return embeddings  # (B, H, embed_dim)


# Temporal Image Encoder

import math
from math import sqrt
import torch
import torch.nn as nn

from pytorch_forecasting.models.timexer.sub_modules import AttentionLayer, FullAttention


def img_to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Split an image tensor into flattened patches.
    Args:
        x: Tensor of shape (B, C, H, W)
        patch_size: Size of each square patch
    Returns:
        Tensor of shape (B, num_patches, C * patch_size * patch_size)
    """
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        "Image height and width must be divisible by patch_size"
    h_patches = H // patch_size
    w_patches = W // patch_size
    # Unfold into patches: (B, C, H', W', p, p)
    x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Move patch dims ahead of channels, then flatten
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
    # Reshape to (B, H'*W', C * p * p)
    x = x.view(B, h_patches * w_patches, C * patch_size * patch_size)
    return x


class TransformerBlock(nn.Module):
    """
    Single Transformer block using custom full attention and a feed-forward network.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        factor: float = None,
    ):
        super().__init__()
        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(embed_dim)
        # FullAttention backend, no causal mask for encoder
        scale = factor or (1.0 / math.sqrt(embed_dim))
        self.attn = AttentionLayer(
            FullAttention(
                mask_flag=False,
                factor=scale,
                attention_dropout=dropout,
                output_attention=False,
            ),
            d_model=embed_dim,
            n_heads=num_heads,
        )
        # Pre-norm & feed-forward network
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, E)
        # Multi-head attention
        res1 = x
        x_attn, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=None)
        x = x_attn + res1
        # Feed-forward
        res2 = x
        x_ff = self.ff(self.norm2(x))
        return x_ff + res2


class TemporalViT(nn.Module):
    """
    Vision Transformer for single-image encoding.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_channels: int,
        num_heads: int,
        num_layers: int,
        patch_size: int,
        num_patches: int,
        dropout: float = 0.0,
        factor: float = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # Linear project flattened patches to embed_dim
        self.input_proj = nn.Linear(num_channels * patch_size * patch_size, embed_dim)
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Positional and temporal embeddings for cls + patches
        total_tokens = 1 + num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, embed_dim))
        self.temporal_embedding = nn.Parameter(torch.randn(1, total_tokens, embed_dim))
        # Transformer encoder
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                factor=factor,
            )
            for _ in range(num_layers)
        ])
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, embed_dim) representing the CLS embedding
        """
        # 1) Patchify
        patches = img_to_patches(x, self.patch_size)  # (B, P, C*p*p)
        B, P, _ = patches.shape
        # 2) Input projection
        tokens = self.input_proj(patches)             # (B, P, E)
        # 3) Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, E)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1+P, E)
        # 4) Add positional + temporal embeddings
        tokens = tokens + self.pos_embedding[:, : P+1, :] + self.temporal_embedding[:, : P+1, :]
        # 5) Transformer
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)
        # Return only the CLS embedding
        return tokens[:, 0, :]  # (B, E)
