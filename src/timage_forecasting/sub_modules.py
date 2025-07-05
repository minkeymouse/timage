import math
from math import sqrt

# Basic Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from typing import cast

# Pytorch Forecasting Modules
import pytorch_forecasting
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import GatedResidualNetwork as _GRN
from pytorch_forecasting.models.tide.sub_modules import _ResidualBlock as _RB

# Time series Encoder
class _TimeSeriesEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int,            # D_y
        future_cov_dim: int,        # D_f
        temporal_hidden_size_future: int,
        temporal_width_future: int, # projected cov_dim
        static_cov_dim: int,        # D_s
        input_chunk_length: int,    # L
        output_chunk_length: int,   # H, only for flattening
        num_encoder_layers: int,
        hidden_size: int,
        embed_dim: int,             # E
        use_layer_norm: bool,
        dropout: float,
    ):
        super().__init__()
        L, H = input_chunk_length, output_chunk_length
        self.output_dim = output_dim
        self.L = L
        self.H = H
        self.future_cov_dim = future_cov_dim

        # 1) optional projection of future covariates
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

        # 2) compute flattened input size = L·D_y + (L+H)·cov_dim + D_s
        encoder_input_dim = L * output_dim + (L + H) * cov_dim + static_cov_dim

        # 3) MLP bottleneck (like TiDE)
        layers = [ _RB(
            input_dim=encoder_input_dim,
            output_dim=hidden_size,
            hidden_size=hidden_size,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        ) ]
        for _ in range(num_encoder_layers - 1):
            layers.append(_RB(
                input_dim=hidden_size,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ))
        self.encoder_mlp = nn.Sequential(*layers)

        # 4) unpack: hidden_size -> L*E
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_size, L * embed_dim),
            nn.Dropout(dropout),
        )

        # 5) learned positional bias over the L history steps
        self.pos_encoding = nn.Parameter(torch.randn(1, L, embed_dim)*0.02)
        self.layer_norm  = nn.LayerNorm(embed_dim) if use_layer_norm else None

    def forward(self, x_in):
        """
        Args:
          x_past:        Tensor of shape (B, L, D_y + D_fpast)
                         — the first D_y channels are the past target values,
                           the last   D_fpast channels are any past covariates
          x_future_cov:  Tensor of shape (B, H, D_ffuture)
                         — the known covariates over the next H time-steps
          x_static:      Tensor of shape (B, D_s)
                         — the static (time-invariant) covariates

        Returns:
          embeddings:    Tensor of shape (B, L, E)
                         — one E-dimensional embedding per past time-step
        """
        x_past, x_future_cov, x_static = x_in
        B = x_past.size(0)
        L, H = self.L, self.H

        # a) last L of the target dims
        x_hist = x_past[:, :L, :self.output_dim]          # (B, L, D_y)

        # b) build dyn covs = last-L past covs  +  H future covs
        if self.future_cov_projection or self.future_cov_dim>0:
            hist_cov = x_past[:, :L, -self.future_cov_dim:]  # (B, L, D_f)
            dyn = torch.cat([hist_cov, x_future_cov], dim=1)  # (B, L+H, D_f)
            if self.future_cov_projection:
                dyn = self.future_cov_projection(dyn)         # (B, L+H, cov_dim)
        else:
            dyn = None

        # c) flatten everything
        parts = [x_hist, dyn, x_static]
        flat  = [t.flatten(1) for t in parts if t is not None]  # each -> (B, ·)
        enc_in = torch.cat(flat, dim=1)                         # (B, encoder_input_dim)

        # d) MLP → single vector
        hidden = self.encoder_mlp(enc_in)                       # (B, hidden_size)

        # e) unpack into L embeddings
        emb_flat   = self.embedding_head(hidden)                # (B, L*E)
        embeddings = emb_flat.view(B, L, -1)                    # (B, L, E)

        # f) add positional bias + optional LN
        embeddings = embeddings + self.pos_encoding              # (B, L, E)
        if self.layer_norm:
            embeddings = self.layer_norm(embeddings)

        return embeddings  # (B, L, E)

# Temporal Image Encoder
class _TemporalImageEncoder(nn.Module):
    """
    Produces one embedding per frame over the history window, with static-conditioned FiLM gating.
    """
    def __init__(
        self,
        embed_dim: int,
        static_cov_dim: int,
        input_chunk_length: int,
        in_chans: int = 1,
        backbone_name: str = "tf_efficientnetv2_b0.in1k",
        pretrained: bool = True,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_chunk_length = input_chunk_length

        # 1) spatial backbone → global max pool → (B*L, feat_ch)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="max",
        )
        feat_ch = cast(int, self.backbone.num_features)

        # 2) project each frame's features into embed_dim
        self.frame_proj = nn.Sequential(
            nn.Linear(feat_ch, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3) static-conditioning FiLM gate: GRN -> Linear to 2*embed_dim
        self.static_gate = nn.Sequential(
            _GRN(
                input_size=static_cov_dim,
                hidden_size=static_cov_dim,
                output_size=static_cov_dim,
                dropout=dropout,
            ),
            nn.Linear(static_cov_dim, embed_dim * 2),
        )

        # 4) learned positional bias over L frames
        self.pos_encoding = nn.Parameter(
            torch.randn(1, input_chunk_length, embed_dim) * 0.02
        )
        self.layer_norm = nn.LayerNorm(embed_dim) if use_layer_norm else None

    def forward(self, x_img: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x_img:    (B, L, C, H, W)
          x_static: (B, D_s)
        Returns:
          embeddings: (B, L, E)
        """
        B, L, C, H, W = x_img.shape
        assert L == self.input_chunk_length, (
            f"Expected {self.input_chunk_length} frames, got {L}"
        )

        # run all frames through the CNN backbone at once
        frames = x_img.view(B * L, C, H, W)   # (B*L, C, H, W)
        feats = self.backbone(frames)         # (B*L, feat_ch)
        embs = self.frame_proj(feats)         # (B*L, E)
        embs = embs.view(B, L, self.embed_dim)  # (B, L, E)

        # add positional encoding
        embs = embs + self.pos_encoding        # (B, L, E)
        if self.layer_norm is not None:
            embs = self.layer_norm(embs)

        # static FiLM gating
        gate_shift = self.static_gate(x_static)      # (B, 2*E)
        gate, shift = gate_shift.chunk(2, dim=1)     # each (B, E)
        gate = gate.sigmoid().unsqueeze(1)           # (B, 1, E)
        shift = shift.unsqueeze(1)                   # (B, 1, E)

        enriched = embs * gate + shift               # (B, L, E)
        out = embs + enriched                        # residual skip

        return out  # (B, L, E)

