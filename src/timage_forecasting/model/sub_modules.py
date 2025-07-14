# Basic Modules
import torch
import torch.nn as nn
import timm
from typing import cast, Any

# Pytorch Forecasting Modules
from pytorch_forecasting.models.temporal_fusion_transformer.sub_modules import GatedResidualNetwork as _GRN
from pytorch_forecasting.models.tide.sub_modules import _ResidualBlock as _RB

import torch
import torch.nn as nn
from pytorch_forecasting.models.tide.sub_modules import _ResidualBlock as _RB

class _TimeSeriesEncoder(nn.Module):
    """
    MLP‐based encoder that turns L steps of (target + encoder cov + decoder cov)
    plus static covariates into an (L × E) embedding.

    Args:
        target_dim:           number of target channels D_y
        enc_cat_dim:          number of encoder categorical covariates
        enc_cont_dim:         number of encoder continuous covariates
        dec_cat_dim:          number of decoder categorical covariates (known a priori)
        dec_cont_dim:         number of decoder continuous covariates (known a priori)
        cat_static:           number of static categorical covariates
        cont_static:          number of static continuous covariates
        input_chunk_length:   history length L
        output_chunk_length:  prediction horizon H (for covariate concatenation)
        encoder_layers:       number of residual blocks
        hidden_size:          hidden size of each MLP block
        embed_dim:            final embedding dimension E per time step
        use_layer_norm:       whether to layer‐norm the final embeddings
        dropout:              dropout rate inside blocks
    """
    def __init__(
        self,
        target_dim: int,
        enc_cat_dim: int,
        enc_cont_dim: int,
        dec_cat_dim: int,
        dec_cont_dim: int,
        cat_static_dim: int,
        cont_static_dim: int,
        input_chunk_length: int,
        output_chunk_length: int,
        encoder_layers: int,
        hidden_size: int,
        embed_dim: int,
        use_layer_norm: bool,
        dropout: float,
    ):
        super().__init__()

        # keep shapes
        L, H = input_chunk_length, output_chunk_length
        D_y = target_dim
        D_enc = enc_cat_dim + enc_cont_dim
        D_dec = dec_cat_dim + dec_cont_dim
        D_static = cat_static_dim + cont_static_dim

        # total flattened MLP input size:
        mlp_in = L * D_y + (L + H) * (D_enc + D_dec) + D_static

        # build a stack of residual MLP blocks
        blocks = [
            _RB(
                input_dim=mlp_in if i == 0 else hidden_size,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            for i in range(encoder_layers)
        ]
        self.encoder_mlp = nn.Sequential(*blocks)

        # project into L × E
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_size, L * embed_dim),
            nn.Dropout(dropout),
        )

        # learned positional bias + optional layer norm
        self.pos_encoding = nn.Parameter(torch.randn(1, L, embed_dim) * 0.02)
        self.layer_norm = nn.LayerNorm(embed_dim) if use_layer_norm else None

        # stash for forward
        self.L = L
        self.embed_dim = embed_dim
        self.D_y = D_y
        self.D_enc = D_enc
        self.D_dec = D_dec
        self.D_static = D_static

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        x must contain:
          - "encoder_cat": (B, L, enc_cat_dim)
          - "encoder_cont": (B, L, enc_cont_dim)
          - "decoder_cat": (B, H, dec_cat_dim)
          - "decoder_cont": (B, H, dec_cont_dim)
          - "static_categorical_features": (B, cat_static_dim)  — optional
          - "static_continuous_features": (B, cont_static_dim)  — optional

        Returns:
          embeddings: (B, L, embed_dim)
        """
        B, L, E = x["encoder_cat"].size(0), self.L, self.embed_dim

        # 1) flatten past target if you want:
        #    here we assume the first D_y dims of encoder_cont are past targets;
        #    adapt if you feed y separately
        #    x_y = x["encoder_cont"][..., : self.D_y]  # if needed

        # 2) flatten all time‐dependent covariates
        enc_cat = x["encoder_cat"].reshape(B, -1)       # (B, L*enc_cat_dim)
        enc_cont = x["encoder_cont"].reshape(B, -1)     # (B, L*enc_cont_dim)
        dec_cat = x["decoder_cat"].reshape(B, -1)       # (B, H*dec_cat_dim)
        dec_cont = x["decoder_cont"].reshape(B, -1)     # (B, H*dec_cont_dim)
        time_flat = torch.cat([enc_cat, enc_cont, dec_cat, dec_cont], dim=1)

        # 3) static
        if "static_categorical_features" in x:
            st_cat = x["static_categorical_features"]
            st_cont = x["static_continuous_features"]
            static_flat = torch.cat([st_cat, st_cont], dim=1)
        else:
            static_flat = x.get("static", torch.zeros(B, self.D_static, device=enc_cat.device))

        # 4) build full MLP input: [Y_flat | time_flat | static_flat]
        #    if you have separate past Y, insert it here
        mlp_input = torch.cat([time_flat, static_flat], dim=1)

        # 5) MLP → bottleneck
        hidden = self.encoder_mlp(mlp_input)            # (B, hidden_size)

        # 6) project → L*E → (B, L, E)
        emb = self.embedding_head(hidden).view(B, L, E)

        # 7) add pos bias + optional LN
        emb = emb + self.pos_encoding
        if self.layer_norm is not None:
            emb = self.layer_norm(emb)
        return emb

# Temporal Image Encoder
class _TemporalImageEncoder(nn.Module):
    """
    Produces one embedding per frame over the history window, with static-conditioned FiLM gating.
    """
    def __init__(
        self,
        input_chunk_length: int,
        cat_static_dim: int,
        cont_static_dim: int,
        image_size: Any,
        embed_dim: int,
        backbone_name: str = "tf_efficientnetv2_b0.in1k",
        pretrained: bool = True,
        train_backbone: bool = True,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_chunk_length = input_chunk_length
        self.train_backbone = train_backbone
        self.cat_static_dim = cat_static_dim
        self.conf_static_dim = cont_static_dim
        self.image_size = image_size

        # 1) spatial backbone → global max pool → (B*L, feat_ch)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=self.image_size[0],
            num_classes=0,
            global_pool="max",
        )
        feat_ch = cast(int, self.backbone.num_features)

        self.frame_proj = nn.Sequential(
            nn.Linear(feat_ch, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        static_cov_dim = self.cat_static_dim + self.conf_static_dim

        self.static_gate = nn.Sequential(
            _GRN(
                input_size=static_cov_dim,
                hidden_size=static_cov_dim,
                output_size=static_cov_dim,
                dropout=dropout,
            ),
            nn.Linear(static_cov_dim, embed_dim * 2),
        )

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

        if not self.train_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False

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

        # enriched = embs * gate + shift               # (B, L, E)
        # out = embs + enriched                        # residual skip

        out = embs * gate + shift

        return out  # (B, L, E)

