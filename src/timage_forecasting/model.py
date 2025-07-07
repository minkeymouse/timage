"""
Timage for image integrated time series forecasting.
"""
from copy import copy
from typing import Optional, Union, Any

import torch
from torch import nn

from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE
from pytorch_forecasting.models.base import BaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.models.tide.sub_modules import _ResidualBlock
from pytorch_forecasting.models.timexer.sub_modules import AttentionLayer, FullAttention

from mamba_ssm import Mamba2

from timage_forecasting.sub_modules import _TimeSeriesEncoder, _TemporalImageEncoder

class Timage(BaseModelWithCovariates):
    """Image integrated time series model for long-term forecasting."""
    def __init__(
        self,
        output_chunk_length: int,
        input_chunk_length: int,
        num_encoder_layers_ts: int = 2,
        image_shape: Optional[tuple[int, int, int]] = None,
        num_decoder_layers: int = 2,
        decoder_output_dim: int = 16,
        hidden_size_ts: int = 128,
        hidden_size_img: int = 128,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        static_categoricals: Optional[list[str]] = None,
        static_reals: Optional[list[str]] = None,
        time_varying_categoricals_encoder: Optional[list[str]] = None,
        time_varying_categoricals_decoder: Optional[list[str]] = None,
        categorical_groups: Optional[dict[str, list[str]]] = None,
        time_varying_reals_encoder: Optional[list[str]] = None,
        time_varying_reals_decoder: Optional[list[str]] = None,
        embedding_sizes: Optional[dict[str, tuple[int, int]]] = None,
        embedding_paddings: Optional[list[str]] = None,
        embedding_labels: Optional[list[str]] = None,
        x_reals: Optional[list[str]] = None,
        x_categoricals: Optional[list[str]] = None,
        x_image: Optional[list[str]] = None,
        temporal_width_future: int = 4,
        temporal_hidden_size_future: int = 32,
        temporal_decoder_hidden: int = 32,
        logging_metrics: nn.ModuleList = None, # type: ignore[assignment]
        image_backbone: str = "tf_efficientnetv2_b0.in1k",
        image_pretrained: bool = True,
        **kwargs,
    ):
        if static_categoricals is None:
            static_categoricals = []
        if static_reals is None:
            static_reals = []
        if time_varying_categoricals_encoder is None:
            time_varying_categoricals_encoder = []
        if time_varying_categoricals_decoder is None:
            time_varying_categoricals_decoder = []
        if categorical_groups is None:
            categorical_groups = {}
        if time_varying_reals_encoder is None:
            time_varying_reals_encoder = []
        if time_varying_reals_decoder is None:
            time_varying_reals_decoder = []
        if embedding_sizes is None:
            embedding_sizes = {}
        if embedding_paddings is None:
            embedding_paddings = []
        if embedding_labels is None:
            embedding_labels = []
        if x_reals is None:
            x_reals = []
        if x_categoricals is None:
            x_categoricals = []
        if x_image is None:
            x_image = []
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        if self.loss is None:
            self.loss = RMSE()

        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        super().__init__(logging_metrics=logging_metrics, **kwargs)
        self.output_dim = len(self.target_names)
        self.image_backbone = image_backbone
        self.image_pretrained = image_pretrained

        self.embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            categorical_groups=categorical_groups,
            embedding_paddings=embedding_paddings,
            x_categoricals=x_categoricals,
        )

        static_cov_dim = len(static_reals) + sum(
            embedding_sizes[cat][1] for cat in static_categoricals if cat in embedding_sizes
        )
        future_cov_dim = len(set(time_varying_reals_decoder) - set(self.target_names)) + sum(
            embedding_sizes[cat][1] for cat in time_varying_categoricals_decoder if cat in embedding_sizes
        )

        self.encoder_ts = _TimeSeriesEncoder(
            output_dim=self.output_dim,
            future_cov_dim=future_cov_dim,
            temporal_hidden_size_future=temporal_hidden_size_future,
            temporal_width_future=temporal_width_future,
            static_cov_dim=static_cov_dim,
            input_chunk_length=input_chunk_length,
            output_chunk_length=output_chunk_length,
            num_encoder_layers=num_encoder_layers_ts,
            hidden_size=hidden_size_ts,
            embed_dim=hidden_size_ts,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        in_channels = image_shape[0] if (image_shape is not None and len(image_shape) == 3) else 1
        self.encoder_img = _TemporalImageEncoder(
            embed_dim=hidden_size_img,
            static_cov_dim=static_cov_dim,
            input_chunk_length=input_chunk_length,
            backbone_name = self.image_backbone,
            pretrained = self.image_pretrained,
            in_chans=in_channels,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        d_model = hidden_size_ts  # set attention model dim equal to time-series embedding dim
        n_heads = 8  # number of attention heads (adjustable if needed)
        self.cross_attn = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=False),
            d_model=d_model,
            n_heads=n_heads,
        )

        self.mamba_ssm = Mamba2(
            d_model=hidden_size_ts,
            headdim=4,   # number of heads (for internal multi-head processing)
            d_state=64,  # state dimension
            d_conv=4,    # convolutional expansion dimension
            expand=2,    # expansion factor in transition
        )

        # Linear layer for lookback skip connection (maps past target sequence to forecast)
        self.lookback_skip = nn.Linear(input_chunk_length * self.output_dim, output_chunk_length * self.output_dim)
        # Decoder MLP: residual blocks to map encoded features to output sequence

        self.L_in = input_chunk_length
        self.H_out = output_chunk_length
        self.decoder_input_dim = self.L_in * hidden_size_ts

        layers = []

        layers.append(_ResidualBlock(
            input_dim=self.decoder_input_dim,
            output_dim=temporal_decoder_hidden,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        ))

        for _ in range(num_decoder_layers - 1):
            layers.append(_ResidualBlock(
                input_dim=temporal_decoder_hidden,
                output_dim=temporal_decoder_hidden,
                hidden_size=temporal_decoder_hidden,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ))

        self.decoder_mlp = nn.Sequential(*layers)
        self.decoder_head = nn.Sequential(
            nn.Linear(temporal_decoder_hidden, output_chunk_length * decoder_output_dim),
            nn.Dropout(dropout),
        )
        # Final output layer: maps intermediate decoder output dim to actual target dim
        self.decoder_output_dim = decoder_output_dim

        self.output_layer = nn.Linear(decoder_output_dim, self.H_out)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the Timage model.
        Args:
            x (dict): Input batch dict, expected keys include:
                      - "encoder_cont": continuous values for encoder (including target and covariates)
                      - "encoder_cat": categorical values for encoder
                      - "decoder_cont": continuous values for decoder (future covariates)
                      - "decoder_cat": categorical values for decoder
                      - "x_image": past images tensor of shape (B, L, C, H, W) if image data is present
        Returns:
            Dict[str, torch.Tensor]: output of the model (after scaling), with key "prediction".
        """
        input_enc_y = x["encoder_cont"][..., self.target_positions]
        input_enc_x_t = 
        input_dec_x_t = 
        input_static = 
        input_img = x["x_image"]  # (B, L_img, C, H, W) or None
        img_emb = self.encoder_img(input_img, input_static)  # shape: (B, L, hidden_size_img)
        ts_emb = self.encoder_ts(input_enc_y, input_enc_x_t, input_static)  # shape: (B, L, hidden_size_ts)

        attn_out = self.cross_attn(ts_emb, img_emb, )
        # query as ts emb, key and value as img emb

        ssm_out = self.mamba_ssm(attn_out)  # shape: (B, L_ts, hidden_size_ts)

        combined_emb = torch.cat([ssm_out, ts_emb], dim=2)  # shape: (B, L_ts, hidden_size_ts + hidden_size_ts)

        B = combined_emb.size(0)
        L = combined_emb.size(1)  # should equal input_chunk_length

        combined_flat = combined_emb.reshape(B, -1)  # shape: (B, L * 2*hidden_size_ts)

        hidden_vec = self.decoder_mlp(combined_flat)  # shape: (B, temporal_decoder_hidden)
        out_flat = self.decoder_head(hidden_vec)      # shape: (B, output_chunk_length * decoder_output_dim)
        # Reshape to (B, output_chunk_length, decoder_output_dim)
        out_seq = out_flat.view(B, self.output_chunk_length, self.decoder_output_dim)
        out_seq = self.output_layer(out_seq)

        y_hist_flat = input_enc_y.reshape(B, -1)  # (B, L * output_dim)
        skip_flat = self.lookback_skip(y_hist_flat)   # (B, H * output_dim)
        skip_seq = skip_flat.view(B, self.output_chunk_length, self.output_dim)  # (B, H, output_dim)

        prediction = out_seq + skip_seq  # (B, H, output_dim)

        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        return self.to_network_output(prediction=prediction)
