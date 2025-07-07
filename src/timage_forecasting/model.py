"""
Timage for image integrated time series forecasting.
"""
from copy import copy
from typing import Optional, Union, Any, Tuple, Dict, List

import torch
from torch import nn
from pytorch_lightning import LightningModule

from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE
from pytorch_forecasting.models.base import BaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.models.tide.sub_modules import _ResidualBlock
from pytorch_forecasting.models.timexer.sub_modules import AttentionLayer, FullAttention
from pytorch_forecasting.metrics import MultiLoss

from mamba_ssm import Mamba2

from timage_forecasting.datamodule import TimeSeriesWithImageDataSet
from timage_forecasting.sub_modules import _TimeSeriesEncoder, _TemporalImageEncoder

class Timage(BaseModelWithCovariates):
    """Image integrated time series model for long-term forecasting."""
    def __init__(
        self,
        input_chunk_length: int,
        output_chunk_length: int,

        num_encoder_layers_ts: int,       # how many layers in your TS encoder
        hidden_size_ts:           int,       # embedding / hidden size of TS encoder
        num_decoder_layers:       int,       # number of residual blocks in decoder
        decoder_output_dim:       int,       # width of the decoder MLP’s output before final projection

        temporal_width_future:    int = 4,   # how many future steps your TS encoder “sees” internally
        temporal_hidden_size_future: int = 32, 
        temporal_decoder_hidden:  int = 32,  # hidden size inside each decoder block

        use_layer_norm:           bool = False,
        dropout:                  float = 0.1,

        image_backbone:           str = "tf_efficientnetv2_b0.in1k",
        image_pretrained:         bool = True,
        logging_metrics = None,

        # Optional parameters
        embedding_sizes:       Optional[dict[str, int]] = None,
        x_categoricals: Optional[List[str]] = None, 
        image_shape: Optional[Tuple[int, int, int]] = None,

        **kwargs
    ):
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])
        self.loss = RMSE()
        
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.embedding_sizes = embedding_sizes
        self.x_categoricals = x_categoricals
        self.image_shape = image_shape
        self.output_dim = len(self.target_names)
        self.image_backbone = image_backbone
        self.image_pretrained = image_pretrained

        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        super().__init__(logging_metrics=logging_metrics, **kwargs)

        if self.embedding_sizes and x_categoricals is not None:
            self.embeddings = MultiEmbedding(
                embedding_sizes=self.embedding_sizes,
                x_categoricals=x_categoricals,
            )

        static_cov_dim = len(self.static_variables)
        future_cov_dim = len(self.decoder_variables)

        self.encoder_ts = _TimeSeriesEncoder(
            output_dim=self.output_dim,
            future_cov_dim=future_cov_dim,
            temporal_hidden_size_future=temporal_hidden_size_future,
            temporal_width_future=temporal_width_future,
            static_cov_dim=static_cov_dim,
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            num_encoder_layers=num_encoder_layers_ts,
            hidden_size=hidden_size_ts,
            embed_dim=hidden_size_ts,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        in_channels = image_shape[0] if (image_shape is not None and len(image_shape) == 3) else 1
        self.encoder_img = _TemporalImageEncoder(
            embed_dim=hidden_size_ts,
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

        self.decoder_mlp = nn.Sequential(
            _ResidualBlock(input_dim=input_chunk_length * 2*hidden_size_ts,  # because we concat SSM + TS
                        output_dim=temporal_decoder_hidden,
                        hidden_size=temporal_decoder_hidden,
                        use_layer_norm=use_layer_norm,
                        dropout=dropout),
            *[
                _ResidualBlock(input_dim=temporal_decoder_hidden,
                            output_dim=temporal_decoder_hidden,
                            hidden_size=temporal_decoder_hidden,
                            use_layer_norm=use_layer_norm,
                            dropout=dropout)
                for _ in range(num_decoder_layers - 1)
            ]
        )
        # If Decoder MLP is too big, we can change to transformer.

        self.decoder_head = nn.Sequential(
            nn.Linear(temporal_decoder_hidden, output_chunk_length * decoder_output_dim),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(decoder_output_dim, self.output_dim)

    def training_step(self, batch: Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]], batch_idx: int):
        # 1) unpack
        x, (y, w) = batch
        #   x: dict with encoder_cat, encoder_cont, decoder_cont, decoder_cat, x_image, target_scale, etc.
        #   y: (B, H, output_dim) ground-truth
        #   w: (B, H) optional sample weights

        # 2) forward pass
        out = self(x)
        preds = out["prediction"]                    # (B, H, output_dim)

        # 3) compute loss (weighted)
        loss = self.loss(preds, y, w)
        # log the raw loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # 4) compute & log any additional metrics you configured in self.logging_metrics
        #    each metric is a torchforecasting Metric module expecting (preds, y)
        for metric in self.logging_metrics:
            name = metric.__class__.__name__.lower()
            value = metric(preds, y)
            # log under train_<metric>, e.g. train_smape, train_mae, etc.
            self.log(f"train_{name}", value, prog_bar=False, on_step=False, on_epoch=True)

        # 5) return loss so Lightning can backprop
        return loss

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # unpack
        enc_cat  = x["encoder_cat"]   # (B, L, n_cat)
        enc_cont = x["encoder_cont"]  # (B, L, n_real)
        dec_cont = x["decoder_cont"]  # (B, H, n_real)
        imgs     = x.get("x_image", None)    # optional (B, L_img, C, H, W)
        tgt_scale= x["target_scale"]

        B, L, _ = enc_cont.shape
        H        = dec_cont.shape[1]

        # 1) static features = time‐zero slice of both cont & cat
        static_cont = enc_cont[:, 0, :]       # (B, n_real)
        static_cat  = enc_cat[:,  0, :]       # (B, n_cat)

        # 2) embed cats if needed
        if hasattr(self, "embeddings"):
            # concatenate all cats along time, then pick time=0
            emb_all = self.embeddings(enc_cat)    # (B, L, ∑emb_dim)
            static_cat = emb_all[:, 0, :]          # (B, ∑emb_dim)

        # 3) build x_static
        x_static = torch.cat([static_cont, static_cat], dim=-1)  # (B, D_s)

        # 4) split real covariates into “target” vs “other”
        #    past targets:
        y_hist = enc_cont[..., self.target_positions]            # (B, L, output_dim)
        #    past other reals:
        mask = torch.ones(enc_cont.size(-1), dtype=torch.bool, device=enc_cont.device)
        mask[self.target_positions] = False
        cov_hist   = enc_cont[..., mask]                        # (B, L, D_f_past)
        cov_future = dec_cont[..., mask]                        # (B, H, D_f_future)

        # 5) call your TS encoder
        ts_input = (
            torch.cat([y_hist, cov_hist], dim=-1),  # x_past: (B, L, D_y + D_f_past)
            cov_future,                             # x_future_cov: (B, H, D_f_future)
            x_static                                # x_static:     (B, D_s)
        )
        ts_emb  = self.encoder_ts(ts_input)       # (B, L, E_ts)

        # 6) image pathway
        img_emb = self.encoder_img(imgs, x_static)  # (B, L, E_ts)

        # 7) cross‐attention + SSM
        attn_out = self.cross_attn(ts_emb, img_emb, img_emb)  # (B, L, E_ts)
        ssm_out  = self.mamba_ssm(attn_out)                   # (B, L, E_ts)

        # 8) decoder MLP + head
        dec_emb   = torch.cat([ssm_out, ts_emb], dim=-1)   # (B, L, 2*E_ts)
        dec_flat  = dec_emb.view(B, -1)                    # (B, L*2*E_ts)
        hidden    = self.decoder_mlp(dec_flat)             # (B, temporal_decoder_hidden)
        out_flat  = self.decoder_head(hidden)              # (B, H * decoder_output_dim)
        out_seq   = out_flat.view(B, H, self.decoder_output_dim)
        out_seq   = self.output_layer(out_seq)             # (B, H, output_dim)

        # 9) look‐back skip
        y_hist_flat = y_hist.reshape(B, -1)                        # (B, L*output_dim)
        skip_flat   = self.lookback_skip(y_hist_flat)             # (B, H*output_dim)
        skip_seq    = skip_flat.view(B, H, self.output_dim)       # (B, H, output_dim)

        # 10) combine + un-normalize + wrap
        preds = out_seq + skip_seq                                # (B, H, output_dim)
        preds = self.transform_output(preds, target_scale=tgt_scale)
        return self.to_network_output(prediction=preds)


    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesWithImageDataSet,
        **model_kwargs: Any
    ) -> "Timage":
        """
        Build a Timage directly from a TimeSeriesWithImageDataSet, pulling in
        all its encoding parameters automatically.
        """
        # 1) pull all TimeSeriesDataSet params
        params = dataset.get_parameters()

        # 2) ensure we use the same target scaler
        model_kwargs.setdefault("output_transformer", dataset.target_normalizer)

        # 3) pass through dataset params for BaseModelWithCovariates
        model_kwargs.setdefault("dataset_parameters", params)

        # 4) automatically infer which features are categorical
        #    (so MultiEmbedding can pick them up without manual x_categoricals)
        cats = []
        cats += params.get("static_categoricals", []) or []
        cats += params.get("time_varying_known_categoricals", []) or []
        cats += params.get("time_varying_unknown_categoricals", []) or []
        if cats:
            model_kwargs.setdefault("x_categoricals", cats)

        # 5) if your DataSet carries an image_shape attr, pull it too
        if hasattr(dataset, "image_shape") and dataset.image_shape is not None:
            model_kwargs.setdefault("image_shape", dataset.image_shape)

        # 6) instantiate
        net = cls(**model_kwargs)

        # 7) sanity‐check your loss matches single vs multi target
        if dataset.multi_target:
            assert isinstance(net.loss, MultiLoss), "Expected MultiLoss for multi_target"
        else:
            assert not isinstance(net.loss, MultiLoss), "Expected single‐target loss"

        return net
