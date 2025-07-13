"""
Timage for image integrated time series forecasting.
"""
from typing import Optional, Any, Tuple, Dict, List, Union

import torch
from torch import nn
from torch.optim import Optimizer

from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.models.tide.sub_modules import _ResidualBlock
from pytorch_forecasting.models.timexer.sub_modules import AttentionLayer, FullAttention

from timage_forecasting.base import BaseModel
from timage_forecasting.dataset import TimeSeriesWithImage
from timage_forecasting.datamodule import EncoderDecoderTimeSeriesDataModule
from timage_forecasting.sub_modules import _TimeSeriesEncoder, _TemporalImageEncoder

import warnings

class Timage(BaseModel):
    """Image integrated time series model for long-term forecasting."""
    def __init__(
        self,
        embedding_size: int,
        encoder_layers: int,
        encoder_hidden_size: int,
        decoder_layers: int,  
        decoder_hidden_size: int,
        loss: nn.Module,
        use_layer_norm: bool = True,
        dropout:                  float = 0.1,
        image_backbone: str = "tf_efficientnetv2_b0.in1k",
        image_pretrained: bool = True,
        metadata: dict = {},

        classification: bool = False,
        image_size: Tuple[int, int, int] = (1,24,24),

        logging_metrics: Optional[list[nn.Module]] = None,
        optimizer: Optional[Union[Optimizer, str]] = "adam",
        optimizer_params: Optional[dict] = None,
        lr_scheduler: Optional[str] = None,
        lr_scheduler_params: Optional[dict] = None,
    ):
        super().__init__(
            loss=loss,
            logging_metrics=logging_metrics,
            optimizer=optimizer,
            optimizer_params=optimizer_params or {},
            lr_scheduler=lr_scheduler,
            lr_scheduler_params=lr_scheduler_params or {},
        )

        # store hyperparameters
        self.embedding_size = embedding_size
        self.encoder_layers = encoder_layers
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_layers = decoder_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.image_backbone = image_backbone
        self.image_pretrained = image_pretrained
        self.classification = classification
        self.image_size = image_size
        self.metadata = metadata

        # extract dims from metadata
        self.target_dim = metadata["target"]
        self.enc_cat_dim = metadata["encoder_cat"]
        self.enc_cont_dim = metadata["encoder_cont"]
        self.dec_cat_dim = metadata["decoder_cat"]
        self.dec_cont_dim = metadata["decoder_cont"]
        self.cat_static = metadata.get("static_categorical_features", 0)
        self.cont_static = metadata.get("static_continuous_features", 0)

        # sequence lengths
        self.L = metadata["max_encoder_length"]
        self.H = metadata["max_prediction_length"]

        # instantiate the time series encoder
        self.encoder_ts = _TimeSeriesEncoder(
            target_dim=self.target_dim,
            enc_cat_dim=self.enc_cat_dim,
            enc_cont_dim=self.enc_cont_dim,
            dec_cat_dim=self.dec_cat_dim,
            dec_cont_dim=self.dec_cont_dim,
            cat_static_dim=self.cat_static,
            cont_static_dim=self.cont_static,
            input_chunk_length=self.L,
            output_chunk_length=self.H,
            encoder_layers=self.encoder_layers,
            hidden_size=self.encoder_hidden_size,
            embed_dim=self.embedding_size,
            use_layer_norm=self.use_layer_norm,
            dropout=self.dropout,
        )

        self.encoder_img = _TemporalImageEncoder(
            embed_dim=hidden_size_ts,
            static_cov_dim=static_cov_dim,
            input_chunk_length=self.input_chunk_length,
            backbone_name = self.image_backbone,
            pretrained = self.image_pretrained,
            in_chans=self.image_size[0],
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        d_model = self.encoder_hidden_size
        n_heads = 8  
        self.cross_attn = AttentionLayer(
            FullAttention(mask_flag=False, attention_dropout=dropout, output_attention=False),
            d_model=d_model,
            n_heads=n_heads,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_hidden_size,
            nhead=8,
            dim_feedforward=hidden_size_ts * 4,
            dropout=dropout,
            batch_first=True,
        )
        
        self.state_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        # Linear layer for lookback skip connection (maps past target sequence to forecast)
        self.lookback_skip = nn.Linear(self.input_chunk_length * self.output_dim, self.output_chunk_length * self.output_dim)

        self.decoder_mlp = nn.Sequential(
            _ResidualBlock(input_dim=self.input_chunk_length * 2*hidden_size_ts,  # because we concat SSM + TS
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

        if not self.classification:
            # regression: produce H × decoder_output_dim, then map to output_dim
            self.decoder_head = nn.Sequential(
                nn.Linear(temporal_decoder_hidden,
                          self.output_chunk_length * decoder_output_dim),
                nn.Dropout(dropout),
            )
            self.output_layer = nn.Linear(decoder_output_dim,
                                          self.output_dim)
        else:
            # classification: collapse H→1 (or just ignore horizon),
            # then produce `num_classes` logits
            #  - if `output_chunk_length>1` you could pick the last step
            self.classification_head = nn.Sequential(
                # take the hidden vector (size temporal_decoder_hidden)
                # → num_classes logits
                nn.Linear(temporal_decoder_hidden, self.num_classes)
            )


    def training_step(
        self,
        batch: Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
        batch_idx: int
    ):
        x, (y, w) = batch
        out = self(x)
        preds = out["prediction"]

        if self.classification:
            # --- prepare labels and compute loss per task ---
            if self.classification_task == "multiclass":
                # preds: (B, C), y: (B,1) or (B,)
                y_true = y.squeeze(-1).long()
                loss   = self.loss(preds, y_true)
                probs  = preds.softmax(dim=-1)

            elif self.classification_task == "binary":
                # preds: (B,1), y: (B,1) or (B,)
                logits = preds.squeeze(-1)
                y_true = y.squeeze(-1).float()
                loss   = self.loss(logits, y_true)
                probs  = logits.sigmoid()

            else:  # multilabel
                # preds: (B, C), y: (B, C)
                y_true = y.float()
                loss   = self.loss(preds, y_true)
                probs  = preds.sigmoid()

            # --- log metrics (all expect probabilities + true labels) ---
            for metric in self.logging_metrics:
                name = metric.__class__.__name__.lower()
                self.log(f"train_{name}", metric(probs, y_true),
                         on_step=False, on_epoch=True)

        else:
            # forecasting: preds (B, H, D), y (B, H, D), w (B, H)
            loss = self.loss(preds, y, w)
            for metric in self.logging_metrics:
                name = metric.__class__.__name__.lower()
                self.log(f"train_{name}", metric(preds, y),
                         on_step=False, on_epoch=True)

        # finally log loss for both modes
        self.log("train_loss", loss,
                 prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def forward(self, x):  # drop the NetworkOutput hint
        # unpack
        y_hist    = x["encoder_target"]    # (B,L) or (B,L,D_y)
        cont_hist = x["encoder_cont"]      # (B,L,C_cont)
        cont_fut  = x["decoder_cont"]      # (B,H,C_cont)
        cat_hist  = x.get("encoder_cat")   # (B,L,C_cat) or None
        cat_fut   = x.get("decoder_cat")   # (B,H,C_cat) or None
        imgs      = x.get("x_image")       # optional image seq
        tgt_scale = x["target_scale"]

        B, L, _ = cont_hist.shape
        H       = cont_fut.size(1)

        # ensure y is 3-D
        if y_hist.ndim == 2:
            y_hist = y_hist.unsqueeze(-1)
        D_y = y_hist.size(-1)

        # static features
        static_cont = cont_hist[:, 0]
        static_cat  = cat_hist[:, 0].float() if cat_hist is not None else None
        x_static    = (
            torch.cat([static_cont, static_cat], dim=-1)
            if static_cat is not None
            else static_cont
        )

        # build covariates: continuous + (float) categoricals
        cov_hist = [cont_hist]
        cov_fut  = [cont_fut]
        if cat_hist is not None:
            cov_hist.append(cat_hist.float())
            cov_fut.append(cat_fut.float())
        cov_hist = torch.cat(cov_hist, dim=-1)  # (B, L, C_cov)
        cov_fut  = torch.cat(cov_fut,  dim=-1)  # (B, H, C_cov)

        # sanity check
        assert cov_hist.size(-1) == cov_fut.size(-1), (
            f"cov dim mismatch {cov_hist.size(-1)} vs {cov_fut.size(-1)}"
        )

        # series input
        series_input = torch.cat([y_hist, cov_hist], dim=-1)

        # run your encoders
        ts_emb  = self.encoder_ts((series_input, cov_fut, x_static))
        img_emb = (
            self.encoder_img(imgs, x_static) if imgs is not None else None
        )

        # cross-attend (if you have images)
        fusion = (
            self.cross_attn(ts_emb, img_emb, img_emb)
            if img_emb is not None
            else ts_emb
        )
        state = self.state_encoder(fusion)

        # decode
        flat    = torch.cat([state, ts_emb], dim=-1).reshape(B, -1)
        hidden  = self.decoder_mlp(flat)
        head    = self.decoder_head(hidden)                    # (B, H * D_dec)
        seq     = head.view(B, H, self.decoder_output_dim)     # (B,H,D_dec)
        out_ts  = self.output_layer(seq)                       # (B,H,out_dim)

        # lookback skip
        skip_in = y_hist.reshape(B, -1)
        skip    = self.lookback_skip(skip_in).view(B, H, self.output_dim)

        preds = self.transform_output(out_ts + skip, tgt_scale)
        return self.to_network_output(prediction=preds)


    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesWithImageDataSet,
        **kwargs,  # your model‐hyperparameters
    ) -> "Timage":
        # 1) grab the full params
        full_params = dataset.get_parameters()
        print("full parameters:", full_params)
        # 2) extract & remove the image keys
        image_cols_start = full_params.pop("image_cols_start")
        image_cols_end   = full_params.pop("image_cols_end")
        image_shape      = tuple(full_params.pop("image_shape"))
        # 3) grab the target normalizer
        transformer = dataset.target_normalizer
        # 4) now call __init__, passing clean TS params + image bits
        return cls(
            dataset_parameters = full_params,
            output_transformer = transformer,
            image_cols_start   = image_cols_start,
            image_cols_end     = image_cols_end,
            image_shape        = image_shape,
            **kwargs,  # num_encoder_layers_ts, hidden_size_ts, …
        )
    
    @property
    def static_variables(self) -> list[str]:
        """List of all static variables in model"""
        return self.dataset_parameters["static_categoricals"] + self.dataset_parameters["static_reals"]

    @property
    def decoder_variables(self) -> list[str]:
        """List of all decoder variables in model (excluding static variables)"""
        return (
            self.dataset_parameters["time_varying_categoricals_decoder"]
            + self.dataset_parameters["time_varying_reals_decoder"]
        )