"""
Timage for image integrated time series forecasting.
"""
from typing import Optional, Any, Tuple, Dict, List

import torch
from torch import nn
from torchmetrics import Accuracy

from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE
from pytorch_forecasting.models.base import BaseModel, BaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding
from pytorch_forecasting.models.tide.sub_modules import _ResidualBlock
from pytorch_forecasting.models.timexer.sub_modules import AttentionLayer, FullAttention

from timage_forecasting.dataset import TimeSeriesWithImageDataSet
from timage_forecasting.sub_modules import _TimeSeriesEncoder, _TemporalImageEncoder

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
    category=UserWarning,
    module="sklearn.utils.validation",
)

class Timage(BaseModelWithCovariates):
    """Image integrated time series model for long-term forecasting."""
    def __init__(
        self,
        dataset_parameters: Dict[str, Any],
        output_transformer: Any,

        image_cols_start: str,
        image_cols_end:   str,
        image_shape:      Tuple[int,int,int],

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
        embedding_sizes:       Optional[dict[str, int]] = None,
        x_categoricals: Optional[List[str]] = None, 
        classification: Optional[bool] = False,
        num_classes: Optional[int] = None,
        classification_task: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            dataset_parameters = dataset_parameters,
            output_transformer = output_transformer,
            **{})
        
        # 1) classification branch
        if classification:
            if classification_task not in {"binary", "multiclass", "multilabel"}:
                raise ValueError(f"Unsupported task {classification_task!r}")

            # ensure num_classes for anything beyond binary
            if classification_task in {"multiclass", "multilabel"} and num_classes is None:
                raise ValueError("num_classes must be set for multiclass/multilabel")

            # pick loss
            if classification_task == "multiclass":
                self.loss = nn.CrossEntropyLoss()
            else:
                # binary or multilabel
                self.loss = nn.BCEWithLogitsLoss()

            # build metrics list
            metrics = []
            # Accuracy
            metrics.append(Accuracy(task=classification_task,
                                    num_classes=num_classes))
            
        else:
            self.loss = RMSE()
            metrics = [SMAPE(), MAE(), RMSE(), MAPE(), MASE()]

        # register them as submodules
        self.logging_metrics = nn.ModuleList(metrics)

        # store 
        self.dataset_parameters = dataset_parameters
        self.output_dim = len(dataset_parameters["target"])
        self.input_chunk_length = dataset_parameters["max_encoder_length"]
        self.output_chunk_length = dataset_parameters["max_prediction_length"]

        self.num_encoder_layers_ts = num_encoder_layers_ts
        self.hidden_size_ts = hidden_size_ts
        self.num_decoder_layers = num_decoder_layers
        self.decoder_output_dim = decoder_output_dim


        self.temporal_width_future = temporal_width_future
        self.temporal_hidden_size_future = temporal_hidden_size_future
        self.temporal_decoder_hidden = temporal_decoder_hidden


        self.use_layer_norm = use_layer_norm
        self.dropout = dropout
        self.image_backbone = image_backbone
        self.image_pretrained = image_pretrained
        self.image_cols_start = image_cols_start
        self.image_cols_end   = image_cols_end
        self.image_shape      = image_shape

        self.embedding_sizes = embedding_sizes

        # Categoricals
        self.x_categoricals = x_categoricals
        self.classification = classification
        self.num_classes = num_classes
        self.classification_task = classification_task

        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        
        if self.embedding_sizes and self.x_categoricals is not None:
            self.embeddings = MultiEmbedding(
                embedding_sizes=self.embedding_sizes,
                x_categoricals=self.x_categoricals,
            )

        static_cov_dim = len(self.static_variables)
        
        # how many real + categorical covariates you actually feed
        cont_dim = len(dataset_parameters["time_varying_unknown_reals"]) \
                 + len(dataset_parameters["time_varying_known_reals"])
        cat_dim  = len(dataset_parameters["time_varying_unknown_categoricals"]) \
                 + len(dataset_parameters["time_varying_known_categoricals"])
        future_cov_dim = cont_dim + cat_dim

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
            input_chunk_length=self.input_chunk_length,
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size_ts,
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