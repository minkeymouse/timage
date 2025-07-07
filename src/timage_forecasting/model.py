"""
Timage for image integrated time series forecasting.
"""
from copy import copy
from typing import Optional, Union, Any, Tuple, Dict, List

import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, AUROC, Recall

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

        # Optional parameters
        embedding_sizes:       Optional[dict[str, int]] = None,
        x_categoricals: Optional[List[str]] = None, 
        image_shape: Optional[Tuple[int, int, int]] = None,
        classification: Optional[bool] = False,
        num_classes: Optional[int] = None,
        classification_task: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.loss = RMSE()
        self.logging_metrics = nn.ModuleList([MAE(), MAPE(), MASE(), RMSE(), SMAPE()])
        # finally:
        self.save_hyperparameters(ignore=["loss","logging_metrics"])
        
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
            # F1
            metrics.append(F1Score(task=classification_task,
                                   num_classes=num_classes))
            # AUROC
            metrics.append(AUROC(task=classification_task,
                                 num_classes=num_classes))
            # Recall
            metrics.append(Recall(task=classification_task,
                                  num_classes=num_classes))

            logging_metrics = nn.ModuleList(metrics)

        # 2) regression branch (unchanged)
        else:
            self.loss = RMSE()
            metrics = [SMAPE(), MAE(), RMSE(), MAPE(), MASE()]

        # register them as submodules
        self.logging_metrics = nn.ModuleList(metrics)

        # store flags
        self.classification       = classification
        self.classification_task  = classification_task
        self.num_classes          = num_classes
        self.decoder_output_dim = decoder_output_dim
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.embedding_sizes = embedding_sizes
        self.x_categoricals = x_categoricals
        self.image_shape = image_shape
        self.output_dim = len(self.target_names)
        self.image_backbone = image_backbone
        self.image_pretrained = image_pretrained
        self.classification = classification

        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        

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

        if not self.classification:
            # regression: produce H × decoder_output_dim, then map to output_dim
            self.decoder_head = nn.Sequential(
                nn.Linear(temporal_decoder_hidden,
                          output_chunk_length * decoder_output_dim),
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

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        enc_cat  = x["encoder_cat"]
        enc_cont = x["encoder_cont"]
        dec_cont = x["decoder_cont"]
        imgs     = x.get("x_image", None)
        tgt_scale= x["target_scale"]

        B, L, _ = enc_cont.shape
        H       = dec_cont.shape[1]

        # static features
        static_cont = enc_cont[:, 0, :]
        static_cat  = enc_cat[:,  0, :]
        if hasattr(self, "embeddings"):
            emb_all    = self.embeddings(enc_cat)
            static_cat = emb_all[:, 0, :]
        x_static = torch.cat([static_cont, static_cat], dim=-1)

        # split targets vs covariates
        y_hist = enc_cont[..., self.target_positions]
        mask   = torch.ones(enc_cont.size(-1), dtype=torch.bool, device=enc_cont.device)
        mask[self.target_positions] = False
        cov_hist   = enc_cont[..., mask]
        cov_future = dec_cont[..., mask]

        # encode time‐series & images
        ts_emb  = self.encoder_ts((torch.cat([y_hist, cov_hist], dim=-1), cov_future, x_static))
        img_emb = self.encoder_img(imgs, x_static)

        # fuse + SSM
        attn_out = self.cross_attn(ts_emb, img_emb, img_emb)
        ssm_out  = self.mamba_ssm(attn_out)

        # decoder MLP
        dec_emb  = torch.cat([ssm_out, ts_emb], dim=-1).view(B, -1)
        hidden   = self.decoder_mlp(dec_emb)

        if self.classification:
            # classification head sees a single vector per sample
            logits = self.classification_head(hidden)  # (B, num_classes)
            return {"prediction": logits}
        else:
            # regression head: reshape → output_dim, add skip, un-normalize
            out_flat = self.decoder_head(hidden)             # (B, H*decoder_output_dim)
            out_seq  = out_flat.view(B, H, self.decoder_output_dim)
            preds_ts = self.output_layer(out_seq)            # (B, H, output_dim)

            # look-back skip
            y_hist_flat = y_hist.reshape(B, -1)
            skip_flat   = self.lookback_skip(y_hist_flat)
            skip_seq    = skip_flat.view(B, H, self.output_dim)

            preds = preds_ts + skip_seq
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
        all its encoding parameters automatically—and if classification=True,
        force output_chunk_length=1.
        """
        # 1) pull all TimeSeriesDataSet params
        params = dataset.get_parameters()

        # 2) ensure we use the same target scaler
        model_kwargs.setdefault("output_transformer", dataset.target_normalizer)

        # 3) pass through dataset params for BaseModelWithCovariates
        model_kwargs.setdefault("dataset_parameters", params)

        # 4) infer categoricals
        cats: List[str] = []
        cats += params.get("static_categoricals", []) or []
        cats += params.get("time_varying_known_categoricals", []) or []
        cats += params.get("time_varying_unknown_categoricals", []) or []
        if cats:
            model_kwargs.setdefault("x_categoricals", cats)

        # 5) pull through image_shape if it exists on the dataset
        if hasattr(dataset, "image_shape") and dataset.image_shape is not None:
            model_kwargs.setdefault("image_shape", dataset.image_shape)

        # 6) enforce classification horizon = 1
        if model_kwargs.get("classification", False):
            specified = model_kwargs.get("output_chunk_length", None)
            if specified is not None and specified != 1:
                raise ValueError(
                    f"output_chunk_length must be 1 for classification tasks, "
                    f"but got output_chunk_length={specified}"
                )
            model_kwargs["output_chunk_length"] = 1

        # 7) instantiate the network
        net = cls(**model_kwargs)

        # 8) sanity‐check your loss matches single vs multi target
        if dataset.multi_target:
            assert isinstance(net.loss, MultiLoss), "Expected MultiLoss for multi_target"
        else:
            assert not isinstance(net.loss, MultiLoss), "Expected single-target loss"

        return net

