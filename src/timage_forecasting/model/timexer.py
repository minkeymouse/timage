"""
Implementation of `nn.Modules` for TimeXer model from pytorch-forecasting.
Only for experiment purpose.
"""

import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import copy
from typing import Optional, Union
import warnings as warn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import (
    MAE,
    MAPE,
    RMSE,
    SMAPE,
    MultiHorizonMetric,
    QuantileLoss,
)
from pytorch_forecasting.metrics.base_metrics import MultiLoss
from pytorch_forecasting.models.base import BaseModelWithCovariates
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class TriangularCausalMask:
    """
    Triangular causal mask for attention mechanism.
    """

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """
    Full attention mechanism with optional masking and dropout.
    Args:
        mask_flag (bool): Whether to apply masking.
        factor (int): Factor for scaling the attention scores.
        scale (float): Scaling factor for attention scores.
        attention_dropout (float): Dropout rate for attention scores.
        output_attention (bool): Whether to output attention weights."""

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.abs)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    """
    Attention layer that combines query, key, and value projections with an attention
    mechanism.
    Args:
        attention (nn.Module): Attention mechanism to use.
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        d_keys (int, optional): Dimension of the keys. Defaults to d_model // n_heads.
        d_values (int, optional):
            Dimension of the values. Defaults to d_model // n_heads.
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        if S == 0:
            # skip the cross attention process since there is no exogenous variables
            queries = self.query_projection(queries)
            return self.out_projection(queries), None

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class DataEmbedding_inverted(nn.Module):
    """
    Data embedding module for time series data.
    Args:
        c_in (int): Number of input features.
        d_model (int): Dimension of the model.
        embed_type (str): Type of embedding to use. Defaults to "fixed".
        freq (str): Frequency of the time series data. Defaults to "h".
        dropout (float): Dropout rate. Defaults to 0.1.
    """

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class PositionalEmbedding(nn.Module):
    """
    Positional embedding module for time series data.
    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the input sequence. Defaults to 5000."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class FlattenHead(nn.Module):
    """
    Flatten head for the output of the model.
    Args:
        n_vars (int): Number of input features.
        nf (int): Number of features in the last layer.
        target_window (int): Target window size.
        head_dropout (float): Dropout rate for the head. Defaults to 0.
        n_quantiles (int, optional): Number of quantiles. Defaults to None."""

    def __init__(self, n_vars, nf, target_window, head_dropout=0, n_quantiles=None):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.n_quantiles = n_quantiles

        if self.n_quantiles is not None:
            self.linear = nn.Linear(nf, target_window * n_quantiles)
        else:
            self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)

        if self.n_quantiles is not None:
            batch_size, n_vars = x.shape[0], x.shape[1]
            x = x.reshape(batch_size, n_vars, -1, self.n_quantiles)
        return x


class EnEmbedding(nn.Module):
    """
    Encoder embedding module for time series data. Handles endogenous feature
    embeddings in this case.
    Args:
        n_vars (int): Number of input features.
        d_model (int): Dimension of the model.
        patch_len (int): Length of the patches.
        dropout (float): Dropout rate. Defaults to 0.1."""

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()

        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """
    Encoder module for the TimeXer model.
    Args:
        layers (list): List of encoder layers.
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.
        projection (nn.Module, optional): Projection layer. Defaults to None.
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    """
    Encoder layer for the TimeXer model.
    Args:
        self_attention (nn.Module): Self-attention mechanism.
        cross_attention (nn.Module): Cross-attention mechanism.
        d_model (int): Dimension of the model.
        d_ff (int, optional):
            Dimension of the feedforward layer. Defaults to 4 * d_model.
        dropout (float): Dropout rate. Defaults to 0.1.
        activation (str): Activation function. Defaults to "relu".
    """

    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        x_glb_attn = self.dropout(
            self.cross_attention(
                x_glb, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )
        x_glb_attn = torch.reshape(
            x_glb_attn, (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])
        ).unsqueeze(1)
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


"""
Time Series Transformer with eXogenous variables (TimeXer)
---------------------------------------------------------
"""



class TimeXer(BaseModelWithCovariates):
    """TimeXer model for time series forecasting with exogenous variables."""

    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        task_name: str = "long_term_forecast",
        features: str = "MS",
        enc_in: int = None,
        hidden_size: int = 256,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 1024,
        dropout: float = 0.2,
        activation: str = "relu",
        patch_length: int = 16,
        factor: int = 5,
        embed_type: str = "fixed",
        freq: str = "h",
        output_size: Union[int, list[int]] = 1,
        loss: MultiHorizonMetric = None,
        learning_rate: float = 1e-3,
        static_categoricals: Optional[list[str]] = None,
        static_reals: Optional[list[str]] = None,
        time_varying_categoricals_encoder: Optional[list[str]] = None,
        time_varying_categoricals_decoder: Optional[list[str]] = None,
        time_varying_reals_encoder: Optional[list[str]] = None,
        time_varying_reals_decoder: Optional[list[str]] = None,
        x_reals: Optional[list[str]] = None,
        x_categoricals: Optional[list[str]] = None,
        embedding_sizes: Optional[dict[str, tuple[int, int]]] = None,
        embedding_labels: Optional[list[str]] = None,
        embedding_paddings: Optional[list[str]] = None,
        categorical_groups: Optional[dict[str, list[str]]] = None,
        logging_metrics: nn.ModuleList = None,
        **kwargs,
    ):
        """An implementation of the TimeXer model.

        TimeXer empowers the canonical transformer with the ability to reconcile
        endogenous and exogenous information without any architectural modifications
        and achieves consistent state-of-the-art performance across twelve real-world
        forecasting benchmarks.

        TimeXer employs patch-level and variate-level representations respectively for
        endogenous and exogenous variables, with an endogenous global token as a bridge
        in-between. With this design, TimeXer can jointly capture intra-endogenous
        temporal dependencies and exogenous-to-endogenous correlations.

        TimeXer model for time series forecasting with exogenous variables.

        Parameters
        ----------
        context_length (int): Length of input sequence used for making predictions.
        prediction_length (int): Number of future time steps to predict.
        task_name (str, optional): Type of forecasting task, either
            'long_term_forecast' or 'short_term_forecast', which corresponds to
            forecasting scenarios implied by the task names.
        features (str, optional): Type of features used in the model ('MS' for
            multivariate forecating with single target, 'M' for multivariate
            forecasting with multiple targets and 'S' for univariate forecasting).
        enc_in (int, optional): Number of input variables for encoder.
        hidden_size (int, optional): Dimension of model embeddings and hidden
            representations.
        n_heads (int, optional): Number of attention heads in multi-head attention
            layers.
        e_layers (int, optional): Number of encoder layers with dual attention
            mechanism.
        d_ff (int, optional): Dimension of feedforward network in transformer layers
        dropout (float, optional): Dropout rate applied throughout the network for
            regularization.
        activation (str, optional): Activation function used in feedforward networks
            ('relu' or 'gelu').
        patch_length (int, optional): Length of each non-overlapping patch for
            endogenous variable tokenization.
        use_norm (bool, optional): Whether to apply normalization to input data.
            Do not change, as it a setting controlled by the pytorch-forecasting API
        factor: Scaling factor for attention scores.
        embed_type: Type of time feature embedding ('timeF' for time-based features)
        freq: Frequency of the time series data('h' for hourly,'d' for daily, etc.).
        static_categoricals (list[str]): names of static categorical variables
        static_reals (list[str]): names of static continuous variables
        time_varying_categoricals_encoder (list[str]): names of categorical
            variables for encoder
        time_varying_categoricals_decoder (list[str]): names of categorical
            variables for decoder
        time_varying_reals_encoder (list[str]): names of continuous variables for
            encoder
        time_varying_reals_decoder (list[str]): names of continuous variables for
            decoder
        x_reals (list[str]): order of continuous variables in tensor passed to
            forward function
        x_categoricals (list[str]): order of categorical variables in tensor passed
            to forward function
        embedding_sizes (dict[str, tuple[int, int]]): dictionary mapping categorical
            variables to tuple of integers where the first integer denotes the
            number of categorical classes and the second the embedding size
        embedding_labels (dict[str, list[str]]): dictionary mapping (string) indices
            to list of categorical labels
        embedding_paddings (list[str]): names of categorical variables for which
            label 0 is always mapped to an embedding vector filled with zeros
        categorical_groups (dict[str, list[str]]): dictionary of categorical
            variables that are grouped together and can also take multiple values
            simultaneously (e.g. holiday during octoberfest). They should be
            implemented as bag of embeddings.
        logging_metrics (nn.ModuleList[LightningMetric]): list of metrics that are
            logged during training. Defaults to
            nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()]).
        **kwargs: additional arguments to :py:class:`~BaseModel`.
        """

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
            embedding_labels = {}
        if x_reals is None:
            x_reals = []
        if x_categoricals is None:
            x_categoricals = []
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE()])
        if loss is None:
            if features == "M":
                loss = MultiLoss([RMSE()] * len(self.target_positions), weights=[0.3,0.5,0.2])
            else:
                loss = MAE()
        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        # loss is a standalone module and is stored separately.
        super().__init__(loss=loss, logging_metrics=logging_metrics, **kwargs)

        if self.hparams.context_length < self.hparams.patch_length:
            raise ValueError(
                f"context_length ({context_length}) must be greater than or equal to"
                f" patch_length ({patch_length}). Model cannot create patches larger"
                " than the sequence length."
            )

        if self.hparams.context_length % self.hparams.patch_length != 0:
            warn.warn(
                f"In the input sequence, the context_length ({context_length}) is not a"
                f" multiple of the patch_length ({patch_length}). This may lead to some"
                "patches being ignored during training."
            )

        self.patch_num = max(
            1, int(self.hparams.context_length // self.hparams.patch_length)
        )
        self.n_target_vars = len(self.target_positions)

        self.enc_in = enc_in
        if enc_in is None:
            self.enc_in = len(self.reals)

        self.n_quantiles = None

        if isinstance(loss, QuantileLoss):
            self.n_quantiles = len(loss.quantiles)

        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads}) "
                f"for the multi-head attention mechanism to work properly."
            )

        self.en_embedding = EnEmbedding(
            self.n_target_vars,
            self.hparams.hidden_size,
            self.hparams.patch_length,
            self.hparams.dropout,
        )

        self.ex_embedding = DataEmbedding_inverted(
            self.hparams.context_length,
            self.hparams.hidden_size,
            self.hparams.embed_type,
            self.hparams.freq,
            self.hparams.dropout,
        )

        if e_layers <= 0:
            raise ValueError(f"e_layers ({e_layers}) must be positive.")
        elif e_layers > 12:
            warn.warn(
                f"e_layers ({e_layers}) is quite high. This might lead to overfitting "
                f"and high computational cost. Consider using 2-6 layers.",
                UserWarning,
            )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.hparams.factor,
                            attention_dropout=self.hparams.dropout,
                            output_attention=False,
                        ),
                        self.hparams.hidden_size,
                        self.hparams.n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            self.hparams.factor,
                            attention_dropout=self.hparams.dropout,
                            output_attention=False,
                        ),
                        self.hparams.hidden_size,
                        self.hparams.n_heads,
                    ),
                    self.hparams.hidden_size,
                    self.hparams.d_ff,
                    dropout=self.hparams.dropout,
                    activation=self.hparams.activation,
                )
                for l in range(self.hparams.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.hparams.hidden_size),
        )
        self.head_nf = self.hparams.hidden_size * (self.patch_num + 1)
        self.head = FlattenHead(
            self.enc_in,
            self.head_nf,
            self.hparams.prediction_length,
            head_dropout=self.hparams.dropout,
            n_quantiles=self.n_quantiles,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset: TimeSeriesDataSet,
        allowed_encoder_known_variable_names: list[str] = None,
        **kwargs,
    ):
        """
        Create model from dataset and set parameters related to covariates.

        Args:
            dataset: timeseries dataset
            allowed_encoder_known_variable_names: list of known variables that are allowed in encoder, defaults to all
            **kwargs: additional arguments such as hyperparameters for model (see ``__init__()``)

        Returns:
            TimeXer
        """  # noqa: E501
        new_kwargs = copy(kwargs)
        new_kwargs.update(
            {
                "context_length": dataset.max_encoder_length,
                "prediction_length": dataset.max_prediction_length,
            }
        )

        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, MAE()))

        return super().from_dataset(
            dataset,
            allowed_encoder_known_variable_names=allowed_encoder_known_variable_names,
            **new_kwargs,
        )

    def _forecast(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forecast for univariate or multivariate with single target (MS) case.

        Args:
            x: Dictionary containing entries for encoder_cat, encoder_cont
        """
        encoder_cont = x["encoder_cont"]
        encoder_time_idx = x.get("encoder_time_idx", None)
        target_pos = self.target_positions

        # masking to ignore the target variable
        mask = torch.ones(encoder_cont.shape[-1], dtype=torch.bool)
        mask[target_pos] = False
        exog_data = encoder_cont[..., mask]

        en_embed, n_vars = self.en_embedding(
            encoder_cont[:, :, target_pos[-1]].unsqueeze(-1).permute(0, 2, 1)
        )
        ex_embed = self.ex_embedding(exog_data, encoder_time_idx)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        if self.n_quantiles is not None:
            dec_out = dec_out.permute(0, 2, 1, 3)
        else:
            dec_out = dec_out.permute(0, 2, 1)

        return dec_out

    def _forecast_multi(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forecast for multivariate with multiple targets (M) case.

        Args:
            x: Dictionary containing entries for encoder_cat, encoder_cont
        Returns:
            Dictionary with predictions
        """

        encoder_cont = x["encoder_cont"]
        encoder_time_idx = x.get("encoder_time_idx", None)
        target_pos = self.target_positions
        encoder_target = encoder_cont[..., target_pos]

        en_embed, n_vars = self.en_embedding(encoder_target.permute(0, 2, 1))

        # use masking to ignore the target variable in encoder_cont under ex_embed.
        mask = torch.ones(
            encoder_cont.shape[-1], dtype=torch.bool, device=encoder_cont.device
        )
        mask[target_pos] = False
        exog_data = encoder_cont[..., mask]
        ex_embed = self.ex_embedding(exog_data, encoder_time_idx)

        # batch_size x sequence_length x hidden_size
        enc_out = self.encoder(en_embed, ex_embed)

        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )  # batch_size x n_vars x sequence_length x hidden_size

        enc_out = enc_out.permute(0, 1, 3, 2)

        dec_out = self.head(enc_out)
        if self.n_quantiles is not None:
            dec_out = dec_out.permute(0, 2, 1, 3)
        else:
            dec_out = dec_out.permute(0, 2, 1)

        return dec_out

    @property
    def decoder_covariate_size(self) -> int:
        """Decoder covariates size.

        Returns:
            int: size of time-dependent covariates used by the decoder
        """
        return len(
            set(self.hparams.time_varying_reals_decoder) - set(self.target_names)
        ) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_decoder
        )

    @property
    def encoder_covariate_size(self) -> int:
        """Encoder covariate size.

        Returns:
            int: size of time-dependent covariates used by the encoder
        """
        return len(
            set(self.hparams.time_varying_reals_encoder) - set(self.target_names)
        ) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.time_varying_categoricals_encoder
        )

    @property
    def static_size(self) -> int:
        """Static covariate size.

        Returns:
            int: size of static covariates
        """
        return len(self.hparams.static_reals) + sum(
            self.embeddings.output_size[name]
            for name in self.hparams.static_categoricals
        )

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x: Dictionary containing model inputs

        Returns:
            Dictionary with model outputs
        """
        if (
            self.hparams.task_name == "long_term_forecast"
            or self.hparams.task_name == "short_term_forecast"
        ):  # noqa: E501
            if self.hparams.features == "M":
                out = self._forecast_multi(x)
            else:
                out = self._forecast(x)
            prediction = out[:, : self.hparams.prediction_length, :]

            target_positions = self.target_positions

            # note: prediction.size(2) is the number of target variables i.e n_targets
            target_indices = range(prediction.size(2))

            if prediction.size(2) != len(target_positions):
                prediction = prediction[:, :, : len(target_positions)]

            # In the case of a single target, the result will be a torch.Tensor
            # with shape (batch_size, prediction_length)
            # In the case of multiple targets, the result will be a list of "n_targets"
            # tensors with shape (batch_size, prediction_length)
            # If quantile predictions are used, the result will have an additional
            # dimension for quantiles, resulting in a shape of
            # (batch_size, prediction_length, n_quantiles)
            if self.n_quantiles is not None:
                # quantile predictions.
                if len(target_indices) == 1:
                    prediction = prediction[..., 0, :]
                else:
                    prediction = [prediction[..., i, :] for i in target_indices]
            else:
                # point predictions.
                if len(target_indices) == 1:
                    prediction = prediction[..., 0]
                else:
                    prediction = [prediction[..., i] for i in target_indices]
            prediction = self.transform_output(
                prediction=prediction, target_scale=x["target_scale"]
            )
            return self.to_network_output(prediction=prediction)
        else:
            return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=1,      # restart every window‐advance
            T_mult=1,
            eta_min=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }