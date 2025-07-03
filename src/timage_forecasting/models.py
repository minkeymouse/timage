"""
Timage for image integrated time series forecasting.
"""
from copy import copy
from typing import Optional, Union, Any

import torch
from torch import nn

from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE
from pytorch_forecasting.models.base import BaseModelWithCovariates
from pytorch_forecasting.models.nn.embeddings import MultiEmbedding

from timage_forecasting.datamodule import TimeSeriesWithImageDataModule
from timage_forecasting.sub_modules import *


class Timage(BaseModelWithCovariates):
    """Image integrated time series model for long-term forecasting."""
    def __init__(
        self,
        output_chunk_length: int,
        input_chunk_length: int,
        num_encoder_layers_ts: int = 2,
        num_encoder_layers_img: int = 2,
        image_shape: Optional[tuple[int, int, int]] = None,
        num_decoder_layers: int = 2,
        decoder_output_dim: int = 16,
        hidden_size_ts: int = 128,
        hidden_size_img: int = 128,
        use_layer_norm: bool = False,
        dropout: float = 0.1,
        output_size: Union[int, list[int]] = 1,
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

        logging_metrics: nn.ModuleList = None,
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
            embedding_labels = {}
        if x_reals is None:
            x_reals = []
        if x_categoricals is None:
            x_categoricals = []
        if x_image is None:
            x_image = []
        if logging_metrics is None:
            logging_metrics = nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE()])

        # loss and logging_metrics are ignored as they are modules
        # and stored before calling save_hyperparameters
        self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        super().__init__(logging_metrics=logging_metrics, **kwargs)
        self.output_dim = len(self.target_names)

        self.embeddings = MultiEmbedding(
            embedding_sizes=self.hparams.embedding_sizes,
            categorical_groups=self.hparams.categorical_groups,
            embedding_paddings=self.hparams.embedding_paddings,
            x_categoricals=self.hparams.x_categoricals,
        )

        self.model = _TideModule(
            output_dim=self.output_dim,
            future_cov_dim=self.encoder_covariate_size,
            static_cov_dim=self.static_size,
            output_chunk_length=output_chunk_length,
            input_chunk_length=input_chunk_length,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            decoder_output_dim=decoder_output_dim,
            hidden_size=hidden_size,
            temporal_decoder_hidden=temporal_decoder_hidden,
            temporal_width_future=temporal_width_future,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
            temporal_hidden_size_future=temporal_hidden_size_future,
        )

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

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        """
        Convenience function to create network from
        :py:class`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Args:
            dataset (TimeSeriesDataSet): dataset where sole predictor is the target.
            **kwargs: additional arguments to be passed to `__init__` method.

        Returns:
            TiDE
        """

        # validate arguments
        assert not isinstance(
            dataset.target_normalizer, NaNLabelEncoder
        ), "only regression tasks are supported - target must not be categorical"

        assert dataset.min_encoder_length == dataset.max_encoder_length, (
            "only fixed encoder length is allowed,"
            " but min_encoder_length != max_encoder_length"
        )

        assert dataset.max_prediction_length == dataset.min_prediction_length, (
            "only fixed prediction length is allowed,"
            " but max_prediction_length != min_prediction_length"
        )

        assert (
            dataset.randomize_length is None
        ), "length has to be fixed, but randomize_length is not None"
        assert (
            not dataset.add_relative_time_idx
        ), "add_relative_time_idx has to be False"

        new_kwargs = copy(kwargs)
        new_kwargs.update(
            {
                "output_chunk_length": dataset.max_prediction_length,
                "input_chunk_length": dataset.max_encoder_length,
            }
        )
        new_kwargs.update(cls.deduce_default_output_parameters(dataset, kwargs, MAE()))
        # initialize class
        return super().from_dataset(dataset, **new_kwargs)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Pass forward of network.
        Returns:
            Dict[str, torch.Tensor]: output of model
        """

        # target
        encoder_y = x["encoder_cont"][..., self.target_positions]
        # covariates
        encoder_features = self.extract_features(x, self.embeddings, period="encoder")

        if self.encoder_covariate_size > 0:
            # encoder_features = self.extract_features(
            #                   x, self.embeddings, period="encoder")
            encoder_x_t = torch.concat(
                [
                    encoder_features[name]
                    for name in self.encoder_variables
                    if name not in self.target_names
                ],
                dim=2,
            )
            input_vector = torch.concat((encoder_y, encoder_x_t), dim=2)

        else:
            encoder_x_t = None
            input_vector = encoder_y

        if self.decoder_covariate_size > 0:
            decoder_features = self.extract_features(
                x, self.embeddings, period="decoder"
            )
            decoder_x_t = torch.concat(
                [decoder_features[name] for name in self.decoder_variables], dim=2
            )
        else:
            decoder_x_t = None

        # statics
        if self.static_size > 0:
            x_s = torch.concat(
                [encoder_features[name][:, 0] for name in self.static_variables], dim=1
            )
        else:
            x_s = None

        x_in = (input_vector, decoder_x_t, x_s)
        prediction = self.model(x_in)

        if self.output_dim > 1:  # for multivariate targets
            # adjust prefictions dimensions according
            # to format required for consequent processes
            # from (batch size, seq len, output_dim) to
            # (output_dim, batch size, seq len)
            prediction = prediction.permute(2, 0, 1)
            prediction = [i.clone().detach().requires_grad_(True) for i in prediction]

        # rescale predictions into target space
        prediction = self.transform_output(prediction, target_scale=x["target_scale"])
        # transform output to format processed by other functions
        return self.to_network_output(prediction=prediction)
