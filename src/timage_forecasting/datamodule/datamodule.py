from typing import Any, Optional, Union, cast

from lightning.pytorch import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler, RobustScaler

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
    GroupNormalizer,
    MultiNormalizer
)

from timage_forecasting.utils import _coerce_to_dict
from timage_forecasting.datamodule.dataset import TimeSeriesWithImage

NORMALIZER = Union[TorchNormalizer, EncoderNormalizer, GroupNormalizer, MultiNormalizer]

class TimeSeriesWithImageDataModule(LightningDataModule):
    """
    doc
    """

    def __init__(
        self,

        # Metadata from training dataset
        train_ds: TimeSeriesWithImage,
        val_ds: Optional[TimeSeriesWithImage] = None,
        test_ds: Optional[TimeSeriesWithImage] = None,
        predict_ds: Optional[TimeSeriesWithImage] = None,

        encoder_length: int = 30,
        prediction_length: int = 1,
        min_prediction_idx: Optional[int] = None,
        
        add_target_scales: bool = False,
        target_normalizer: Union[
            NORMALIZER, str, list[NORMALIZER], tuple[NORMALIZER], None
        ] = "auto",
        categorical_encoders: Optional[dict[str, NaNLabelEncoder]] = None,
        scalers: Optional[
            dict[
                str,
                Union[TorchNormalizer, EncoderNormalizer],
            ]
        ] = None,

        batch_size: int = 32,
        num_workers: int = 0,
        train_val_split: tuple = (0.8, 0.2),
        train_val_test_split: tuple = (0.7, 0.15, 0.15),
    ):
        super().__init__()
        # Need dataset
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.val_ds = val_ds
        self.predict_ds = predict_ds
        self.train_ds_metadata = train_ds.get_metadata()

        # Encoder, Decoder spec
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.min_prediction_idx = min_prediction_idx

        # Scaler and Normalizers
        self.add_target_scales = add_target_scales
        self.target_normalizer = target_normalizer
        self.categorical_encoders = categorical_encoders
        self.scalers = scalers

        # Training spec
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.train_val_test_split = train_val_test_split

        if isinstance(target_normalizer, str) and target_normalizer.lower() == "auto":
            self._target_normalizer = TorchNormalizer()
        else:
            self._target_normalizer = target_normalizer

        self._categorical_encoders = _coerce_to_dict(categorical_encoders)
        self._scalers = _coerce_to_dict(scalers)
        self._metadata = None

    def _prepare_metadata(self):
        ds_meta   = self.train_ds_metadata
        cols      = ds_meta["cols"]
        col_type  = ds_meta["col_type"]
        col_known = ds_meta["col_known"]
        image_size= ds_meta["img_size"]

        # encoder (all time-varying) features
        enc_cat  = [c for c in cols["x"] if col_type[c] == "C"]
        enc_cont = [c for c in cols["x"] if col_type[c] == "F"]
        enc_img  = cols["img"]  # all image columns

        # decoder (known at forecast) features
        dec_cat  = [c for c in cols["x"] if col_known[c] == "K" and col_type[c] == "C"]
        dec_cont = [c for c in cols["x"] if col_known[c] == "K" and col_type[c] == "F"]
        dec_img  = cols.get("img_known", [])  # only those images flagged K

        # static
        st_cat   = [c for c in cols["st"] if col_type[c] == "C"]
        st_cont  = [c for c in cols["st"] if col_type[c] == "F"]

        # targets
        target_count = len(cols["y"])

        metadata = {
            "encoder_cat": len(enc_cat),
            "encoder_cont": len(enc_cont),
            "encoder_img": len(enc_img),
            "decoder_cat": len(dec_cat),
            "decoder_cont": len(dec_cont),
            "decoder_img": len(dec_img),
            "static_cat": len(st_cat),
            "static_cont": len(st_cont),
            "target": target_count,
            "image_size": image_size,
            "encoder_length": self.encoder_length,
            "prediction_length": self.prediction_length,
        }
        return metadata

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self._prepare_metadata()
        return self._metadata

    def _preprocess_data(self, series_idx: Union[int, torch.Tensor]) -> dict[str, Any]:
        if isinstance(series_idx, torch.Tensor):
            series_idx = int(series_idx.item())

        sample = self.train_ds[series_idx]
        raw_ts     = sample["x"]        # shape (T, n_time_features)
        raw_img    = sample["img"]      # shape (T, C, H, W)
        raw_y      = sample["y"]        # shape (T,) or (T, n_targets)
        raw_st     = sample["st"]       # shape (n_static,)
        valid_mask = sample["valid_mask"]
        times      = sample["t"]
        cutoff     = sample["cutoff_time"]

        ts_cov  = torch.as_tensor(raw_ts, dtype=torch.float32)
        img_seq = torch.as_tensor(raw_img, dtype=torch.float32)
        y_seq   = torch.as_tensor(raw_y, dtype=torch.float32)
        st_cov  = torch.as_tensor(raw_st, dtype=torch.float32)

        ts_feature_cols = self.train_ds_metadata["cols"]["x"]
        is_cat = [self.train_ds_metadata["col_type"][c]=="C" for c in ts_feature_cols]
        cat_idxs = [i for i,flag in enumerate(is_cat) if flag]
        cont_idxs= [i for i,flag in enumerate(is_cat) if not flag]

        ts_cat = []
        if self.categorical_encoders is not None:
            for idx in cat_idxs:
                col = ts_feature_cols[idx]
                codes = self.categorical_encoders[col].transform(ts_cov[:, idx].numpy())
                ts_cat.append(torch.as_tensor(codes, dtype=torch.long))
        ts_cat = torch.stack(ts_cat, dim=1) if ts_cat else torch.zeros((ts_cov.size(0),0),dtype=torch.long)

        if hasattr(self, "_scalers"):
            ts_cont = []
            for idx in cont_idxs:
                col = ts_feature_cols[idx]
                vals = ts_cov[:, idx].unsqueeze(1).numpy()
                scaled = self._scalers[col].transform(vals)
                ts_cont.append(torch.as_tensor(scaled.squeeze(1), dtype=torch.float32))
            ts_cont = torch.stack(ts_cont, dim=1)
        else:
            ts_cont = ts_cov[:, cont_idxs]

        static_cols = self.train_ds_metadata["cols"]["st"]
        is_cat_st = [self.train_ds_metadata["col_type"][c]=="C" for c in static_cols]
        st_cat = []
        st_cont = []
        for i, col in enumerate(static_cols):
            if is_cat_st[i] and self.categorical_encoders is not None:
                code = self.categorical_encoders[col].transform([st_cov[i].item()])[0]
                st_cat.append(torch.tensor(code, dtype=torch.long))
            else:
                st_cont.append(st_cov[i])
        st_cat  = torch.stack(st_cat).unsqueeze(0)  if st_cat else torch.zeros((1,0),dtype=torch.long)
        st_cont = torch.stack(st_cont).unsqueeze(0) if st_cont else torch.zeros((1,0),dtype=torch.float32)

        # 8) (optional) normalize your target
        if isinstance(self._target_normalizer, TorchNormalizer):
            y_seq = self._target_normalizer.transform(y_seq.unsqueeze(-1)).squeeze(-1)

        return {
            "ts_covariates":     {"categorical": ts_cat,  "continuous": ts_cont},
            "images":            img_seq,
            "target":            y_seq,
            "static_covariates": {"categorical": st_cat,  "continuous": st_cont},
            "group":             sample["group"],
            "valid_mask":        valid_mask,
            "times":             times,
            "cutoff_time":       cutoff,
        }

    class _ProcessedDataset(Dataset):
        """PyTorch Dataset for processed encoder-decoder time series data.

        Parameters
        ----------
        dataset : TimeSeries
            The base time series dataset that provides access to raw data and metadata.
        data_module : EncoderDecoderTimeSeriesDataModule
            The data module handling preprocessing and metadata configuration.
        windows : List[Tuple[int, int, int, int]]
            List of window tuples containing
            (series_idx, start_idx, enc_length, pred_length).
        """

        def __init__(
            self,
            dataset: TimeSeriesWithImage,
            data_module: "TimeSeriesWithImageDataModule",
            windows: list[tuple[int, int, int, int]],
        ):
            self.dataset = dataset
            self.data_module = data_module
            self.windows = windows

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, idx):
            series_idx, start_idx, enc_length, pred_length = self.windows[idx]
            data = self.data_module._preprocess_data(series_idx)

            end_idx = start_idx + enc_length + pred_length
            encoder_indices = slice(start_idx, start_idx + enc_length)
            decoder_indices = slice(start_idx + enc_length, end_idx)

            target_scale = data["target"][encoder_indices]
            target_scale = target_scale[~torch.isnan(target_scale)].abs().mean()
            if torch.isnan(target_scale) or target_scale == 0:
                target_scale = torch.tensor(1.0)

            encoder_mask = (
                data["time_mask"][encoder_indices]
                if "time_mask" in data
                else torch.ones(enc_length, dtype=torch.bool)
            )
            decoder_mask = (
                data["time_mask"][decoder_indices]
                if "time_mask" in data
                else torch.zeros(pred_length, dtype=torch.bool)
            )

            encoder_cat = data["features"]["categorical"][encoder_indices]
            encoder_cont = data["features"]["continuous"][encoder_indices]
            encoder_image = data["images"][encoder_indices]

            features = data["features"]
            metadata = self.data_module.metadata

            known_cat_indices = [
                i
                for i, col in enumerate(metadata["cols"]["x"])
                if metadata["col_type"].get(col) == "C"
                and metadata["col_known"].get(col) == "K"
            ]

            known_cont_indices = [
                i
                for i, col in enumerate(metadata["cols"]["x"])
                if metadata["col_type"].get(col) == "F"
                and metadata["col_known"].get(col) == "K"
            ]

            cat_map = {
                orig_idx: i
                for i, orig_idx in enumerate(self.data_module.categorical_indices)
            }
            cont_map = {
                orig_idx: i
                for i, orig_idx in enumerate(self.data_module.continuous_indices)
            }

            mapped_known_cat_indices = [
                cat_map[idx] for idx in known_cat_indices if idx in cat_map
            ]
            mapped_known_cont_indices = [
                cont_map[idx] for idx in known_cont_indices if idx in cont_map
            ]

            decoder_cat = (
                features["categorical"][decoder_indices][:, mapped_known_cat_indices]
                if mapped_known_cat_indices
                else torch.zeros((pred_length, 0))
            )

            decoder_cont = (
                features["continuous"][decoder_indices][:, mapped_known_cont_indices]
                if mapped_known_cont_indices
                else torch.zeros((pred_length, 0))
            )

            decoder_image   = data["images"][decoder_indices]

            x = {
                "encoder_cat": encoder_cat,
                "encoder_cont": encoder_cont,
                "encoder_img": encoder_image,
                "decoder_cat": decoder_cat,
                "decoder_cont": decoder_cont,
                "decoder_img": decoder_image,
                "encoder_lengths": torch.tensor(enc_length),
                "decoder_lengths": torch.tensor(pred_length),
                "decoder_target_lengths": torch.tensor(pred_length),
                "groups": data["group"],
                "encoder_time_idx": torch.arange(enc_length),
                "decoder_time_idx": torch.arange(enc_length, enc_length + pred_length),
                "target_scale": target_scale,
                "encoder_mask": encoder_mask,
                "decoder_mask": decoder_mask,
            }
            if data["static"] is not None:
                raw_st_tensor = data.get("static")
                static_col_names = self.data_module.time_series_with_image_metadata["cols"]["st"]

                is_categorical_mask = torch.tensor(
                    [
                        self.data_module.time_series_with_image_metadata["col_type"].get(col_name)
                        == "C"
                        for col_name in static_col_names
                    ],
                    dtype=torch.bool,
                )

                is_continuous_mask = ~is_categorical_mask

                st_cat_values_for_item = raw_st_tensor[is_categorical_mask]
                st_cont_values_for_item = raw_st_tensor[is_continuous_mask]

                if st_cat_values_for_item.shape[0] > 0:
                    x["static_categorical_features"] = st_cat_values_for_item.unsqueeze(
                        0
                    )
                else:
                    x["static_categorical_features"] = torch.zeros(
                        (1, 0), dtype=torch.float32
                    )

                if st_cont_values_for_item.shape[0] > 0:
                    x["static_continuous_features"] = st_cont_values_for_item.unsqueeze(
                        0
                    )
                else:
                    x["static_continuous_features"] = torch.zeros(
                        (1, 0), dtype=torch.float32
                    )

            y = data["target"][decoder_indices]
            if y.ndim == 1:
                y = y.unsqueeze(-1)

            return x, y

    def _create_windows(self, indices: torch.Tensor) -> list[tuple[int, int, int, int]]:
        """Generate sliding windows for training, validation, and testing.

        Returns
        -------
        List[Tuple[int, int, int, int]]
            A list of tuples, where each tuple consists of:
            - ``series_idx`` : int
              Index of the time series in `time_series_dataset`.
            - ``start_idx`` : int
              Start index of the encoder window.
            - ``enc_length`` : int
              Length of the encoder input sequence.
            - ``pred_length`` : int
              Length of the decoder output sequence.
        """
        windows = []

        for idx in indices:
            series_idx = cast(int, idx.item())
            sample = self.train_ds[series_idx]
            sequence_length = len(sample["y"])

            if sequence_length < self.encoder_length + self.prediction_length:
                continue

            effective_min_prediction_idx = (
                self.min_prediction_idx
                if self.min_prediction_idx is not None
                else self.encoder_length
            )

            max_prediction_idx = sequence_length - self.prediction_length + 1

            if max_prediction_idx <= effective_min_prediction_idx:
                continue

            for start_idx in range(
                0, max_prediction_idx - effective_min_prediction_idx
            ):
                if (
                    start_idx + self.encoder_length + self.prediction_length
                    <= sequence_length
                ):
                    windows.append(
                        (
                            series_idx,
                            start_idx,
                            self.encoder_length,
                            self.prediction_length,
                        )
                    )

        return windows

    def setup(self, stage: Optional[str] = None):
        """Prepare the datasets for training, validation, testing, or prediction.

        Parameters
        ----------
        stage : Optional[str], default=None
            Specifies the stage of setup. Can be one of:
            - ``"fit"`` : Prepares training and validation datasets.
            - ``"test"`` : Prepares the test dataset.
            - ``"predict"`` : Prepares the dataset for inference.
            - ``None`` : Prepares ``fit`` datasets.
        """

        if hasattr(self, "test_dataset"):
            total_train_series = len(self.train_ds)
            self._split_indices = torch.randperm(total_train_series)

            self._train_size = int(self.train_val_split[0] * total_train_series)
            self._val_size = int(self.train_val_split[1] * total_train_series)

        elif hasattr(self, "pred_ds"):
            # This is full prediction mode so we use pred_ds to produce output.

            stage == "prediction"


        else:
            total_series = len(self.train_ds)
            self._split_indices = torch.randperm(total_series)

            self._train_size = int(self.train_val_test_split[0] * total_series)
            self._val_size = int(self.train_val_test_split[1] * total_series)

            self._train_indices = self._split_indices[: self._train_size]
            self._val_indices = self._split_indices[
                self._train_size : self._train_size + self._val_size
            ]
            self._test_indices = self._split_indices[self._train_size + self._val_size :]

        if stage is None or stage == "fit":
            if not hasattr(self, "train_dataset") or not hasattr(self, "val_dataset"):
                self.train_windows = self._create_windows(self._train_indices)
                self.val_windows = self._create_windows(self._val_indices)

                self.train_dataset = self._ProcessedDataset(
                    self.train_ds,
                    self,
                    self.train_windows,
                )
                self.val_dataset = self._ProcessedDataset(
                    self.train_ds,
                    self,
                    self.val_windows,
                )

        elif stage == "test":
            if hasattr(self, "test_dataset"):
                self.test_windows = self._create_windows(self.test_indices)
            if not hasattr(self, "test_dataset"):
                self.test_windows = self._create_windows(self._test_indices)
                self.test_dataset = self._ProcessedDataset(
                    self.train_ds,
                    self,
                    self.test_windows,
                )
        elif stage == "predict":
            predict_indices = torch.arange(len(self.))
            self.predict_windows = self._create_windows(predict_indices)
            self.predict_dataset = self._ProcessedDataset(
                self.train_ds,
                self,
                self.predict_windows,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        x_batch = {
            "encoder_cat": torch.stack([x["encoder_cat"] for x, _ in batch]),
            "encoder_cont": torch.stack([x["encoder_cont"] for x, _ in batch]),

            "encoder_img":   torch.stack([x["encoder_img"]  for x,_ in batch]),

            "decoder_cat": torch.stack([x["decoder_cat"] for x, _ in batch]),
            "decoder_cont": torch.stack([x["decoder_cont"] for x, _ in batch]),

            "decoder_img":   torch.stack([x["decoder_img"] for x,_ in batch]),
            
            "encoder_lengths": torch.stack([x["encoder_lengths"] for x, _ in batch]),
            "decoder_lengths": torch.stack([x["decoder_lengths"] for x, _ in batch]),
            "decoder_target_lengths": torch.stack(
                [x["decoder_target_lengths"] for x, _ in batch]
            ),

            "groups": torch.stack([x["groups"] for x, _ in batch]),
            "encoder_time_idx": torch.stack([x["encoder_time_idx"] for x, _ in batch]),
            "decoder_time_idx": torch.stack([x["decoder_time_idx"] for x, _ in batch]),
            "target_scale": torch.stack([x["target_scale"] for x, _ in batch]),
            "encoder_mask": torch.stack([x["encoder_mask"] for x, _ in batch]),
            "decoder_mask": torch.stack([x["decoder_mask"] for x, _ in batch]),
        }

        if "static_categorical_features" in batch[0][0]:
            x_batch["static_categorical_features"] = torch.stack(
                [x["static_categorical_features"] for x, _ in batch]
            )
            x_batch["static_continuous_features"] = torch.stack(
                [x["static_continuous_features"] for x, _ in batch]
            )

        y_batch = torch.stack([y for _, y in batch])
        return x_batch, y_batch
