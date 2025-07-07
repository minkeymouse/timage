from typing import Any, Tuple, Dict, List, Optional, Union
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    GroupNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)

NORMALIZER = Union[TorchNormalizer, NaNLabelEncoder, EncoderNormalizer]

class TimeSeriesWithImageDataSet(TimeSeriesDataSet):
    def __init__(
        self,
        *args,
        image_cols_start: str,
        image_cols_end: str,
        image_shape: Tuple[int, int, int],
        **kwargs,
    ):
        # pass all the usual TS args via *args, **kwargs, plus these two
        super().__init__(*args, **kwargs)

        # 1) locate the start/end positions in the self.reals list
        if image_cols_start not in self.reals or image_cols_end not in self.reals:
            raise ValueError(f"image_cols_start/end must be real variable names, got "
                             f"{image_cols_start!r} or {image_cols_end!r} not in {self.reals}")
        start_idx = self.reals.index(image_cols_start)
        end_idx   = self.reals.index(image_cols_end)
        if start_idx > end_idx:
            raise ValueError(f"{image_cols_start!r} comes after {image_cols_end!r} in self.reals")

        # 2) build the flattened‐image column list
        image_cols = self.reals[start_idx : end_idx + 1]
        assert len(image_cols) == image_shape[0] * image_shape[1] * image_shape[2], (
            f"{len(image_cols)} pixels ≠ expected {image_shape[0]}×{image_shape[1]}×{image_shape[2]}"
        )
        self.image_cols = image_cols

        assert len(image_cols) == image_shape[0] * image_shape[1] * image_shape[2], (
            f"{len(image_cols)} != {image_shape}"
        )
        self.image_cols  = image_cols
        self.image_shape = image_shape  # (C, H, W)
        # find where those cols live in the continuous vector
        self._image_idx = [self.reals.index(c) for c in image_cols]
        # build mask to drop them from x_cont
        keep = torch.ones(len(self.reals), dtype=torch.bool)
        keep[self._image_idx] = False
        self._keep_mask = keep

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], Any]:
        x, (y, w) = super().__getitem__(idx)
        cont = x["x_cont"]                      # (seq_len, D)
        flat = cont[:, self._image_idx]         # (seq_len, N)
        x["x_cont"] = cont[:, self._keep_mask]  # (seq_len, D-N)
        C, H, W = self.image_shape
        L = flat.size(0)
        x["x_image"] = flat.view(L, C, H, W)    # (seq_len, C, H, W)
        return x, (y, w)

    @staticmethod
    def _collate_fn(batches: Any) -> Any:
        batch_x, (batch_y, batch_w) = TimeSeriesDataSet._collate_fn(batches)
        imgs = torch.stack([b[0]["x_image"] for b in batches], dim=0)
        # imgs: (B, seq_len, C, H, W)
        batch_x["x_image"] = imgs
        return batch_x, (batch_y, batch_w)


class TimeSeriesWithImageDataModule(LightningDataModule):
    """
    DataModule for TimeSeriesWithImageDataSet.
    
    - You pass **all** of the usual TimeSeriesDataSet args into ts_kwargs (including data=your_df)
    - Plus: image_cols, image_shape, a separate test_df, and your splits/batch settings.
    """
    def __init__(
        self,
        *,
        # image-specific:
        image_cols_start: str,
        image_cols_end:   str,
        image_shape: Tuple[int, int, int],
        # full & test DataFrames:
        df: pd.DataFrame,
        test_df: pd.DataFrame,
        # split + loader settings:
        val_split: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 4,
        # all the standard TS-DataSet kwargs:
        **ts_kwargs: Any,
    ):
        super().__init__()
        self.df           = df
        self.test_df      = test_df
        self.image_cols_start = image_cols_start
        self.image_cols_end   = image_cols_end
        self.image_shape  = image_shape
        self.val_split    = val_split
        self.batch_size   = batch_size
        self.num_workers  = num_workers
        self.ts_kwargs    = ts_kwargs

    def setup(self, stage: Optional[str] = None):
        # ---- fit: carve train / val out of self.df ----
        if stage in (None, "fit"):
            # simple random split of the rows (you could also do time-based)
            frac_train = 1.0 - self.val_split
            df_train = self.df.sample(frac=frac_train, random_state=0)
            df_val   = self.df.drop(df_train.index)

            base = {**self.ts_kwargs}
            # override the `data=` for each
            base["data"] = df_train
            self.train_ds = TimeSeriesWithImageDataSet(
                **base,
                image_cols_start=self.image_cols_start,
                image_cols_end=self.image_cols_end,
                image_shape=self.image_shape
            )
            base["data"] = df_val
            self.val_ds   = TimeSeriesWithImageDataSet(
                **base,
                image_cols_start=self.image_cols_start,
                image_cols_end=self.image_cols_end,
                image_shape=self.image_shape
            )

        # ---- test: always build from test_df ----
        if stage in (None, "test"):
            base = {**self.ts_kwargs, "data": self.test_df}
            self.test_ds = TimeSeriesWithImageDataSet(
                **base,
                image_cols_start=self.image_cols_start,
                image_cols_end=self.image_cols_end,
                image_shape=self.image_shape
            )

    def train_dataloader(self):
        return self.train_ds.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return self.val_ds.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        if not hasattr(self, "test_ds"):
            return None
        return self.test_ds.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=self.num_workers
        )
