from typing import Optional, Union, cast, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pytorch_forecasting.utils._coerce import _coerce_to_list

class TimeSeriesWithImage(Dataset):
    """PyTorch Dataset for time series data stored in pandas DataFrame.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        data_future: Optional[pd.DataFrame] = None,
        time: Optional[str] = None,
        target: Optional[Union[str, list[str]]] = None,
        group: Optional[list[str]] = None,
        weight: Optional[str] = None,
        num: Optional[list[Union[str, list[str]]]] = None,
        cat: Optional[list[Union[str, list[str]]]] = None,
        known: Optional[list[Union[str, list[str]]]] = None,
        unknown: Optional[list[Union[str, list[str]]]] = None,
        static: Optional[list[Union[str, list[str]]]] = None,

        image_cols_start: Optional[str] = None,
        image_cols_end: Optional[str] = None,
        image_shape: Optional[Tuple[int, int, int]] = None,
        future_image: Optional[bool] = False,
    ):
        
        if time is None:
            raise ValueError("Require Time Index column")
        if target is None:
            raise ValueError("Require Target column")
        
        if image_cols_start is None or image_cols_end is None or image_shape is None:
            raise ValueError(
                "TimeSeriesWithImage requires image_cols_start, image_cols_end, "
                "and image_shape to be provided and non-None"
            )
    
        self.data = data
        self.data_future = data_future
        self.time = time
        self.target = target
        self.group = group
        self.weight = weight
        self.num = num
        self.cat = cat
        self.known = known
        self.unknown = unknown
        self.static = static
        self.image_cols_start = image_cols_start
        self.image_cols_end   = image_cols_end
        self.image_shape      = image_shape
        self.future_image = future_image

        start = cast(int, data.columns.get_loc(image_cols_start))
        end   = cast(int, data.columns.get_loc(image_cols_end))
        image_cols = data.columns.to_list()[start : end + 1]

        super().__init__()

        # handle defaults, coercion, and derived attributes
        self._target = _coerce_to_list(target)
        self._group = _coerce_to_list(group)
        self._num = _coerce_to_list(num)
        self._cat = _coerce_to_list(cat)
        self._known = _coerce_to_list(known)
        self._unknown = _coerce_to_list(unknown)
        self._static = _coerce_to_list(static)
        self._img = _coerce_to_list(image_cols)

        self.feature_cols = [
            col
            for col in data.columns
            if col
            not in [self.time]
            + self._group
            + [self.weight]
            + self._target
            + self._img
            + self._static
        ]
        
        if self._group:
            self._groups = self.data.groupby(self._group).groups
            self._group_ids = list(self._groups.keys())
        else:
            self._groups = {"_single_group": self.data.index}
            self._group_ids = ["_single_group"]

        if self.data_future is not None and self._group:
            self._future_groups = self.data_future.groupby(self._group).groups

        self.known_ts_feats   = [c for c in self.feature_cols if c in self._known]
        self.unknown_ts_feats = [c for c in self.feature_cols if c not in self._known]
        self.group_to_idx = {g: i for i, g in enumerate(self._group_ids)}

        self.unknown_img = self._img if not self.future_image else []
        self.known_img   = self._img if     self.future_image else []

        self._prepare_metadata()

        self.group = self._group
        self.target = self._target
        self.num = self._num
        self.cat = self._cat
        self.known_ts = self._known
        self.unknown_ts = self._unknown
        self.static = self._static

    def _prepare_metadata(self):
        """Prepare metadata for the dataset.
        """
        self.metadata = {
            "cols": {
                "y": self._target,
                "x": self.feature_cols,
                "img": self._img,
                "ts_known": self.known_ts_feats,
                "ts_unknown": self.unknown_ts_feats,
                "st": self._static,
                "img_known": self.known_img,
                "img_unknown": self.unknown_img,
            },
            "col_type": {},
            "col_known": {},
            "img_size": self.image_shape
        }

        for col in self.feature_cols + self._target + self._static + self._img:
            if col in self._img:
                t = "IMG"
            elif col in self._cat:
                t = "C"
            else:
                t = "F"
            self.metadata["col_type"][col] = t

        # 2) col_known: exactly which columns your decoder will see
        #    - targets are historical only ("U")
        #    - known_feats are known at forecast time ("K")
        #    - unknown_feats are historical only ("U")
        #    - static always known ("K")
        #    - images known only if future_image=True
        for col in self._target:
            self.metadata["col_known"][col] = "U"
        for col in self.known_ts_feats:
            self.metadata["col_known"][col] = "K"
        for col in self.unknown_ts_feats:
            self.metadata["col_known"][col] = "U"
        for col in self._static:
            self.metadata["col_known"][col] = "K"
        for col in self.known_img:
            self.metadata["col_known"][col] = "K"
        for col in self.unknown_img:
            self.metadata["col_known"][col] = "U"
                
        self.metadata["cols"]["mask"] = ["valid_mask"]

    def __len__(self) -> int:
        """Return number of time series in the dataset."""
        return len(self._group_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Fetch one series (past + optional future), with image frames."""
        group_key = self._group_ids[index]
        group_idx = self.group_to_idx[group_key]

        if self._group:
            mask = self._groups[group_key]
            series = self.data.loc[mask]
        else:
            series = self.data

        t_past = series[self.time].values
        y_past = series[self._target].values
        if y_past.ndim == 1:
            y_past = y_past.reshape(-1, 1)
        x_past = series[self.feature_cols].values
        img_past = series[self._img].values.astype(float)

        # store cutoff
        cutoff = t_past.max()

        # now, if there's future data, merge it
        if self.data_future is not None:
            if self._group:
                fut_mask = self._future_groups[group_key]
                fut = self.data_future.loc[fut_mask]
            else:
                fut = self.data_future

            t_fut = fut[self.time].values
            # build full timeline
            times = np.unique(np.concatenate([t_past, t_fut]))
            times.sort()
            N = len(times)

            # allocate arrays (+ mask for “is real?”)
            x_all   = np.full((N, len(self.feature_cols)), np.nan, dtype=float)
            y_all   = np.full((N, len(self._target)),      np.nan, dtype=float)
            img_all = np.full((N, *self.image_shape),      np.nan, dtype=float)
            valid   = np.zeros(N, dtype=bool)

            idx_map = {t: i for i, t in enumerate(times)}
            # fill past
            for i, t in enumerate(t_past):
                j = idx_map[t]
                x_all[j]   = x_past[i]
                y_all[j]   = y_past[i]
                img_all[j] = img_past[i].reshape(self.image_shape)
                valid[j]   = True

            # fill known future features (and images if requested)
            for i, t in enumerate(t_fut):
                j = idx_map[t]
                for col in self._known:
                    if col in self.feature_cols:
                        k = self.feature_cols.index(col)
                        x_all[j, k] = fut[col].iat[i]
                if self.future_image:
                    img_all[j] = fut[self._img].values[i].reshape(self.image_shape)
                # leave valid[j] = False, so model knows these are “to predict”

            # overwrite past arrays
            t_arr   = times
            x_arr   = x_all
            y_arr   = y_all
            img_arr = img_all

        else:
            # no future—just past
            t_arr   = t_past
            x_arr   = x_past
            y_arr   = y_past
            img_arr = img_past.reshape(len(t_past), *self.image_shape)
            valid   = np.ones(len(t_past), dtype=bool)

        t_tensor = torch.tensor(t_arr, dtype=torch.long)

        if self._static:
            st_vals = series[self._static].iloc[0].values.astype(float)
            st_tensor = torch.tensor(st_vals, dtype=torch.float32)
        else:
            # make an empty float32 tensor
            st_tensor = torch.empty(0, dtype=torch.float32)

        # build output dict
        out = {
            "t":             t_tensor,
            "y":             torch.tensor(y_arr, dtype=torch.float32),
            "x":             torch.tensor(x_arr, dtype=torch.float32),
            "img":           torch.tensor(img_arr, dtype=torch.float32),
            "group":         torch.tensor([group_idx], dtype=torch.long),
            "st":            st_tensor,
            "cutoff_time":   torch.tensor(cutoff, dtype=torch.float32),
            "valid_mask":    torch.tensor(valid, dtype=torch.bool),
        }

        # weights if present
        if self.weight:
            w_arr = (fut[self.weight].values if self.data_future is not None else series[self.weight].values)
            out["weights"] = torch.tensor(w_arr, dtype=torch.float32)

        return out


    def get_metadata(self) -> dict:
        """Return metadata about the dataset.

        Returns
        -------
        Dict
            Dictionary containing:
            - cols: column names for y, x, static features, and images
            - col_type: mapping of columns to their types (F/C)
            - col_known: mapping of columns to their future known status (K/U)
        """
        return self.metadata
