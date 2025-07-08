from typing import Any, Tuple, Dict
import torch
from pytorch_forecasting import TimeSeriesDataSet

class TimeSeriesWithImageDataSet(TimeSeriesDataSet):
    """
    Extends TimeSeriesDataSet by carving out a contiguous block of 'real' columns
    as flattened image pixels, reshaping them into (C, H, W) frames for the history window.
    """

    def __init__(
        self,
        *,
        image_cols_start: str,
        image_cols_end: str,
        image_shape: Tuple[int, int, int],
        **ts_kwargs: Any,
    ):
        # Initialize base TS dataset
        super().__init__(**ts_kwargs)

        # stash image info
        self.image_cols_start = image_cols_start
        self.image_cols_end   = image_cols_end
        self.image_shape      = image_shape  # (C, H, W)

        # identify the pixel columns among self.reals
        if image_cols_start not in self.reals or image_cols_end not in self.reals:
            raise ValueError(
                f"image_cols_start/end must be in self.reals; "
                f"got {image_cols_start!r}, {image_cols_end!r} not in {self.reals}"
            )
        start_idx = self.reals.index(image_cols_start)
        end_idx   = self.reals.index(image_cols_end)
        if start_idx > end_idx:
            raise ValueError(f"{image_cols_start!r} must come before {image_cols_end!r}")

        # build and validate the flattened‐image column list
        self.image_cols = self.reals[start_idx : end_idx + 1]
        C, H, W = image_shape
        expected = C * H * W
        if len(self.image_cols) != expected:
            raise ValueError(
                f"Found {len(self.image_cols)} image columns but expected {expected} "
                f"({C}×{H}×{W})"
            )

        # compute masks for slicing contiguously
        self._image_idx = [ self.reals.index(c) for c in self.image_cols ]
        keep_mask = torch.ones(len(self.reals), dtype=torch.bool)
        keep_mask[self._image_idx] = False
        self._keep_mask = keep_mask

        # pull target metadata for downstream slicing
        params = self.get_parameters()
        self.target_positions = params["target_positions"]

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Any]:
        x, (y, w) = super().__getitem__(idx)

        # concatenate encoder+decoder continuous features
        enc = x["encoder_cont"]    # (L, D)
        dec = x["decoder_cont"]    # (H, D)
        all_cont = torch.cat([enc, dec], dim=0)  # (L+H, D)

        # split off image pixels and the rest
        flat_pixels = all_cont[:, self._image_idx]      # (L+H, N)
        cont_no_img = all_cont[:, self._keep_mask]      # (L+H, D-N)

        # restore encoder/decoder splits
        L = enc.size(0)
        x["encoder_cont"] = cont_no_img[:L]
        x["decoder_cont"] = cont_no_img[L:]

        # only take history-window frames for images
        pixel_hist = flat_pixels[:L]                    # (L, N)
        C, H, W = self.image_shape
        x["x_image"] = pixel_hist.view(L, C, H, W)      # (L, C, H, W)

        return x, (y, w)

    @staticmethod
    def _collate_fn(batches: Any) -> Any:
        # standard TSDataSet batching
        batch_x, (batch_y, batch_w) = TimeSeriesDataSet._collate_fn(batches)
        # stack the image tensors
        imgs = torch.stack([ b[0]["x_image"] for b in batches ], dim=0)
        batch_x["x_image"] = imgs  # (B, L, C, H, W)
        return batch_x, (batch_y, batch_w)

    def get_parameters(self) -> Dict[str, Any]:
        # ensure TS-DataSet parameters bubble up
        return super().get_parameters()
