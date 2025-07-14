# train_tide.py
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.models.tide import TiDEModel  # explicit import path

import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted",
    category=UserWarning,
    module="sklearn"
)

# --------------------------------------------------------------------------- #
# Hydra entrypoint
# --------------------------------------------------------------------------- #
@hydra.main(config_path="../../config", config_name="solar_tide", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # 1) ------------------------------------------------------------------ reproducibility
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # 2) ------------------------------------------------------------------ load raw data
    df      = pd.read_csv(cfg.experiment.df_path,  parse_dates=["date"])
    test_df = pd.read_csv(cfg.experiment.test_df_path, parse_dates=["date"])

    # 3) ------------------------------------------------------------------ categorical dtypes
    cat_cols = ["panel_id", "solar_term", "hod", "dow", "moy"]
    for d in (df, test_df):
        d[cat_cols] = d[cat_cols].astype(str).astype("category")      # keeps codes stable

    # 4) ------------------------------------------------------------------ train / val split
    cutoff_idx = int(df["time_idx"].max() * (1 - cfg.experiment.cutoff))
    train_df   = df[df["time_idx"] <= cutoff_idx].copy()
    val_df     = df[df["time_idx"]  > cutoff_idx].copy()

    # 5) ------------------------------------------------------------------ TimeSeriesDataSets
    #    Always build val/test from *train* to share encoders & scalers
    train_ds = hydra.utils.instantiate(cfg.dataset, data=train_df)
    train_ds.save("train_dataset.pkl")  
    val_ds   = TimeSeriesDataSet.from_dataset(
        train_ds, val_df, stop_randomization=True, predict=True
    )
    test_ds  = TimeSeriesDataSet.from_dataset(
        train_ds, test_df, predict=True
    )

    # 6) ------------------------------------------------------------------ DataLoaders
    dl_kwargs = dict(
        batch_size = cfg.experiment.batch_size,
        num_workers= cfg.experiment.num_workers,
    )
    train_loader = train_ds.to_dataloader(train=True,  **dl_kwargs)
    val_loader   = val_ds.to_dataloader  (train=False, **dl_kwargs)
    test_loader  = test_ds.to_dataloader (train=False, **dl_kwargs)

    # 7) ------------------------------------------------------------------ model & trainer
    model   = TiDEModel.from_dataset(train_ds, **cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    # 8) ------------------------------------------------------------------ raw predict & CSV
    horizon = train_ds.max_prediction_length
    raw, x  = model.predict(test_ds, mode="raw", return_x=True)

    preds      = raw["prediction"].detach().cpu().numpy().flatten()
    time_idxs  = x["decoder_time_idx"].cpu().numpy().flatten()

    # decode panel_id from encoded group tensor → original labels
    encoded_panels = x["groups"][:, 0]        # (N,)
    panel_labels   = train_ds.transform_values(
        name="panel_id",
        values=encoded_panels.cpu(),
        inverse=True,
        group_id=True,
    )
    panel_col = panel_labels.repeat(horizon)

    out = pd.DataFrame(
        {
            "panel_id": panel_col,
            "time_idx": time_idxs,
            "kwh_pred": preds,
        }
    )

    Path("outputs").mkdir(exist_ok=True)
    csv_path = Path("outputs") / "solar_predictions.csv"
    out.to_csv(csv_path, index=False)
    print(f"✅ Wrote {len(out):,} rows to {csv_path}")

    # 9) ------------------------------------------------------------------ (optional) log config
    print("\nHydra config used:")
    print(OmegaConf.to_yaml(cfg, resolve=True))


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
