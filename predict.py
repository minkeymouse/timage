#!/usr/bin/env python3
"""
predict.py – inference and (optional) online fine-tuning.
• Each model fine-tunes in its own Trainer → no ModelCheckpoint clash
• Warm-starts from the base checkpoints
• Saves weights-only ⇒ compatible with torch ≥ 2.6 (no pickling error)
"""
from __future__ import annotations
import argparse, warnings, pickle
from pathlib import Path
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (ModelCheckpoint,
                                         EarlyStopping,
                                         LearningRateMonitor)
from lightning.pytorch.loggers import CSVLogger
from pytorch_forecasting import TemporalFusionTransformer

from models.timexr_model import TimeXer
from dataset.dataset import PredictionDataModule
import config as cfg

warnings.filterwarnings("ignore", message="X does not have valid feature names")
TARGET_COLS = ["humid_obs", "degC_obs", "mmHg_obs"]
BASE_CKPT = {
    "tft": "base_ckpts/tft_base.ckpt",
    "txr": "base_ckpts/txr_base.ckpt",
}

# ───────────────── helper ──────────────────────────────────────────────────
def to_df(batch_out, targets) -> pd.DataFrame:
    """Convert PF predict() output → DataFrame(horizon × targets)."""
    if isinstance(batch_out, list):
        batch_out = torch.cat(
            [b["prediction"] if isinstance(b, dict) else b for b in batch_out], dim=0
        )
    if isinstance(batch_out, torch.Tensor):
        batch_out = batch_out.detach().cpu()
        if batch_out.ndim == 3:                       # (batch, horizon, targets)
            batch_out = batch_out.reshape(-1, batch_out.shape[-1])
        if batch_out.shape[1] != len(targets):        # (targets, horizon) → transpose
            batch_out = batch_out.T
    return pd.DataFrame(batch_out.numpy(), columns=targets)

def parse_args():
    p = argparse.ArgumentParser("Fine-tune (optional) & predict – no metrics")
    p.add_argument("--train",  action="store_true",
                   help="online fine-tune both models before predicting")
    p.add_argument("--epochs", type=int, default=10,
                   help="#epochs for fine-tune")
    return p.parse_args()

# ───────────────── main ────────────────────────────────────────────────────
def main(online_train=False, epochs=10):
    pl.seed_everything(cfg.SEED, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Data
    dm = PredictionDataModule(); dm.prepare_data()
    dm.full_ds = pickle.load(open("artifacts/full_ds.pkl", "rb"))
    dm.setup(stage="predict")
    if online_train:
        dm.setup(stage="fit")

    # 2) Model instances
    tft = TemporalFusionTransformer.from_dataset(dm.full_ds, **cfg.TFT_PARAMS)
    txr = TimeXer.from_dataset(dm.full_ds, **cfg.TXR_PARAMS)

    # 3) Optional fine-tune (one Trainer per model)
    if online_train:
        logdir = Path("logs/predict"); logdir.mkdir(parents=True, exist_ok=True)

        def fine_tune(model, name):
            # ---- warm-start from base checkpoint ----
            model.load_state_dict(
                torch.load(BASE_CKPT[name], map_location="cpu", weights_only=False)["state_dict"],
                strict=False
            )
            # ---- checkpoint callback (weights-only) ----
            ckpt = ModelCheckpoint(
                dirpath=logdir/f"ckpts/{name}",
                filename=f"{name}-" + "{epoch:02d}-{val_loss:.4f}",
                monitor="val_loss", mode="min",
                save_top_k=1, save_weights_only=True
            )
            trainer = Trainer(
                accelerator=device, devices=1, max_epochs=epochs, precision=32,
                callbacks=[ckpt, LearningRateMonitor(),
                           EarlyStopping("val_loss", patience=cfg.PATIENCE)],
                logger=CSVLogger(logdir, name=name)
            )
            trainer.fit(model, datamodule=dm)
            return ckpt.best_model_path

        path_tft = fine_tune(tft, "tft")
        path_txr = fine_tune(txr, "txr")
    else:
        path_tft = BASE_CKPT["tft"]
        path_txr = BASE_CKPT["txr"]

    # 4) Load best weights (weights-only files → state_dict)
    for mdl, path in ((tft, path_tft), (txr, path_txr)):
        mdl.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=False)
        mdl.to(device).eval()

    # 5) Predict
    loader  = dm.predict_dataloader()
    df_tft  = to_df(tft.predict(loader, mode="prediction"), TARGET_COLS)
    df_txr  = to_df(txr.predict(loader, mode="prediction"), TARGET_COLS)
    df_ens  = 0.5 * df_tft + 0.5 * df_txr

    # 6) Append hourly date column
    base_df = pd.read_csv("input/base_data.csv", parse_dates=["date"])
    # rows where all targets are finite (i.e. last observation)
    obs_mask = ~base_df[TARGET_COLS].isna().any(axis=1)
    last_date = base_df.loc[obs_mask, "date"].max()

    horizon   = len(df_ens)
    df_ens.insert(
        0, "date",
        pd.date_range(last_date + pd.Timedelta(hours=1), periods=horizon, freq="h")
    )

    # 7) Save
    out_dir = Path("output"); out_dir.mkdir(exist_ok=True)
    df_ens.to_csv(out_dir/"prediction.csv", index=False)
    print(f"✅ prediction.csv saved to {out_dir}")

# entry-point
if __name__ == "__main__":
    args = parse_args()
    main(online_train=args.train, epochs=args.epochs)
