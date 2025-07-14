# train_timexr.py
from __future__ import annotations
import torch
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from models.timexr_model import TimeXer
from dataset.dataset import ForecastingDataModule
from timage_forecasting.callbacks import WalkForward
import config as cfg

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
)


def average_checkpoints(ckpt_paths: list[str], out_path: str):
    """Load a list of Lightning .ckpt files, average their state_dicts, and save raw."""
    state_dicts = [
        torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        for p in ckpt_paths
    ]
    avg_state: dict[str, torch.Tensor] = {}
    for k in state_dicts[0].keys():
        stacked = torch.stack([sd[k] for sd in state_dicts], dim=0)
        avg_state[k] = stacked.mean(dim=0)
    # save only the averaged weights
    torch.save({"state_dict": avg_state}, out_path)
    print(f"✅ Averaged {len(ckpt_paths)} checkpoints into {out_path}")


def main() -> None:
    pl.seed_everything(cfg.SEED, workers=True)

    # ── data ─────────────────────────────────────────────────────────────────
    dm = ForecastingDataModule()
    dm.prepare_data()
    dm.setup(stage="fit")

    # ── model ───────────────────────────────────────────────────────────────
    model = TimeXer.from_dataset(dm.full_ds, **cfg.TXR_PARAMS)

    # ── callbacks & logger ──────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints/txr",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )
    logger = CSVLogger(save_dir="logs", name="timexr")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        default_root_dir="logs/txr",
        strategy="ddp",
        logger=logger,
        callbacks=[
            WalkForward(),
            ckpt_cb,
            LearningRateMonitor(),
            EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE),
        ],
        max_epochs=cfg.MAX_EPOCHS,
        log_every_n_steps=cfg.LOG_INTERVAL,
        gradient_clip_val=cfg.GRAD_CLIP_VAL,
        reload_dataloaders_every_n_epochs=1,
    )

    # ── train & validate ────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)

    # ── average best-K + last ckpts ─────────────────────────────────────────
    best_ckpts = list(ckpt_cb.best_k_models.keys())
    last_ckpt = ckpt_cb.last_model_path
    ckpts_to_avg = best_ckpts.copy()
    if last_ckpt and last_ckpt not in ckpts_to_avg:
        ckpts_to_avg.append(last_ckpt)

    averaged_path = Path("checkpoints/txr") / "averaged_timexr.ckpt"
    if len(ckpts_to_avg) > 1:
        average_checkpoints(ckpts_to_avg, str(averaged_path))
    else:
        print("ℹ️ Not enough checkpoints to average (need ≥2)")

    # ── load & test ─────────────────────────────────────────────────────────
    ckpt_to_test = averaged_path if averaged_path.exists() else ckpt_cb.best_model_path
    print(f"🔍 Testing from checkpoint: {ckpt_to_test}")

    # 1) build fresh model architecture
    model = TimeXer.from_dataset(dm.full_ds, **cfg.TXR_PARAMS)

    # 2) load only the averaged weights
    ckpt_data = torch.load(str(ckpt_to_test), map_location="cpu")
    model.load_state_dict(ckpt_data["state_dict"])

    # 3) move to device & eval
    model = model.to(cfg.DEVICE).eval()

    # 4) run test
    dm.setup(stage="test")
    test_metrics = trainer.test(model, datamodule=dm)
    print(f"✅ Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
