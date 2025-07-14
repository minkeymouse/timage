# train_tft.py  (TFT walk-forward + test + checkpoint averaging)
from __future__ import annotations
import warnings
from pathlib import Path

import torch                            # ← needed for torch.load & stack
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from pytorch_forecasting.models import TemporalFusionTransformer
from dataset.dataset import ForecastingDataModule
from timage_forecasting.callbacks import WalkForward
import config as cfg

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names",
)


def average_checkpoints(ckpt_paths: list[str], out_path: str):
    """Load a list of Lightning ckpts, average their state_dicts, and save minimal."""
    state_dicts = [
        torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        for p in ckpt_paths
    ]
    avg_state: dict[str, torch.Tensor] = {}
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key] for sd in state_dicts], dim=0)
        avg_state[key] = stacked.mean(dim=0)
    torch.save({"state_dict": avg_state}, out_path)
    print(f"✅ Averaged {len(ckpt_paths)} checkpoints into {out_path}")


def main() -> None:
    pl.seed_everything(cfg.SEED, workers=True)

    # ── data ─────────────────────────────────────────────────────────────────
    dm = ForecastingDataModule()
    dm.prepare_data()
    dm.setup(stage="fit")

    # ── model ───────────────────────────────────────────────────────────────
    model = TemporalFusionTransformer.from_dataset(
        dm.full_ds,        # give TFT the full dataset so it sees known covariates
        **cfg.TFT_PARAMS,
    )

    # ── callbacks & logger ──────────────────────────────────────────────────
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints/tft",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=5,
        save_last=True,
    )
    logger = CSVLogger(save_dir="logs", name="tft")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,
        default_root_dir="logs/tft",
        max_epochs=cfg.MAX_EPOCHS,
        log_every_n_steps=cfg.LOG_INTERVAL,
        gradient_clip_val=cfg.GRAD_CLIP_VAL,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[
            WalkForward(),
            ckpt_cb,
            LearningRateMonitor(),
            EarlyStopping(monitor="val_loss", patience=cfg.PATIENCE),
        ],
        logger=logger,
    )

    # ── train & validate ────────────────────────────────────────────────────
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)

    # ── average top-k + last checkpoints ────────────────────────────────────
    best_ckpts = list(ckpt_cb.best_k_models.keys())
    last_ckpt = ckpt_cb.last_model_path
    to_avg = best_ckpts.copy()
    if last_ckpt and last_ckpt not in to_avg:
        to_avg.append(last_ckpt)

    avg_path = Path("checkpoints/tft") / "averaged_tft.ckpt"
    if len(to_avg) > 1:
        average_checkpoints(to_avg, str(avg_path))
    else:
        print("ℹ️ Not enough checkpoints to average (need ≥2)")

    # ── manual load & test ──────────────────────────────────────────────────
    ckpt_to_test = avg_path if avg_path.exists() else ckpt_cb.best_model_path
    print(f"🔍 Testing from checkpoint: {ckpt_to_test}")

    model = TemporalFusionTransformer.from_dataset(dm.full_ds, **cfg.TFT_PARAMS)

    ckpt_data = torch.load(str(ckpt_to_test), map_location="cpu")
    model.load_state_dict(ckpt_data["state_dict"])

    model = model.to(cfg.DEVICE).eval()

    dm.setup(stage="test")
    test_metrics = trainer.test(model, datamodule=dm)
    print(f"✅ Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
