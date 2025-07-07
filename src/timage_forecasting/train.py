# scripts/train.py
import os
import pathlib
from datetime import datetime
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from timage_forecasting.datamodule import TimeSeriesWithImageDataModule
from timage_forecasting.model import Timage


def get_arg_parser() -> argparse.ArgumentParser:
    """Minimal CLI around the YAML config."""
    p = argparse.ArgumentParser(description="Train Timage forecasting model")
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML/OMEGACONF config file",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (optional)",
    )
    return p


def init_callbacks(cfg):
    """Lightning callbacks."""
    ckpt_cb = ModelCheckpoint(
        monitor=cfg.train.monitor,
        dirpath=pathlib.Path(cfg.train.save_dir) / "checkpoints",
        filename=f"{cfg.exp_name}" + "-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
        auto_insert_metric_name=True,
    )
    es_cb = EarlyStopping(
        monitor=cfg.train.monitor,
        mode="min",
        patience=cfg.train.early_stop_patience,
        verbose=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    return [ckpt_cb, es_cb, lr_cb]


def main():
    # ---- CLI & Config ----------------------------------------------------------------
    parser = get_arg_parser()
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)  # merge CL-args if you like: OmegaConf.merge(...)
    pl.seed_everything(cfg.train.seed, workers=True)

    # ---- Loggers ---------------------------------------------------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_logger = TensorBoardLogger(
        save_dir=cfg.train.save_dir,
        name=cfg.exp_name,
        version=run_id,
    )

    # ---- Data ------------------------------------------------------------------------
    datamodule = TimeSeriesWithImageDataModule(
        data_csv=cfg.data.csv_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        input_chunk_length=cfg.data.input_chunk_length,
        output_chunk_length=cfg.data.output_chunk_length,
        # … any other constructor kwargs you exposed …
    )

    # ---- Model -----------------------------------------------------------------------
    model = Timage(
        in_img_channels=cfg.model.in_img_channels,
        img_backbone=cfg.model.backbone,
        ts_hidden_size=cfg.model.ts_hidden_size,
        fusion_dim=cfg.model.fusion_dim,
        decoder_hidden=cfg.model.decoder_hidden,
        learning_rate=cfg.train.lr,
        # … etc …
    )

    # ---- Callbacks -------------------------------------------------------------------
    cbs = init_callbacks(cfg)

    # ---- Trainer ---------------------------------------------------------------------
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,          # "16-mixed" for AMP
        gradient_clip_val=cfg.train.grad_clip,  # None or float
        accumulate_grad_batches=cfg.train.accum_grad,
        max_epochs=cfg.train.max_epochs,
        deterministic=True,
        callbacks=cbs,
        logger=tb_logger,
        log_every_n_steps=cfg.train.log_every_n,
        enable_progress_bar=True,
        resume_from_checkpoint=args.resume,
    )

    # ---- Optional LR finder / tuner --------------------------------------------------
    if cfg.train.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
        new_lr = lr_finder.suggestion()
        print(f"[AutoLR] setting LR to {new_lr}")
        model.hparams.learning_rate = new_lr

    # ---- Fit / Validate / Test -------------------------------------------------------
    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule, ckpt_path="best")
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
