#!/usr/bin/env python
import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pandas import read_csv

# ensure src/ is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from timage_forecasting.datamodule import TimeSeriesWithImageDataModule
from timage_forecasting.model import Timage

def main():
    data_dir   = "experiments/task5"
    train_path = os.path.join(data_dir, "train_processed_img_task5.csv")
    test_path  = os.path.join(data_dir, "test_processed_img_task5.csv")

    # 1) Load & filter to panel_id == 0
    df      = read_csv(train_path, parse_dates=["date"])
    test_df = read_csv(test_path,  parse_dates=["date"])
    df      = df[df.panel_id == 0].reset_index(drop=True)
    test_df = test_df[test_df.panel_id == 0].reset_index(drop=True)

    # 2) Cast known categoricals to strings
    cats = ["solar_term", "hod", "dow", "moy"]
    for col in cats:
        df[col]      = df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    # 3) Define image columns & fill NaNs everywhere
    image_cols = [f"sat_v{i}" for i in range(1, 577)]
    # — target
    df["kwh"]      = df["kwh"].fillna(0.0)
    test_df["kwh"] = test_df["kwh"].fillna(0.0)
    # — image bands
    df[image_cols]      = df[image_cols].fillna(0.0)
    test_df[image_cols] = test_df[image_cols].fillna(0.0)

    # 4) Shared TimeSeriesDataSet settings
    ts_kwargs = dict(
        time_idx="time_idx",
        target="kwh",
        group_ids=["panel_id"],
        static_categoricals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=cats,
        time_varying_unknown_reals=image_cols,
        max_encoder_length=24,
        max_prediction_length=24,
        allow_missing_timesteps=True,
    )

    # 5) Build the LightningDataModule
    dm = TimeSeriesWithImageDataModule(
        df=df,
        test_df=test_df,
        image_cols_start="sat_v1",
        image_cols_end="sat_v576",
        image_shape=(1, 24, 24),
        batch_size=64,
        num_workers=4,
        val_split=0.2,
        **ts_kwargs,
    )

    # 6) Fit datamodule to get train_ds & its fitted scalers/params
    dm.setup(stage="fit")
    train_ds  = dm.train_ds
    ds_params = train_ds.get_parameters()    # all TimeSeriesDataSet args
    scaler    = train_ds.target_normalizer   # fitted target scaler

    # 7) Instantiate model DIRECTLY
    model = Timage(
        # architecture params
        input_chunk_length=24,
        output_chunk_length=24,
        num_encoder_layers_ts=2,
        hidden_size_ts=64,
        num_decoder_layers=2,
        decoder_output_dim=32,
        image_backbone="tf_efficientnetv2_b0.in1k",
        image_pretrained=True,
        # dataset / scaling
        dataset_parameters=ds_params,
        output_transformer=scaler,
    )

    # 8) Trainer + checkpoint callback
    ckpt    = ModelCheckpoint(monitor="val_loss", save_top_k=1)
    devices = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=devices,
        callbacks=[ckpt],
    )

    # 9) Run training
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
