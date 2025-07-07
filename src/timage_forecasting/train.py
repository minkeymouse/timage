# train_timage_task5.py

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pandas as pd

from timage_forecasting.datamodule import TimeSeriesWithImageDataModule
from timage_forecasting.models import Timage


def main(
    train_csv: str = "experiments/task5/train_processed_img_task5.csv",
    test_csv: str  = "experiments/task5/test_processed_img_task5.csv",
    output_dir: str = "lightning_logs/task5",
    batch_size: int = 16,
    max_encoder_length: int = 48,
    max_prediction_length: int = 12,
    num_encoder_layers_ts: int = 3,
    hidden_size_ts: int = 128,
    num_decoder_layers: int = 4,
    decoder_output_dim: int = 128,
    temporal_width_future: int = 8,
    temporal_hidden_size_future: int = 64,
    image_backbone: str = "tf_efficientnetv2_b0.in1k",
    image_pretrained: bool = True,
    dropout: float = 0.1,
    learning_rate: float = 5e-4,
    max_epochs: int = 40,
    gpus: int = 1,
):
    # --- 1) load and preprocess train & test ---
    sat_cols = [f"sat_v{i}" for i in range(1, 576 + 1)]

    def _load(path):
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[df["panel_id"] == 0].reset_index(drop=True)
        df[sat_cols] = df[sat_cols].fillna(method="ffill").fillna(0.0)
        return df

    df_train = _load(train_csv)
    df_test  = _load(test_csv)

    # --- 2) DataModule for train/val split ---
    dm = TimeSeriesWithImageDataModule(
        dataframe=df_train,
        time_idx="time_idx",
        target="kwh",
        group_ids=["panel_id"],
        static_categoricals=["panel_id"],
        time_varying_known_reals=["solar_term", "hod", "dow", "moy", "time_idx"],
        time_varying_unknown_reals=["kwh"],
        image_cols=sat_cols,
        image_shape=(1, 24, 24),
        batch_size=batch_size,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        num_workers=4,
        train_fraction=0.7,
        val_fraction=0.2,
        shuffle=True,
    )

    # --- 3) instantiate model ---
    model = Timage.from_dataset(
        dm.train_dataloader().dataset,
        input_chunk_length=max_encoder_length,
        output_chunk_length=max_prediction_length,
        num_encoder_layers_ts=num_encoder_layers_ts,
        hidden_size_ts=hidden_size_ts,
        num_decoder_layers=num_decoder_layers,
        decoder_output_dim=decoder_output_dim,
        temporal_width_future=temporal_width_future,
        temporal_hidden_size_future=temporal_hidden_size_future,
        image_backbone=image_backbone,
        image_pretrained=image_pretrained,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    # --- 4) Trainer + callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        gpus=gpus if torch.cuda.is_available() and gpus > 0 else 0,
        max_epochs=max_epochs,
        gradient_clip_val=0.1,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    # --- 5) fit on train+val ---
    trainer.fit(model, dm)

    # --- 6) test on the *new* test CSV ---
    test_dm = TimeSeriesWithImageDataModule(
        dataframe=df_test,
        time_idx="time_idx",
        target="kwh",
        group_ids=["panel_id"],
        static_categoricals=["panel_id"],
        time_varying_known_reals=["solar_term", "hod", "dow", "moy", "time_idx"],
        time_varying_unknown_reals=["kwh"],
        image_cols=sat_cols,
        image_shape=(1, 24, 24),
        batch_size=batch_size,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        num_workers=4,
        train_fraction=0.0,  # no train split here
        val_fraction=0.0,
        test_fraction=1.0,   # everything goes to test
        shuffle=False,
    )

    trainer.test(model, datamodule=test_dm)
    preds = trainer.predict(model, datamodule=test_dm)
    print(f"Done. Best checkpoint: {checkpoint_callback.best_model_path}")
    # preds is a list of batches of (B, H, 1) tensors
    return preds


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train & test Timage on task5 (panel_id=0)")
    parser.add_argument("--train_csv", default="experiments/task5/train_processed_img_task5.csv")
    parser.add_argument("--test_csv",  default="experiments/task5/test_processed_img_task5.csv")
    parser.add_argument("--output_dir", default="lightning_logs/task5")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_encoder_length", type=int, default=48)
    parser.add_argument("--max_prediction_length", type=int, default=12)
    parser.add_argument("--num_encoder_layers_ts", type=int, default=3)
    parser.add_argument("--hidden_size_ts", type=int, default=128)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--decoder_output_dim", type=int, default=128)
    parser.add_argument("--temporal_width_future", type=int, default=8)
    parser.add_argument("--temporal_hidden_size_future", type=int, default=64)
    parser.add_argument("--image_backbone", type=str, default="tf_efficientnetv2_b0.in1k")
    parser.add_argument("--image_pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--gpus", type=int, default=1)

    args = parser.parse_args()
    main(**vars(args))
