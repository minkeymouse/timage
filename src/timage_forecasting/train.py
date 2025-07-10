# src/timage_forecasting/train.py

import pandas as pd
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from timage_forecasting.dataset import TimeSeriesWithImageDataSet
from timage_forecasting.model import Timage

# src/timage_forecasting/train.py
@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.experiment.seed, workers=True)

    df      = pd.read_csv(cfg.experiment.df_path,      parse_dates=["date"])
    test_df = pd.read_csv(cfg.experiment.test_df_path, parse_dates=["date"])
    cutoff  = int(df["time_idx"].max() * (1 - cfg.experiment.cutoff))
    train_df = df[df["time_idx"] <= cutoff]
    val_df   = df[df["time_idx"] >  cutoff]

    # instantiate datasets
    train_ds = hydra.utils.instantiate(cfg.dataset, data=train_df)
    val_ds   = hydra.utils.instantiate(cfg.dataset, data=val_df)
    test_ds  = hydra.utils.instantiate(cfg.dataset, data=test_df)

    # make dataloaders
    collate = TimeSeriesWithImageDataSet._collate_fn
    train_loader = train_ds.to_dataloader(train=True,
                                          batch_size=cfg.experiment.batch_size,
                                          num_workers=cfg.experiment.num_workers,
                                          collate_fn=collate)
    val_loader   = val_ds  .to_dataloader(train=False,
                                          batch_size=cfg.experiment.batch_size,
                                          num_workers=cfg.experiment.num_workers,
                                          collate_fn=collate)
    test_loader  = test_ds .to_dataloader(train=False,
                                          batch_size=cfg.experiment.batch_size,
                                          num_workers=cfg.experiment.num_workers,
                                          collate_fn=collate)

    # now build the model from the dataset
    model = Timage.from_dataset(
        train_ds,
        **cfg.model  # only your model‐hyperparameters (no image bits!),
    )

    # trainer / fit / test
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model,  test_loader)

if __name__ == "__main__":
    main()

