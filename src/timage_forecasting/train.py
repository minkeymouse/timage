# src/timage_forecasting/train.py
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from timage_forecasting.dataset import TimeSeriesWithImageDataSet
from timage_forecasting.model import Timage

@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    # reproducibility
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # 1) load & split your CSVs
    df       = pd.read_csv(cfg.dataset.df_path,       parse_dates=["date"])
    test_df  = pd.read_csv(cfg.dataset.test_df_path,  parse_dates=["date"])
    cutoff   = int(df["timd_idx"].max() * (1.0 - cfg.experiment.cutoff))
    train_df = df[df["timd_idx"] <= cutoff]
    val_df   = df[df["timd_idx"] >  cutoff]

    # 2) instantiate your TimeSeriesWithImageDataSet via Hydra
    train_ds = hydra.utils.instantiate(cfg.dataset, df=train_df)
    val_ds   = hydra.utils.instantiate(cfg.dataset, df=val_df)
    test_ds  = hydra.utils.instantiate(cfg.dataset, df=test_df)

    # 3) wrap them in DataLoaders using to_dataloader()
    train_loader = train_ds.to_dataloader(
        train=True,
        batch_size   = cfg.experiment.batch_size,
        shuffle      = True,
        num_workers  = cfg.experiment.num_workers,
        collate_fn   = train_ds._collate_fn,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size   = cfg.experiment.batch_size,
        shuffle      = False,
        num_workers  = cfg.experiment.num_workers,
        collate_fn   = val_ds._collate_fn,
    )
    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size   = cfg.experiment.batch_size,
        shuffle      = False,
        num_workers  = cfg.experiment.num_workers,
        collate_fn   = test_ds._collate_fn,
    )

    # 4) instantiate model & trainer via Hydra
    model   = hydra.utils.instantiate(cfg.model,   dataset=train_ds)
    trainer = hydra.utils.instantiate(cfg.trainer)

    # 5) train + validate
    trainer.fit(
        model,
        train_dataloaders = train_loader,
        val_dataloaders   = val_loader,
    )

    # 6) final test
    trainer.test(
        model,
        test_dataloaders = test_loader,
    )

if __name__ == "__main__":
    main()
