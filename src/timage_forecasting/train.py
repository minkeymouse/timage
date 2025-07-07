# src/timage_forecasting/train.py
import pandas as pd
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf, DictConfig

from timage_forecasting.datamodule import TimeSeriesWithImageDataModule
import inspect
from pytorch_forecasting import TimeSeriesDataSet
print(inspect.signature(TimeSeriesDataSet.__init__))
from timage_forecasting.model import Timage

@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    # 1) reproducibility
    print("––––––––– cfg.datamodule –––––––––")
    print(OmegaConf.to_yaml(cfg.datamodule))
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # 2) load your data, casting parse_dates to a list
    parse_dates = list(cfg.datamodule.parse_dates)
    df      = pd.read_csv(cfg.datamodule.df_path,      parse_dates=parse_dates)
    test_df = pd.read_csv(cfg.datamodule.test_df_path, parse_dates=parse_dates)

    # 2a) cast all declared categorical columns to string
    cat_cols = (
        cfg.datamodule.static_categoricals
        + cfg.datamodule.time_varying_known_categoricals
        + cfg.datamodule.time_varying_unknown_categoricals
    )
    for col in cat_cols:
        df[col]      = df[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    # 3) filter to only the panels you care about
    keep_ids = list(cfg.datamodule.keep_ids)
    df      = df     [df    .panel_id.isin(keep_ids)].reset_index(drop=True)
    test_df = test_df[test_df.panel_id.isin(keep_ids)].reset_index(drop=True)

    # 4) instantiate your LightningDataModule
    datamodule = hydra.utils.instantiate(
        cfg.datamodule,
        df=df,
        test_df=test_df,
        _recursive_=False,
    )
    datamodule.setup(stage="fit")

    # 4) instantiate Timage via from_dataset (so BaseModelWithCovariates.__init__ runs cleanly)
    #    filter out the _target_ key so we only pass real hyperparameters
    model_kwargs = { k: v for k, v in cfg.model.items() if k != "_target_" }
    model = Timage.from_dataset(datamodule.train_ds, **model_kwargs)
    # 6) instantiate your Trainer
    trainer = hydra.utils.instantiate(cfg.trainer)

    # 7) fit + test
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
