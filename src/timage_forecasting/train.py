import torch
import pytorch_lightning as pl
from timage_forecasting.datamodule import TimeSeriesWithImageDataModule
from timage_forecasting.model import Timage


if __name__ == "__main__":
    model = Timage()
    datamodule = TimeSeriesWithImageDataModule()
    trainer = pl.Trainer(accelerator=config.ACCELARATOR, devices=config.DEVICES, )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.validate(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)