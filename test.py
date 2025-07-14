from timage_forecasting.datamodule import TimeSeriesWithImage, TimeSeriesWithImageDataModule
import pandas as pd
import numpy as np

df = pd.read_csv("experiments/train_processed_img_task5.csv")

print(df)