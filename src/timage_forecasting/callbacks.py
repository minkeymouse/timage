from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

class WalkForward(Callback):
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dm = trainer.datamodule
        dm.advance_window()
        dm.setup(stage="fit")


from pathlib import Path
import pandas as pd


class PredictionWriter(Callback):
    """
    A callback for Trainer.predict that gathers all batch‐wise DataFrames
    and writes a single file at the end.
    """

    def __init__(
        self,
        output_dir: str = "output/predictions",
        filename: str = "predictions.csv",
        format: str = "csv",
    ):
        """
        Args:
            output_dir: where to save the file
            filename: name of the file
            format: "csv", "json", or "pickle"
        """
        self.output_dir = Path(output_dir)
        self.filename = filename
        self.format = format.lower()
        # will accumulate list of DataFrames
        self._dfs: list[pd.DataFrame] = []

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: pd.DataFrame,
        batch: any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # each outputs is the DataFrame returned by predict_step
        self._dfs.append(outputs)

    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # concatenate
        if not self._dfs:
            return
        df = pd.concat(self._dfs, ignore_index=True)

        # ensure directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / self.filename

        # write in chosen format
        if self.format == "csv":
            df.to_csv(path, index=False)
        elif self.format == "json":
            df.to_json(path, orient="records", lines=True)
        elif self.format in ("pkl", "pickle"):
            df.to_pickle(path)
        else:
            raise ValueError(f"Unknown format: {self.format}")

        # clear for next call
        self._dfs.clear()
        print(f"✅ Wrote predictions to {path}")
