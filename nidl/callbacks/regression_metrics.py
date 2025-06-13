from pytorch_lightning.callbacks import Callback
import torch
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


class RegressionMetricsCallback(Callback):
    """
    Callback to compute and log regression metrics at the end of each validation epoch.
    This callback computes RMSE, MAE, R2, and Pearson correlation coefficient
    using the outputs from the validation step.
    # TODO: handle multivariate regression + logging at different moment (training step, epoch or validation epoch)
    """

    def __init__(self, 
                 when: str="on_validation_epoch_end", 
                 prog_bar: bool=True):
        """
        Parameters
        ----------
        when: str in {"on_train_epoch_end", "on_validation_epoch_end"}, 
            default="on_validation_epoch_end"
            Specifies when to compute the metrics.

        prog_bar: bool, default=True
            Whether to display the metrics in the progress bar.
        """
        super().__init__()
        self.when = when
        self.prog_bar = prog_bar
        self._cache = []

    def compute_metrics(self, outputs):
        if not outputs:
            return

        # Aggregate all y_true and y_pred
        y_true = torch.cat([out["y_true"] for out in outputs])
        y_pred = torch.cat([out["y_pred"] for out in outputs])

        y_true_np = y_true.cpu().detach().numpy().flatten()
        y_pred_np = y_pred.cpu().detach().numpy().flatten()

        # Compute metrics
        rmse = root_mean_squared_error(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        r2 = r2_score(y_true_np, y_pred_np)
        pearson_r, _ = pearsonr(y_true_np, y_pred_np)

        return {"rmse": rmse, "mae": mae, "r2": r2, "pearson_r": pearson_r}

    def on_validation_batch_end(self, trainer, pl_module, outputs, 
                                batch, batch_idx, dataloader_idx=0):
        if self.when == "on_validation_epoch_end":
            assert "y_pred" in outputs and "y_true" in outputs, \
                "'y_pred' or 'y_true' are missing in your estimator's outputs " \
                "(should be returned in 'validation_step')"
            self._cache.append(outputs)

    def on_train_batch_end(self, trainer, pl_module, outputs, 
                           batch, batch_idx):
        if self.when == "on_train_epoch_end":
            assert "y_pred" in outputs and "y_true" in outputs, \
                "'y_pred' or 'y_true' are missing in your estimator's outputs " \
                "(should be returned in 'training_step')"
            self._cache.append(outputs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.when == "on_validation_epoch_end":
            metrics = self.compute_metrics(self._cache)
            for m, value in metrics.items():
                pl_module.log(f"val_{m}", value, prog_bar=self.prog_bar, on_epoch=True)
            # Free the cache
            self._cache = []

    def on_train_epoch_end(self, trainer, pl_module):
        if self.when == "on_train_epoch_end":
            metrics = self.compute_metrics(self._cache)
            for m, value in metrics.items():
                pl_module.log(f"train_{m}", value, prog_bar=self.prog_bar, on_epoch=True)
            # Free the cache
            self._cache = []