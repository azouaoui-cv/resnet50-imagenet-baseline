"""
Base Pytorch Lightning classification model
"""
import logging
import pdb

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from baseline.utils import load_obj
from omegaconf import DictConfig

from .metrics import MyAccuracy, MyTopK

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class LitClassifier(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.lr_scheduler = cfg.scheduler
        self.optimizer = cfg.optimizer
        self.best_val_acc = torch.tensor(0.0)
        self.train_accuracy = MyAccuracy()
        self.val_accuracy = MyAccuracy()
        self.test_accuracy = MyAccuracy()
        self.test_top5 = MyTopK()

    def loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        return F.cross_entropy(outputs, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        loss = self.loss(out, y)

        self.log("train_loss", loss)
        self.log("train_acc_s", self.train_accuracy(out, y))

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        out = self(x)

        val_loss = self.loss(out, y)
        self.log("val_loss", val_loss)

        results = {"val_acc": self.val_accuracy(out, y)}
        return results

    def test_step(self, batch, batch_idx):
        x, y = batch

        if len(x.shape) == 5:  # TenCrop
            bs, ncrops, c, h, w = x.size()
            res = self(x.view(-1, c, h, w))
            out = res.view(bs, ncrops, -1).mean(1)

        else:
            out = self(x)

        results = {
            "test_acc": self.test_accuracy(out, y),
            "test_top5": self.test_top5(out, y),
        }

        return results

    def training_epoch_end(self, outputs):
        self.log("train_acc_e", self.train_accuracy.compute(), prog_bar=True)

    def validation_epoch_end(self, outputs):

        val_acc = self.val_accuracy.compute()

        if self.best_val_acc < val_acc.cpu():
            self.best_val_acc = val_acc.cpu()
            logger.debug(f"New best val acc: {self.best_val_acc:.2f}")

        self.log("val_acc", val_acc, prog_bar=True)
        self.log("best_val_acc", self.best_val_acc, prog_bar=True)

    def test_epoch_end(self, outputs):
        self.log("test_acc_all", self.test_accuracy.compute())
        self.log("test_top5", self.test_top5.compute())

    def configure_optimizers(self):
        optimizer = load_obj(self.optimizer.class_name)(
            self.parameters(), **self.optimizer.params
        )
        scheduler = load_obj(self.lr_scheduler.class_name)(
            optimizer, **self.lr_scheduler.params
        )
        return [optimizer], [scheduler]

    def count_params(self) -> list:
        res = []
        counter = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                count = param.numel()
                res.append(f"\tParam {name} : {count}")
                counter += count
        res.append(f"Total number trainable params : {counter}")
        return res

    def __repr__(self):
        msg = super().__repr__()
        msg += "\n".join(self.count_params())
        return msg

    # Rework the progress_bar_dict
    def get_progress_bar_dict(self) -> dict:
        # Get the running_loss
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = None
        if running_train_loss is not None:
            avg_training_loss = running_train_loss.cpu().item()
        elif self.trainer.train_loop.automatic_optimization:
            avg_training_loss = float("NaN")

        tqdm_dict = {}
        if avg_training_loss is not None:
            tqdm_dict["loss"] = f"{avg_training_loss:.3e}"
        return tqdm_dict
