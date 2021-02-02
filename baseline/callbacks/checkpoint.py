import logging

from baseline.utils import load_obj
from pytorch_lightning.callbacks import ModelCheckpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CustomModelCheckpoint(ModelCheckpoint):
    """
    This class is a hotfix for a bug occurring with ModelCheckpoint (pl 1.0.7):
    when resuming training, the trainer gets stuck at the latest epoch.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, trainer, module):
        # Update manually lr_scheduler, which is not updated correctly when
        # a training is resumed
        if module.lr_scheduler is not None:

            lr_scheduler = load_obj(module.lr_scheduler.class_name)(
                optimizer=trainer.optimizers[0],
                last_epoch=trainer.current_epoch - 1,
                **module.lr_scheduler.params,
            )
            trainer.lr_schedulers[0]["scheduler"] = lr_scheduler

        # Force checkpoint saving after each epoch
        self.save_checkpoint(trainer, module)
