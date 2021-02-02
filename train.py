"""
File to train a model for image classificaton
"""
import logging
import os
import shutil
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import GPUStatsMonitor, LearningRateMonitor

from baseline.callbacks import CustomModelCheckpoint, LitProgressBar
from baseline.data import DataModule
from baseline.utils import load_obj

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


warnings.simplefilter(action="ignore", category=FutureWarning)


def train(cfg: DictConfig) -> None:
    """
    Train a model for image classification

    Args:
        cfg: hydra configuration
    """
    # Load pre-existing config file
    if os.path.exists("config.yaml"):
        logging.info("Loading pre-existing config file")
        cfg = OmegaConf.load("config.yaml")
    else:
        # copy initial config to a separate file to avoid overwriting it
        # when hydra resumes training and initializes again
        shutil.copy2(".hydra/config.yaml", "config.yaml")

    # Check for checkpoint
    ckpt_path = os.path.join(os.getcwd(), cfg.checkpoint.params.dirpath, "last.ckpt")
    if os.path.exists(ckpt_path):
        logging.info(f"Loading existing checkpoint @ {ckpt_path}")
    else:
        logging.info("No existing ckpt found. Training from scratch")
        ckpt_path = None

    # Display configuration
    logger.info(OmegaConf.to_yaml(cfg))
    # Seed everything
    seed_everything(cfg.training.seed)
    # Load datamodule
    data = DataModule(cfg)
    # Callbacks
    callbacks = [
        CustomModelCheckpoint(**cfg.checkpoint.params),
        LearningRateMonitor(),
        LitProgressBar(),
    ]
    if cfg.trainer.params.gpus:
        callbacks.append(GPUStatsMonitor())
    # Logger
    trainer_logger = load_obj(cfg.logger.class_name)(**cfg.logger.params)
    # Load model
    model = load_obj(cfg.models.class_name)(cfg)

    # Save model id
    with open("id", "w") as f:
        f.write(cfg.id)

    # Instantiate trainer
    trainer = Trainer(
        resume_from_checkpoint=ckpt_path,
        callbacks=callbacks,
        logger=trainer_logger,
        **cfg.trainer.params,
    )

    # Display model architecture alongside parameters and data
    logger.info(model)
    logger.info(data)
    logger.info(f"random seed: {cfg.training.seed}")
    # Fit trainer
    trainer.fit(model, datamodule=data)
    # Final test


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Current working directory: {os.getcwd()}")
    try:
        train(cfg)
    except Exception as e:
        logger.critical(e, exc_info=True)


########
# Main #
########
if __name__ == "__main__":
    main()
