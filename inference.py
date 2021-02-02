import logging
import os

import hydra
import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from baseline.data import DataModule
from baseline.utils import load_obj

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test(cfg):

    # Load model with current config
    datamodule = DataModule(cfg)
    # Check for test checkpoint
    if cfg.test.checkpoint is not None:
        logger.info("Attempting to test checkpoint")
        checkpoint = to_absolute_path(cfg.test.checkpoint)
        # Get checkpoint path
        ckpt_path = os.path.join(
            checkpoint,
            cfg.checkpoint.params.dirpath,
            "last.ckpt",
        )
        logger.debug(f"ckpt path: {ckpt_path}")
        if os.path.exists(ckpt_path):
            logger.info(f"Loading existing checkpoint @ {ckpt_path}")
            ckpt_cfg = OmegaConf.load(os.path.join(checkpoint, "config.yaml"))
            # Reload trainer
            trainer = pl.Trainer(
                resume_from_checkpoint=ckpt_path,
                gpus=cfg.test.gpus,
            )
            # Load model
            model = load_obj(ckpt_cfg.model.class_name)(ckpt_cfg)
        else:
            logger.info("No existing ckpt found. Aborting")
            return None
    else:
        logger.info("No test checkpoint has been provided")
        return None
    # Save results
    txt = f"{model.__repr__()}\n"
    txt += f"{datamodule.__repr__()}\n"
    txt += f"exp ID: {ckpt_cfg.id}\n"
    # Test trainer
    results = trainer.test(
        model=model,
        ckpt_path="best",
        datamodule=datamodule,
    )
    txt += f"{str(results)}\n"
    logger.info(txt)

    # write results to file
    with open("test_results.txt", "w") as f:
        f.write(txt)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    logger.info(f"Current working directory : {os.getcwd()}")
    try:
        test(cfg)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()
