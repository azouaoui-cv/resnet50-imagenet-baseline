"""
Lightning Data Module class
"""
import logging

import pytorch_lightning as pl
from baseline.utils import load_obj
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

MEANS = {
    "ImageNet": (0.485, 0.456, 0.406),
}
STDS = {
    "ImageNet": (0.229, 0.224, 0.225),
}


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.name = cfg.datasets.name
        self.class_name = cfg.dataset.class_name
        self.root = to_absolute_path(cfg.datasets.path)
        self.loader_params = cfg.data.loader_params
        ##############
        # Transforms #
        ##############
        means, stds = MEANS[self.name], STDS[self.name]
        logger.debug(f"hard coded means: {means}, stds: {stds}")
        train_crop_size = cfg.datasets.trfs_params.train_crop_size
        val_crop_size = cfg.datasets.trfs_params.val_crop_size
        val_resize = cfg.datasets.trfs_params.val_resize

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(train_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.Resize(val_resize),
                transforms.CenterCrop(val_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
            ]
        )
        self.test_transforms = self.val_transforms

    def __repr__(self):
        msg = f"Dataset: {self.name} ({self.class_name}) @ {self.root}"
        return msg

    def setup(self, stage=None):
        train_data = load_obj(self.class_name)(
            root=self.root,
            split="train",
            transform=self.train_transforms,
        )
        val_data = load_obj(self.class_name)(
            root=self.root,
            split="val",
            transform=self.val_transforms,
        )
        test_data = load_obj(self.class_name)(
            root=self.root,
            split="val",
            transform=self.test_transforms,
        )

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, **self.loader_params)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, **self.loader_params)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, **self.loader_params)
