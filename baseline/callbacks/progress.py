"""
Custom progress bar callback for Pytorch Lightning
"""
###########
# Imports #
###########
# Standard library
import importlib
import sys
import pdb
# progress bar
if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm
else:
    from tqdm import tqdm
# base callback
from pytorch_lightning.callbacks import ProgressBar
###########
# Classes #
###########
class LitProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.bar_format = "{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"

    def init_sanity_tqdm(self) -> tqdm:
        bar = super().init_sanity_tqdm()
        bar.bar_format = self.bar_format
        return bar

    def init_train_tqdm(self) -> tqdm:
        bar = super().init_train_tqdm()
        bar.bar_format = self.bar_format
        return bar

    def init_validation_tqdm(self) -> tqdm:
        bar = super().init_validation_tqdm()
        bar.bar_format = self.bar_format
        return bar

    def init_test_tqdm(self) -> tqdm:
        bar = super().init_test_tqdm()
        bar.bar_format = self.bar_format
        return bar
