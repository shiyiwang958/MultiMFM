"""Base LightningModule with list-based metric logging.

Ported from DNA-MFM@187fe7b ``lightning_modules/general_module.py``. Changes:
package imports; ``wandb`` is imported lazily (only when ``--wandb`` is on).
"""

import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from dmfm.utils.log_utils import get_logger


def _wandb():
    import wandb

    return wandb

logger = get_logger(__name__)


def _ragged_log_frame(log):
    return pd.DataFrame({key: pd.Series(value) for key, value in log.items()})



class GeneralModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args

        self.iter_step = -1
        self._log = defaultdict(list)
        self.generator = np.random.default_rng()
        self.last_log_time = time.time()


    def try_print_log(self):

        step = self.iter_step if self.args.validate else self.trainer.global_step
        if (step + 1) % self.args.print_freq == 0:
            print(os.environ["MODEL_DIR"])
            log = self._log
            log = {key: log[key] for key in log if "iter_" in key}

            log = self.gather_log(log, self.trainer.world_size)
            mean_log = self.get_log_mean(log)
            mean_log.update(
                {'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})
            if self.trainer.is_global_zero:
                print(str(mean_log))
                self.log_dict(mean_log, batch_size=1)
                if self.args.wandb:
                    _wandb().log(mean_log, step=int(self.trainer.global_step))
            for key in list(log.keys()):
                if "iter_" in key:
                    del self._log[key]

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
            data = data.item() if data.numel() == 1 else data.reshape(-1).tolist()
        log = self._log
        if self.args.validate or self.stage == 'train':
            log["iter_" + key].append(data)
        log[self.stage + "_" + key].append(data)

    def on_train_epoch_end(self):
        log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update(
            {'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:
                _wandb().log(mean_log, step=int(self.trainer.global_step))

        for key in list(log.keys()):
            if "train_" in key:
                del self._log[key]

    def on_validation_epoch_end(self):
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update(
            {'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:
                _wandb().log(mean_log, step=int(self.trainer.global_step))

            path = os.path.join(
                os.environ["MODEL_DIR"], f"val_{self.trainer.global_step}.csv"
            )
            _ragged_log_frame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]



    def gather_log(self, log, world_size):
        if world_size == 1:
            return log
        log_list = [None] * world_size
        torch.distributed.all_gather_object(log_list, log)
        log = {key: sum([l[key] for l in log_list], []) for key in log}
        return log

    def get_log_mean(self, log):
        out = {}
        for key in log:
            try:
                out[key] = np.nanmean(log[key])
            except Exception:
                pass
        return out
