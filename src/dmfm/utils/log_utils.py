"""Logging helpers (ported from DNA-MFM@187fe7b ``utils/logging.py``).

Changes vs. the original: no import-time side effects. The original created
``./workdir/default/log.out`` when imported and could ``dup2`` stdout into a
file when W&B logging was on; here ``get_logger`` only adds a file handler when
``MODEL_DIR`` is set at call time, and ``Logger`` is the same stdout tee that
``parse_train_args`` installs.
"""

from datetime import datetime
import logging
import os
import socket


def get_logger(name):
    logger = logging.Logger(name)
    level = {"crititical": 50, "error": 40, "warning": 30, "info": 20, "debug": 10}[
        os.environ.get("LOGGER_LEVEL", "info")
    ]
    logger.setLevel(level)
    formatter = logging.Formatter(
        f"%(asctime)s [{socket.gethostname()}:%(process)d] [%(levelname)s] %(message)s"
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    model_dir = os.environ.get("MODEL_DIR")
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(model_dir, "log.out"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


class Logger(object):
    """Tee ``sys.stdout``/``sys.stderr`` into a log file."""

    def __init__(self, logpath, syspart):
        self.terminal = syspart
        self.log = open(logpath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

    def isatty(self):
        return False


def lg(*args):
    print(f"[{datetime.now()}] [{socket.gethostname()}]", *args)
