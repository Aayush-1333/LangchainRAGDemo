import logging
import sys

from colorlog import ColoredFormatter


def setup_logger(name: str = __name__, mode: str = "DEBUG") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO
    }

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_levels[mode])  # Control console verbosity

        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            },
            reset=True,
            secondary_log_colors={},
            style='%'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
