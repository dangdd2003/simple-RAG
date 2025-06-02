import logging
from logging import Logger


def set_logger(name: str = __name__) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.propagate = False
    return logger
