import logging
import sys

from config.settings import settings


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler(stream=sys.stdout),
        ],
    )


setup_logging()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
