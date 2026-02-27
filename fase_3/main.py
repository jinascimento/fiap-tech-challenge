import logging
import os
import sys

from config.settings import settings
from pre_processing.pipeline import run as pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler(stream=sys.stdout),
    ],
)

logger = logging.getLogger("fase_3")

os.environ["HF_TOKEN"] = settings.HF_TOKEN


if __name__ == "__main__":
    logger.info("Executando o pipeline para carregamento dos dados para treinamento")

    pipeline()
