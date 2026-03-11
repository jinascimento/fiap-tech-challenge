import os

from config.logger import get_logger
from config.settings import settings
from pre_processing.pipeline import run as pipeline

logger = get_logger("fase_3")

os.environ["HF_TOKEN"] = settings.HF_TOKEN


if __name__ == "__main__":
    logger.info("Executando o pipeline para carregamento dos dados para treinamento")

    pipeline()

    logger.info("Realizando o fine tuning no modelo")

    logger.info("...")
