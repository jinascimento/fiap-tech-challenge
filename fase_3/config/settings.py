from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"

    # Output files
    TRAIN_FILE: Path = DATA_DIR / "dataset_medico_treinamento.jsonl"
    LOG_FILE: Path = LOG_DIR / "preprocessing_audit.log"

    # Model
    MODEL_NAME: str = "meta-llama/Llama-3.2-3B"

    # Utils
    HF_TOKEN: str = "HF_XYZ"
    FORCE_DOWNLOAD: bool = False

    class Config:
        env_file = ".env"


settings = Settings()

# Inicialização de diretórios
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
