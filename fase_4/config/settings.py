from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    LOG_FILE: Path = LOG_DIR / "fase_4.log"

    # GCP
    PROJECT_ID: str = "gen-lang-client-0574046440"
    BUCKET_NAME: str = "tc4-saude-mulher-storage-bucket"
    GCP_LOCATION: str = "us-central1"

    # Audio generation
    GOOGLE_CREDENTIALS_FILE: Path = BASE_DIR / "google-credentials.json"
    TEST_AUDIOS_DIR: Path = BASE_DIR / "test_audios"

    # Text-to-Speech
    TTS_LANGUAGE_CODE: str = "pt-BR"
    TTS_VOICE_NAME: str = "pt-BR-Neural2-C"

    class Config:
        env_file = ".env"


settings = Settings()

# Inicializacao de diretorios
settings.TEST_AUDIOS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
