import logging
from pathlib import Path

import joblib

from .diabetes import load_dataset, prepare_dataset, train_mlp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "mlp_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
COLUMNS_PATH = ARTIFACTS_DIR / "columns.pkl"


def train_and_save_model():
    """Treina o modelo e salva os artefatos em disco."""
    logger.info("Iniciando treinamento do modelo...")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df_processed = prepare_dataset(df)

    baseline_params = {
        "hidden_layer_sizes": (32, 32),
        "learning_rate_init": 0.01,
        "activation": "relu",
    }

    model, scaler, X_test, _ = train_mlp(df_processed, baseline_params)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(X_test.columns, COLUMNS_PATH)

    logger.info("Modelo treinado e salvo com sucesso.")

    return model, scaler, X_test.columns


def load_inference_model(force_retrain: bool = False):
    """
    Carrega o modelo treinado. Se não existir, treina um novo.
    """
    if force_retrain or not (
        MODEL_PATH.exists() and SCALER_PATH.exists() and COLUMNS_PATH.exists()
    ):
        return train_and_save_model()

    try:
        logger.info("Carregando modelo do disco...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        columns = joblib.load(COLUMNS_PATH)
        return model, scaler, columns
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}. Retreinando...")
        return train_and_save_model()
