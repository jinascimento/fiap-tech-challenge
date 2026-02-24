import json
import logging

from config.settings import settings
from datasets import load_dataset
from pre_processing.utils import clean_and_format

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("pipeline")


def fetch_external_data() -> list[dict]:
    data = []

    cache_dir = settings.RAW_DATA_DIR.as_posix()

    ds_pubmed = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_labeled",
        split="train",
        cache_dir=cache_dir,
    )

    for item in ds_pubmed:
        context = clean_and_format(" ".join(item["context"]["contexts"]))
        question = clean_and_format(item["question"])
        answer = clean_and_format(item["long_answer"])

        raw = {
            "instruction": f"Com base no contexto médico, responda: {question}\nContexto: {context}",
            "output": answer,
        }

        data.append(raw)

    logger.info(f"Dados para PubMedQA ({len(data):_}) carregados")

    ds_med = load_dataset(
        "AnonymousSub/MedQuAD_47441_Question_Answer_Pairs",
        split="train",
        cache_dir=cache_dir,
    )

    for item in ds_med:
        question = item["Questions"]
        answer = item["Answers"]

        if question is None or answer is None:
            continue

        cleaned_question = clean_and_format(question)
        cleaned_answer = clean_and_format(answer)

        raw = {
            "instruction": f"Pergunta Clínica: {cleaned_question}",
            "output": cleaned_answer,
        }

        data.append(raw)

    logger.info(f"Dados para MedQuAD ({len(data):_}) carregados")

    return data


def load_synthetic_from_json() -> list[dict]:
    """Carrega dados sintéticos a partir de um arquivo JSON."""
    synthetic_file = settings.DATA_DIR / "hospital_protocols.json"

    if not synthetic_file.exists():
        logger.warning(f"O arquivo {synthetic_file} não foi encontrado. Pulando...")

        return []

    logger.info(f"Carregando e tratando dados de {synthetic_file}...")

    with open(synthetic_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    return [
        {
            "instruction": clean_and_format(item["instruction"]),
            "output": clean_and_format(item["output"]),
        }
        for item in raw_data
    ]


def run():
    if settings.TRAIN_FILE.exists() and not settings.FORCE_DOWNLOAD:
        logger.info("Execução interrompida, dados já carregados.")
        return

    try:
        external_data = fetch_external_data()
        internal_data = load_synthetic_from_json()
        final_dataset = external_data + internal_data

        with open(settings.TRAIN_FILE, "w", encoding="utf-8") as f:
            for item in final_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Pipeline concluído. Dataset salvo em: {settings.TRAIN_FILE}")
    except Exception as e:
        logger.error(f"Falha no pipeline: {e}", exc_info=True)


if __name__ == "__main__":
    run()
