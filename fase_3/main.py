import argparse
import logging
import os

from config.settings import settings
from pre_processing.pipeline import run as pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("fase_3")

os.environ["HF_TOKEN"] = settings.HF_TOKEN


def main():
    parser = argparse.ArgumentParser(
        description="Fase 3: pipeline de dados e/ou fine-tuning Llm (QLoRA + LoRA)."
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Executa o pipeline de pré-processamento (gera dataset JSONL).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Executa o fine-tuning (QLoRA + LoRA) usando o dataset em data/dataset_medico_treinamento.jsonl.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Executa pipeline e em seguida o treinamento.",
    )
    args = parser.parse_args()

    if args.full:
        args.pipeline = True
        args.train = True

    if args.pipeline:
        logger.info(
            "Executando o pipeline para carregamento dos dados para treinamento"
        )
        pipeline()

    if args.train:
        from trainning.train_llm import run_training

        logger.info("Iniciando fine-tuning Llm (QLoRA + LoRA)")
        run_training(
            model_name=settings.MODEL_NAME,
            train_file=settings.TRAIN_FILE,
            output_dir=settings.BASE_DIR / "output_llm",
        )

    if not (args.pipeline or args.train):
        parser.print_help()
        logger.info("Nenhuma ação escolhida. Use --pipeline, --train ou --full.")


if __name__ == "__main__":
    main()
