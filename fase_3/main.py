import argparse
import logging
import os

from config.settings import settings

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
        "--preprocess",
        action="store_true",
        help="Executa o pipeline de pré-processamento (gera dataset JSONL).",
    )

    parser.add_argument(
        "--agent",
        action="store_true",
        help="Prepara os dados para consumo do agente (base relacional e vetorial).",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Executa o fine-tuning (QLoRA + LoRA) usando o dataset em data/dataset_medico_treinamento.jsonl.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Executa pre-processamento, cria a bases relacional e vetorial para o agente e treina o modelo.",
    )

    parser.add_argument(
        "--app",
        action="store_true",
        help="Instruções sobre como executar a aplicação web.",
    )

    args = parser.parse_args()

    if args.full:
        args.preprocess = True
        args.agent = True
        args.train = True

    if args.preprocess:
        from pre_processing.pipeline import run as preprocess

        preprocess()

    if args.agent:
        from agents.utils import create_vector_store, setup_database

        setup_database()
        create_vector_store()

    if args.train:
        from trainning.train_llm import run_training

        run_training(
            model_name=settings.MODEL_NAME,
            train_file=settings.TRAIN_FILE,
            output_dir=settings.BASE_DIR / "output_llm",
        )

    if args.app:
        logger.info("""
            Para executar a aplicação web, siga os passos abaixo:

            1. Certifique-se de que os pre-requisitos foram satisfeitos
                1.1. Execute o pré-processamento dos dados usando --preprocess
                1.2. Prepare as bases de dados para o agente usando --agent
                1.3. Treine o modelo usando --train
            2. Execute a aplicação web usando o comando: `uv run streamlit run ui.py`
        """)

    if not (args.preprocess or args.agent or args.train or args.app):
        parser.print_help()

        logger.info("Nenhuma opção escolhida.")


if __name__ == "__main__":
    main()
