import sqlite3

from config.logger import get_logger
from config.settings import settings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger("agent")


def setup_database():
    is_database_exists = settings.DATABASE_FILE.exists()
    is_regeneration_forced = settings.DATABASE_FORCE_REGENERATION

    if is_database_exists and not is_regeneration_forced:
        logger.info(f"Base de dados já existe em {settings.DATABASE_FILE}")

        return

    conn = sqlite3.connect(settings.DATABASE_FILE)
    cursor = conn.cursor()

    # Tabela de pacientes (dados anonimizados)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pacientes (
            id INTEGER PRIMARY KEY,
            idade INTEGER,
            genero TEXT,
            historico TEXT,
            risco_previo TEXT
        )
    """)

    # Tabela de exames
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exames (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            tipo_exame TEXT,
            resultado TEXT,
            data TEXT,
            FOREIGN KEY (patient_id) REFERENCES pacientes (id)
        )
    """)

    # Adiciona dados sintéticos para teste
    pacientes_teste = [
        (123, 45, "M", "Hipertensão leve, ex-fumante", "MÉDIO"),
        (456, 30, "F", "Sem comorbidades, atleta", "BAIXO"),
        (789, 68, "M", "Diabetes Tipo 2, Insuficiência Renal Crônica", "ALTO"),
    ]

    exames_teste = [
        (123, "Creatinina", "1.4 mg/dL", "2023-10-01"),
        (123, "Glicose", "105 mg/dL", "2023-10-01"),
        (789, "Creatinina", "2.8 mg/dL", "2023-10-05"),
        (789, "Ureia", "90 mg/dL", "2023-10-05"),
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO pacientes VALUES (?,?,?,?,?)", pacientes_teste
    )
    cursor.executemany(
        "INSERT OR REPLACE INTO exames (patient_id, tipo_exame, resultado, data) VALUES (?,?,?,?)",
        exames_teste,
    )

    conn.commit()
    conn.close()

    logger.info(f"Base de dados criada e populada: {settings.DATABASE_FILE}")


def create_vector_store():
    is_database_exists = settings.EMBEDDING_VECTOR_STORE.exists()
    is_regeneration_forced = settings.EMBEDDING_FORCE_REGENERATION

    if is_database_exists and not is_regeneration_forced:
        logger.info(f"Base vetorial já existe em {settings.EMBEDDING_VECTOR_STORE}")

        return

    loader = PyPDFDirectoryLoader(settings.PROTOCOLS_DIR)
    docs = loader.load()

    logger.info(f"Documentos carregados: {len(docs)} páginas.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    logger.info(f"Total de chunks gerados: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(str(settings.EMBEDDING_VECTOR_STORE))

    logger.info(f"Base vetorial criada e salvo em: {settings.EMBEDDING_VECTOR_STORE}")
