from config.settings import settings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


def criar_vector_store():
    print("--- INICIANDO INGESTÃO DE PROTOCOLOS ---")

    # 1. Carrega todos os PDFs da pasta data/protocols
    loader = PyPDFDirectoryLoader(settings.PROTOCOLS_DIR)
    docs = loader.load()
    print(f"Documentos carregados: {len(docs)} páginas.")

    # 2. Divide em pedaços menores (Chunks)
    # Importante para a LLM não se perder e para caber no contexto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    print(f"Total de chunks gerados: {len(chunks)}")

    # 3. Define o modelo de Embedding
    # Usando um modelo do HuggingFace que roda na sua CPU/GPU local (sem custo)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Cria e salva o banco de vetores localmente
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(str(settings.DATABASE_VECTOR_STORE))

    print("✅ Banco de vetores criado com sucesso em 'data/vector_index'!")


if __name__ == "__main__":
    criar_vector_store()
