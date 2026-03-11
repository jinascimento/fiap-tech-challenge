import os
import re
import sqlite3

from agents.constants import SELECT_EXAMS, SELECT_PATIENT
from agents.state import AgentState
from config.logger import get_logger
from config.settings import settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = str(settings.DATABASE_VECTOR_STORE)

embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

if os.path.exists(INDEX_PATH):
    vector_db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    vector_db = None
    print("⚠️ Alerta: Banco de vetores não encontrado em", INDEX_PATH)


def node_extractor(state: AgentState):
    """Recupera o ID do paciente a partir do texto da pergunta."""
    texto = state["input"]

    match = re.search(r"\d{3,}", texto)
    patient_id = match.group(0) if match else None

    if not patient_id:
        logger.error("Falha ao extrair ID do paciente.")

        return {
            "patient_id": "DESCONHECIDO",
            "logs": ["Falha ao extrair ID do paciente."],
        }

    logger.info(f"ID {patient_id} extraído com sucesso.")

    return {
        "patient_id": patient_id,
        "logs": [f"ID {patient_id} extraído com sucesso."],
    }


def node_loader(state: AgentState):
    """Carrega os dados do paciente a partir do banco de dados."""
    conn = sqlite3.connect("data/hospital.db")
    cursor = conn.cursor()

    # Simulando um ambiente real, implementamos uma camada de persistência (com SQLite,
    # pela simplicidade), essa poderia alimentada por processos de ETL que garantissem
    # a integridade, consistência e anonimização dos dados.

    patient_id = state["patient_id"]

    cursor.execute(SELECT_PATIENT, (patient_id,))

    patient = cursor.fetchone()

    cursor.execute(SELECT_EXAMS, (patient_id,))

    exams = cursor.fetchall()

    conn.close()

    message = f"Dados recuperados via SQL para o paciente {patient_id}."

    if not patient:
        message = "Paciente não encontrado no banco de dados."

        logger.warning(message)

        return {"logs": [message]}

    data_context = {
        "perfil": {"idade": patient[0], "historico": patient[1]},
        "exames": [{"tipo": e[0], "valor": e[1]} for e in exams],
    }

    logger.info(message)

    return {"patient_data": data_context, "logs": [message]}


def node_classifier(state: AgentState):
    """Classifica a pergunta do médico."""
    pergunta = state["input"].lower()

    categoria = "DUVIDA_GERAL"

    if "exame" in pergunta or "laudo" in pergunta:
        categoria = "ANALISE_CLINICA"

    message = f"Pergunta classificada como {categoria}."

    logger.info(message)

    return {"category": categoria, "logs": [message]}


def node_assistant(state: AgentState):
    """Analisa os dados do paciente usando RAG e gera uma resposta contextualizada."""
    logger.info("--- EXECUTANDO ANALISTA COM RAG ---")

    pergunta_usuario = state.get("input", "")
    dados_paciente = state.get("patient_data", {})

    contexto_rag = (
        "Não foram encontrados protocolos específicos nos documentos internos."
    )
    fontes_encontradas = ["Base de conhecimento geral"]

    # 1. Busca no FAISS (Vector Store)
    if vector_db:
        # Criamos uma query que combina a pergunta com o contexto do paciente
        query_busca = f"Paciente: {dados_paciente}. Pergunta: {pergunta_usuario}"
        docs = vector_db.similarity_search(query_busca, k=3)

        if docs:
            contexto_rag = "\n\n".join([d.page_content for d in docs])
            fontes_encontradas = list(
                set(
                    [
                        os.path.basename(d.metadata.get("source", "Protocolo"))
                        for d in docs
                    ]
                )
            )

    # 2. Lógica de Risco baseada no conteúdo recuperado
    risco = "BAIXO"
    if any(
        term in contexto_rag.lower()
        for term in ["sepse", "crítico", "urgente", "elevado"]
    ):
        risco = "ALTO"

    # 3. Resposta Estruturada
    analise_clinica = f"Com base nos protocolos ({', '.join(fontes_encontradas)}):\n{contexto_rag[:500]}..."

    resposta_final = (
        f"**ANÁLISE DO CASO:**\n{analise_clinica}\n\n"
        f"**SUGESTÃO:** Seguir protocolo de {'Urgência' if risco == 'ALTO' else 'Monitoramento'}."
    )

    return {
        "response": resposta_final,
        "sources": fontes_encontradas,
        "risk_level": risco,
        "logs": [f"RAG: {len(fontes_encontradas)} documentos consultados."],
    }


def node_reviewer(state: AgentState):
    """Revisa a resposta do assistente e adiciona um disclaimer."""
    resposta_atual = state.get("response", "")
    nivel_risco = state.get("risk_level", "BAIXO")
    fontes = state.get("sources", [])

    disclaimer = (
        "\n\n--- AVISO DE SEGURANÇA ---\n"
        "Esta é uma sugestão gerada por IA baseada em protocolos institucionais. "
        "Não substitui o julgamento clínico. Validação humana obrigatória antes de qualquer conduta."
    )

    alerta_emergencia = ""

    if nivel_risco == "ALTO":
        alerta_emergencia = "\n⚠️ ALERTA: Indicadores de risco elevado detectados. Considere avaliação imediata."

    rodape_fontes = "\nFontes consultadas: " + ", ".join(fontes) if fontes else ""

    resposta_final = f"{alerta_emergencia}\n{resposta_atual}{disclaimer}{rodape_fontes}"

    logger.info(f"Resposta final: {resposta_final}.")

    return {
        "response": resposta_final,
        "logs": ["Revisão final concluída: Disclaimer e fontes anexados."],
    }
