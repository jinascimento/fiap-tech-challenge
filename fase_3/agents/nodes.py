import os
import re
import sqlite3

from agents.constants import SELECT_EXAMS, SELECT_PATIENT
from agents.state import AgentState
from config.logger import get_logger
from config.settings import settings

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from trainning.infer_llm import generate_answer, load_model

logger = get_logger("agent")

model, tokenizer = load_model()

vector_db = None
embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)


if settings.EMBEDDING_VECTOR_STORE.exists():
    vector_db = FAISS.load_local(
        settings.EMBEDDING_VECTOR_STORE.as_posix(),
        embeddings,
        allow_dangerous_deserialization=True,
    )
else:
    logger.warning(
        f"Base vetorial não encontrada em {settings.EMBEDDING_VECTOR_STORE}."
    )
    vector_db = None


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

    # Simulando um ambiente real, implementamos uma camada de persistência
    # (com SQLite, pela simplicidade), essa base poderia ser alimentada por
    # processos de ETL que garantissem a integridade, consistência e
    # anonimização dos dados.

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

    # Busca no FAISS (Vector Store)
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

    # Lógica de Risco baseada no conteúdo recuperado
    risco = "BAIXO"
    terms = ["sepse", "crítico", "urgente", "elevado"]
    rag_context = contexto_rag.lower()

    if any(term in rag_context for term in terms):
        risco = "ALTO"

    # Resposta estruturada
    dados = (
        f"Dados do paciente: {dados_paciente}\n"
        f"Pergunta do médico: {pergunta_usuario}\n"
        f"Protocolos consultados: {', '.join(fontes_encontradas)}\n"
        f"Contexto RAG: {contexto_rag[:500]}...\n"
    )

    analise_clinica = (
        "Com base nos dados abaixo, responda à pergunta do médico, "
        f"destacando os pontos críticos que levaram à classificação de risco {risco}:\n\n"
        f"{dados}"
    )

    analise = generate_answer(model, tokenizer, analise_clinica)

    sugestao_caso = (
        f"Baseado na análise abaixo, faça recomendações que o médico possa seguir:\n\n"
        f"{analise}"
    )

    sugestao = generate_answer(model, tokenizer, sugestao_caso)

    resposta_final = (
        f"**ANÁLISE DO CASO:** {analise}\n\n**SUGESTÃO:** {sugestao}"
        # f"**SUGESTÃO:** Seguir protocolo de {'Urgência' if risco == 'ALTO' else 'Monitoramento'}."
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
        alerta_emergencia = "\n⚠️ ALERTA: Indicadores de risco elevado detectados. Considere avaliação imediata.\n\n"

    rodape_fontes = "\nFontes consultadas: " + ", ".join(fontes) if fontes else ""

    resposta_final = f"{alerta_emergencia}\n{resposta_atual}{disclaimer}{rodape_fontes}"

    logger.info(f"Resposta final: {resposta_final}.")

    return {
        "response": resposta_final,
        "logs": ["Revisão final concluída: Disclaimer e fontes anexados."],
    }
