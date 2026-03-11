import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    input: str  # Pergunta do médico
    patient_id: str  # ID para busca no DB
    patient_data: dict  # Dados recuperados (exames, histórico)
    category: str  # Classificação (Exame, Protocolo, Alerta)
    response: str  # Resposta final
    sources: list[str]  # EXPLICABILIDADE: De onde veio a info?
    risk_level: str  # SEGURANÇA: Nível de criticidade (Baixo, Médio, Alto)
    logs: Annotated[list[str], operator.add]
