from langgraph.graph import END, StateGraph

from .nodes import (
    node_assistant,
    node_classifier,
    node_extractor,
    node_loader,
    node_reviewer,
)
from .state import AgentState


def check_id(state: AgentState):
    """Roteador para decidir se seguimos para o banco de dados ou pedimos clarificação."""
    if state["patient_id"] == "DESCONHECIDO":
        return "ask_id"
    return "loader"


def ask_id(state: AgentState):
    """Nó de interrupção amigável quando o ID falta."""
    return {
        "response": "Sinto muito, mas não consegui identificar o ID do paciente na sua mensagem. Por favor, informe o código do paciente (ex: 123) para que eu possa consultar o prontuário.",
        "logs": [
            "Interrupção: O identificador do paciente não foi fornecido ou localizado."
        ],
    }


workflow = StateGraph(AgentState)

workflow.add_node("classifier", node_classifier)
workflow.add_node("extractor", node_extractor)
workflow.add_node("ask_id", ask_id)
workflow.add_node("loader", node_loader)
workflow.add_node("assistant", node_assistant)
workflow.add_node("reviewer", node_reviewer)

workflow.set_entry_point("classifier")

workflow.add_edge("classifier", "extractor")

workflow.add_conditional_edges(
    "extractor",
    check_id,
    {"ask_id": "ask_id", "loader": "loader"},
)

workflow.add_edge("ask_id", END)

workflow.add_edge("loader", "assistant")
workflow.add_edge("assistant", "reviewer")
workflow.add_edge("reviewer", END)

app = workflow.compile()
