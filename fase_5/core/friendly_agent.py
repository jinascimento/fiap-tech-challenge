"""Agente Pydantic AI que humaniza o relatorio STRIDE.

Usa o modelo Gemini 2.5 Flash via Google AI Studio (provider ``google``).
Requer a variavel de ambiente ``GEMINI_API_KEY`` (ou ``GOOGLE_API_KEY``),
carregada pelo ``dotenv`` em :mod:`app`.

O agente recebe o :class:`RelatorioTecnico` serializado em JSON e devolve um
:class:`RelatorioAmigavel` tipado. A saida estruturada e garantida pelo
Pydantic AI via ``output_type``.
"""

from __future__ import annotations

import os

from pydantic_ai import Agent

from core.schemas import RelatorioAmigavel, RelatorioTecnico

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


SYSTEM_PROMPT = """\
Voce e um especialista em seguranca da informacao com o dom de explicar riscos \
tecnicos para publico nao-tecnico (gestores, product owners, stakeholders).

Sua tarefa e traduzir um relatorio STRIDE (que sera fornecido em JSON) em uma \
versao amigavel em portugues brasileiro.

Regras obrigatorias:
1. Seja fiel ao relatorio original. Nao invente ameacas nem contramedidas que \
nao estejam presentes.
2. Ao citar um componente, use o nome exatamente como aparece (ex: \
'edge_ddos_protection').
3. Selecione entre 5 e 8 ameacas mais criticas para 'principais_riscos'. \
Priorize ameacas de maior impacto (comprometimento de dados, indisponibilidade, \
elevacao de privilegio).
4. Em cada ameaca amigavel:
   - 'explicacao_leigo': use analogias do cotidiano, evite jargao tecnico.
   - 'o_que_pode_acontecer': descreva a consequencia concreta em uma frase.
   - 'como_mitigar_em_portugues': traduza a contramedida original em linguagem \
     clara, sem alterar a intencao.
5. Atribua 'severidade' com base em impacto e alcance: 'critica' para \
comprometimento total ou vazamento massivo, 'alta' para exposicao seria, \
'media' para riscos importantes mas contidos, 'baixa' para riscos residuais.
6. 'resumo_executivo' deve ter 2-3 frases voltadas para um gestor.
7. 'proximos_passos' deve conter de 3 a 5 acoes praticas em ordem de \
prioridade.
8. Responda sempre em portugues brasileiro.
"""


def _build_agent() -> Agent[None, RelatorioAmigavel]:
    return Agent(
        f"google:{DEFAULT_MODEL}",
        output_type=RelatorioAmigavel,
        system_prompt=SYSTEM_PROMPT,
    )


_agent: Agent[None, RelatorioAmigavel] | None = None


def _get_agent() -> Agent[None, RelatorioAmigavel]:
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


def humanizar(rel: RelatorioTecnico) -> RelatorioAmigavel:
    """Executa o agente e retorna a versao amigavel do relatorio.

    Args:
        rel: Relatorio tecnico parseado.

    Returns:
        :class:`RelatorioAmigavel` validado pelo Pydantic.

    Raises:
        RuntimeError: Se ``GEMINI_API_KEY`` nao estiver definida.
    """
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise RuntimeError(
            "GEMINI_API_KEY (ou GOOGLE_API_KEY) nao esta definida. Copie "
            ".env.example para .env e cole sua chave do Google AI Studio "
            "(https://aistudio.google.com/apikey)."
        )

    payload = rel.model_dump_json(indent=2)
    prompt = (
        "Abaixo esta o relatorio STRIDE tecnico em JSON. Gere a versao "
        "amigavel seguindo estritamente o schema de saida.\n\n"
        f"```json\n{payload}\n```"
    )
    result = _get_agent().run_sync(prompt)
    return result.output
