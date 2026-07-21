"""Modelos Pydantic usados pelo pipeline STRIDE.

Divide-se em dois grupos:

* Tecnico: preenchido pelo parser a partir do texto bruto do modelo STRIDE.
  Deve ser fiel ao relatorio original.
* Amigavel: preenchido pelo agente Pydantic AI (Gemini). Traducao para
  linguagem acessivel a nao-tecnicos.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CategoriaSTRIDE = Literal[
    "Spoofing",
    "Tampering",
    "Repudiation",
    "Information Disclosure",
    "Denial of Service",
    "Elevation of Privilege",
]

Severidade = Literal["baixa", "media", "alta", "critica"]


class Componente(BaseModel):
    """Componente identificado no diagrama pelo modelo STRIDE."""

    idx: int = Field(description="Indice numerico do componente (ex: 3 em component#3)")
    nome: str
    boundaries: list[str] = Field(default_factory=list)


class AmeacaTecnica(BaseModel):
    """Ameaca STRIDE tal como aparece no relatorio original."""

    categoria: CategoriaSTRIDE
    componente_ref: str = Field(
        description="Referencia crua ao componente, ex: 'edge_ddos_protection (component#3)'",
    )
    descricao: str
    contramedida: str | None = None


class RelatorioTecnico(BaseModel):
    """Representacao estruturada do relatorio bruto emitido pelo modelo STRIDE."""

    arquitetura: str = Field(description="Nome da arquitetura analisada")
    componentes: list[Componente]
    ameacas: list[AmeacaTecnica]

    @property
    def total_componentes(self) -> int:
        return len(self.componentes)

    @property
    def total_ameacas(self) -> int:
        return len(self.ameacas)


class AmeacaAmigavel(BaseModel):
    """Ameaca reescrita em linguagem acessivel para nao-tecnicos."""

    componente: str = Field(description="Nome do componente, exatamente como no relatorio original")
    categoria: CategoriaSTRIDE
    severidade: Severidade = Field(
        description="Estimativa de severidade baseada no impacto e alcance da ameaca",
    )
    explicacao_leigo: str = Field(
        description="Explicacao em portugues simples, com analogias do dia a dia, sem jargao",
    )
    o_que_pode_acontecer: str = Field(
        description="Consequencia concreta caso a ameaca se materialize",
    )
    como_mitigar_em_portugues: str = Field(
        description="Contramedida traduzida em linguagem clara, fiel ao original",
    )


class RelatorioAmigavel(BaseModel):
    """Relatorio final humanizado, retornado pelo agente Pydantic AI."""

    resumo_executivo: str = Field(
        description="2-3 frases resumindo os riscos para um gestor nao tecnico",
    )
    total_componentes: int
    total_ameacas: int
    principais_riscos: list[AmeacaAmigavel] = Field(
        description="Entre 5 e 8 ameacas mais criticas, priorizadas por severidade",
    )
    proximos_passos: list[str] = Field(
        description="Entre 3 e 5 acoes praticas recomendadas em ordem de prioridade",
    )
