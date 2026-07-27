"""Parser do texto bruto emitido pelo modelo STRIDE.

Converte o relatorio em texto no formato:

    ##Relatorio de Modelagem de Ameacas (STRIDE) - <arquitetura>
    Componentes identificados (N)
    #<idx> <nome> - boundaries: <b1>, <b2>, ...
    ...

    ##Ameacas STRIDE identificadas (M)
    <Categoria> (K)
    <componente> (component#<idx>): <descricao>
    Contramedida: <texto>
    ...

numa instancia de :class:`RelatorioTecnico`.

O parser e tolerante a espacos extras, linhas em branco entre secoes, ausencia
de contramedida e nomes de arquitetura com traco.
"""

from __future__ import annotations

import re
from typing import get_args

from core.schemas import (
    AmeacaTecnica,
    CategoriaSTRIDE,
    Componente,
    RelatorioTecnico,
)

_CATEGORIAS_VALIDAS: set[str] = set(get_args(CategoriaSTRIDE))

_HEADER_RE = re.compile(
    r"^##\s*Relatorio de Modelagem de Ameacas\s*\(STRIDE\)\s*-\s*(?P<arq>.+)$",
    re.IGNORECASE,
)
_COMPONENTES_HEADER_RE = re.compile(r"^Componentes identificados\s*\(\d+\)\s*$", re.IGNORECASE)
_AMEACAS_HEADER_RE = re.compile(
    r"^##\s*Ameacas STRIDE identificadas\s*\(\d+\)\s*$", re.IGNORECASE
)
_COMPONENTE_RE = re.compile(
    r"^#(?P<idx>\d+)\s+(?P<nome>[^-\n]+?)\s*(?:-\s*boundaries:\s*(?P<bounds>.+))?$"
)
_CATEGORIA_RE = re.compile(r"^(?P<cat>[A-Za-z][A-Za-z ]+?)\s*\((?P<n>\d+)\)\s*$")
_AMEACA_RE = re.compile(
    r"^(?P<comp>[a-zA-Z0-9_\-]+)\s*\(component#(?P<idx>\d+)\)\s*:\s*(?P<desc>.+)$"
)
_CONTRAMEDIDA_RE = re.compile(r"^Contramedida\s*:\s*(?P<texto>.+)$", re.IGNORECASE)


class StrideParserError(ValueError):
    """Erro elevado quando o texto nao segue o formato esperado."""


def parse(texto: str) -> RelatorioTecnico:
    """Converte o relatorio STRIDE bruto em :class:`RelatorioTecnico`.

    Args:
        texto: Saida crua do modelo STRIDE.

    Returns:
        Relatorio tipado com componentes e ameacas.

    Raises:
        StrideParserError: Se o cabecalho principal nao for encontrado.
    """
    linhas = [linha.rstrip() for linha in texto.splitlines()]

    arquitetura = _extrair_arquitetura(linhas)
    componentes = _extrair_componentes(linhas)
    ameacas = _extrair_ameacas(linhas)

    return RelatorioTecnico(
        arquitetura=arquitetura,
        componentes=componentes,
        ameacas=ameacas,
    )


def _extrair_arquitetura(linhas: list[str]) -> str:
    for linha in linhas:
        match = _HEADER_RE.match(linha.strip())
        if match:
            return match.group("arq").strip()
    raise StrideParserError(
        "Cabecalho '##Relatorio de Modelagem de Ameacas (STRIDE) - <arquitetura>' nao encontrado."
    )


def _extrair_componentes(linhas: list[str]) -> list[Componente]:
    componentes: list[Componente] = []
    dentro_da_secao = False

    for linha in linhas:
        stripped = linha.strip()

        if _COMPONENTES_HEADER_RE.match(stripped):
            dentro_da_secao = True
            continue

        if not dentro_da_secao:
            continue

        if _AMEACAS_HEADER_RE.match(stripped):
            break

        if not stripped:
            continue

        match = _COMPONENTE_RE.match(stripped)
        if not match:
            continue

        bounds_raw = match.group("bounds") or ""
        boundaries = [b.strip() for b in bounds_raw.split(",") if b.strip()]

        componentes.append(
            Componente(
                idx=int(match.group("idx")),
                nome=match.group("nome").strip(),
                boundaries=boundaries,
            )
        )

    return componentes


def _extrair_ameacas(linhas: list[str]) -> list[AmeacaTecnica]:
    ameacas: list[AmeacaTecnica] = []
    dentro_da_secao = False
    categoria_atual: CategoriaSTRIDE | None = None
    ameaca_pendente: dict[str, str] | None = None

    def flush_pendente() -> None:
        nonlocal ameaca_pendente
        if ameaca_pendente is None or categoria_atual is None:
            return
        ameacas.append(
            AmeacaTecnica(
                categoria=categoria_atual,
                componente_ref=ameaca_pendente["componente_ref"],
                descricao=ameaca_pendente["descricao"],
                contramedida=ameaca_pendente.get("contramedida"),
            )
        )
        ameaca_pendente = None

    for linha in linhas:
        stripped = linha.strip()

        if _AMEACAS_HEADER_RE.match(stripped):
            dentro_da_secao = True
            continue

        if not dentro_da_secao or not stripped:
            continue

        cat_match = _CATEGORIA_RE.match(stripped)
        if cat_match and cat_match.group("cat").strip() in _CATEGORIAS_VALIDAS:
            flush_pendente()
            categoria_atual = cat_match.group("cat").strip()
            continue

        ameaca_match = _AMEACA_RE.match(stripped)
        if ameaca_match and categoria_atual is not None:
            flush_pendente()
            ameaca_pendente = {
                "componente_ref": (
                    f"{ameaca_match.group('comp')} (component#{ameaca_match.group('idx')})"
                ),
                "descricao": ameaca_match.group("desc").strip(),
            }
            continue

        contramedida_match = _CONTRAMEDIDA_RE.match(stripped)
        if contramedida_match and ameaca_pendente is not None:
            ameaca_pendente["contramedida"] = contramedida_match.group("texto").strip()
            flush_pendente()
            continue

    flush_pendente()
    return ameacas
