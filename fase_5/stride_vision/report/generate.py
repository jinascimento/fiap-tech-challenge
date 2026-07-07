"""Gera o Relatorio de Modelagem de Ameacas (STRIDE) em Markdown a partir do
grafo reconstruido e da lista de ameacas geradas pelo knowledge_base.engine.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from ..knowledge_base.engine import Threat


def build_markdown_report(image_name: str, graph: nx.DiGraph, threats: list[Threat]) -> str:
    lines: list[str] = [f"# Relatorio de Modelagem de Ameacas (STRIDE) - {image_name}", ""]

    lines.append(f"## Componentes identificados ({graph.number_of_nodes()})")
    for node_id, data in sorted(graph.nodes(data=True)):
        boundaries = ", ".join(data["boundaries"]) or "(nenhuma)"
        lines.append(f"- `#{node_id}` **{data['cls_name']}** - boundaries: {boundaries}")
    lines.append("")

    skipped = graph.graph.get("flows_skipped_no_direction", 0)
    lines.append(f"## Fluxos reconstruidos ({graph.number_of_edges()})")
    if skipped:
        lines.append(f"_({skipped} seta(s) detectada(s) sem keypoints de direcao suficientes - ignoradas na reconstrucao do grafo)_")
    for u, v, data in graph.edges(data=True):
        crossing = ", ".join(data["crosses_boundaries"]) or "nenhum cruzamento"
        lines.append(
            f"- `#{u}` ({graph.nodes[u]['cls_name']}) -> `#{v}` ({graph.nodes[v]['cls_name']}) - cruza: {crossing}"
        )
    lines.append("")

    lines.append(f"## Ameacas STRIDE identificadas ({len(threats)})")
    by_category: dict[str, list[Threat]] = {}
    for t in threats:
        by_category.setdefault(t.category, []).append(t)
    for category in sorted(by_category):
        items = by_category[category]
        lines.append(f"### {category} ({len(items)})")
        for t in items:
            lines.append(f"- **{t.target_label}** (`{t.scope}#{t.target_id}`): {t.description}")
            lines.append(f"  - Contramedida: {t.mitigation}")
        lines.append("")

    return "\n".join(lines)


def save_report(content: str, path: str | Path) -> None:
    Path(path).write_text(content, encoding="utf-8")
