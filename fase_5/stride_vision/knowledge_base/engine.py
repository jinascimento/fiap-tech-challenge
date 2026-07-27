"""Aplica a base de conhecimento STRIDE (stride_rules.yaml) sobre o grafo
reconstruido (stride_vision.graph.reconstruct.build_graph) e gera a lista de
ameacas: uma por (componente, categoria STRIDE aplicavel) e uma por (fluxo que
cruza boundary, categoria STRIDE da regra de cruzamento).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import yaml


@dataclass(frozen=True)
class Threat:
    scope: str  # "component" ou "flow"
    target_id: int
    target_label: str
    category: str
    description: str
    mitigation: str


def load_rules(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def generate_threats(graph: nx.DiGraph, rules: dict) -> list[Threat]:
    threats: list[Threat] = []
    comp_rules = rules.get("components", {})

    for node_id, data in graph.nodes(data=True):
        cls_name = data["cls_name"]
        rule = comp_rules.get(cls_name)
        if not rule:
            continue
        for t in rule.get("threats", []):
            threats.append(
                Threat(
                    scope="component",
                    target_id=node_id,
                    target_label=cls_name,
                    category=t["category"],
                    description=t["description"],
                    mitigation=t["mitigation"],
                )
            )

    crossing_rule = rules.get("boundary_crossing_rule")
    if crossing_rule:
        for u, v, data in graph.edges(data=True):
            crosses = data.get("crosses_boundaries")
            if not crosses:
                continue
            src_label = graph.nodes[u]["cls_name"]
            dst_label = graph.nodes[v]["cls_name"]
            for t in crossing_rule["threats"]:
                threats.append(
                    Threat(
                        scope="flow",
                        target_id=v,
                        target_label=dst_label,
                        category=t["category"],
                        description=(
                            f"[Fluxo {src_label} (#{u}) -> {dst_label} (#{v})] "
                            f"{t['description']} (cruza: {', '.join(crosses)})"
                        ),
                        mitigation=t["mitigation"],
                    )
                )

    return threats
