"""Reconstroi um grafo dirigido da arquitetura a partir das deteccoes dos dois
modelos (components + flows): nos = componentes, arestas = fluxos (com direcao
tail->tip), atributos = quais trust boundaries cada no esta dentro e quais
boundaries cada aresta cruza.

Funciona hoje com os labels ground-truth (formato YOLO/YOLO-pose) como um
stand-in para as deteccoes do modelo -- e o mesmo parser serve para a saida de
um `model.predict(...)` do Ultralytics assim que os modelos estiverem
treinados (basta montar os `Component`/`Flow` a partir das boxes preditas).
"""
from __future__ import annotations

import math
from pathlib import Path

import networkx as nx

from ..schema import BBox, Component, Flow, COMPONENT_CLASSES

# distancia maxima (normalizada, 0-1) entre um keypoint tail/tip e o centro do
# componente mais proximo para aceitar a associacao. Diagramas costumam ter
# setas curtas: 0.15 ~= 15% da largura/altura da imagem.
MAX_ENDPOINT_DISTANCE = 0.15


def parse_components_label(path: str | Path) -> list[Component]:
    components = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        parts = line.split()
        if not parts:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = (float(v) for v in parts[1:5])
        components.append(
            Component(id=i, cls_name=COMPONENT_CLASSES[cls_id], bbox=BBox(xc, yc, w, h))
        )
    return components


def parse_flows_label(path: str | Path) -> list[Flow]:
    flows = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        parts = line.split()
        if not parts:
            continue
        xc, yc, w, h = (float(v) for v in parts[1:5])
        tail = (float(parts[5]), float(parts[6])) if int(parts[7]) else None
        tip = (float(parts[8]), float(parts[9])) if int(parts[10]) else None
        flows.append(Flow(id=i, bbox=BBox(xc, yc, w, h), tail=tail, tip=tip))
    return flows


def containing_boundaries(components: list[Component], point: tuple[float, float]) -> list[Component]:
    """Boundaries cujo bbox contem o ponto, da mais externa (maior area) pra mais interna."""
    x, y = point
    hits = [c for c in components if c.is_boundary and c.bbox.contains(x, y)]
    return sorted(hits, key=lambda c: c.bbox.area, reverse=True)


def nearest_non_boundary(
    components: list[Component], point: tuple[float, float], max_dist: float = MAX_ENDPOINT_DISTANCE
) -> Component | None:
    x, y = point
    candidates = [c for c in components if not c.is_boundary]
    if not candidates:
        return None

    def dist(c: Component) -> float:
        return math.hypot(c.bbox.x_center - x, c.bbox.y_center - y)

    best = min(candidates, key=dist)
    return best if dist(best) <= max_dist else None


def build_graph(components: list[Component], flows: list[Flow]) -> nx.DiGraph:
    graph = nx.DiGraph()
    non_boundary = [c for c in components if not c.is_boundary]

    for c in non_boundary:
        boundaries = containing_boundaries(components, (c.bbox.x_center, c.bbox.y_center))
        graph.add_node(
            c.id,
            cls_name=c.cls_name,
            bbox=c.bbox,
            boundaries=[b.cls_name for b in boundaries],
        )

    skipped_no_direction = 0
    for f in flows:
        if f.tail is None or f.tip is None:
            # seta sem os dois keypoints visiveis: sabemos que existe uma
            # comunicacao ali perto, mas nao a direcao -- ver EDA secao 5.
            skipped_no_direction += 1
            continue
        src = nearest_non_boundary(components, f.tail)
        dst = nearest_non_boundary(components, f.tip)
        if src is None or dst is None or src.id == dst.id:
            continue
        src_boundaries = set(graph.nodes[src.id]["boundaries"])
        dst_boundaries = set(graph.nodes[dst.id]["boundaries"])
        crosses = sorted(src_boundaries.symmetric_difference(dst_boundaries))
        graph.add_edge(src.id, dst.id, flow_id=f.id, crosses_boundaries=crosses)

    graph.graph["flows_skipped_no_direction"] = skipped_no_direction
    return graph
