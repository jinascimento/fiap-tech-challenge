"""Roda os dois modelos treinados (components + flows) sobre uma imagem real
de diagrama e monta as mesmas estruturas (`Component`/`Flow`) usadas pelo
pipeline baseado em labels ground-truth (`stride_vision.pipeline`), permitindo
gerar o Relatorio STRIDE a partir de uma imagem nunca vista em treino.

Uso (pesos default ja apontam para models/, versionados no git -- nao precisa treinar):
    python -m stride_vision.infer \\
        --image caminho/para/diagrama.jpg \\
        --image-name "Arquitetura AWS" --output report_aws.md

Para usar pesos de um treino proprio (ex.: apos rodar train_components/train_flows):
    python -m stride_vision.infer \\
        --image caminho/para/diagrama.jpg \\
        --components-weights runs/detect/runs/components/yolov8n_components/weights/best.pt \\
        --flows-weights runs/pose/runs/flows/yolov8n_flows/weights/best.pt \\
        --image-name "Arquitetura AWS" --output report_aws.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .graph.reconstruct import build_graph
from .knowledge_base.engine import generate_threats, load_rules
from .report.generate import build_markdown_report, save_report
from .report.visualize import draw_annotations
from .schema import BBox, Component, Flow

DEFAULT_RULES = Path(__file__).parent / "knowledge_base" / "stride_rules.yaml"
MODELS_DIR = Path(__file__).parent.parent / "models"
DEFAULT_COMPONENTS_WEIGHTS = MODELS_DIR / "components_yolov8n.pt"
DEFAULT_FLOWS_WEIGHTS = MODELS_DIR / "flows_yolov8n_pose.pt"


def predict_components(model, image_path: str, conf: float = 0.25) -> list[Component]:
    result = model.predict(image_path, conf=conf, verbose=False)[0]
    components = []
    if result.boxes is None:
        return components
    xywhn = result.boxes.xywhn.tolist()
    classes = result.boxes.cls.tolist()
    confs = result.boxes.conf.tolist()
    names = result.names
    for i, (box, cls_id, c) in enumerate(zip(xywhn, classes, confs)):
        xc, yc, w, h = box
        components.append(
            Component(id=i, cls_name=names[int(cls_id)], bbox=BBox(xc, yc, w, h), confidence=c)
        )
    return components


def predict_flows(model, image_path: str, conf: float = 0.25, kpt_conf_thresh: float = 0.5) -> list[Flow]:
    result = model.predict(image_path, conf=conf, verbose=False)[0]
    flows = []
    if result.boxes is None or result.keypoints is None:
        return flows
    xywhn = result.boxes.xywhn.tolist()
    confs = result.boxes.conf.tolist()
    kpts_xy = result.keypoints.xyn.tolist()  # (N, 2, 2) -> [[tail_x, tail_y], [tip_x, tip_y]]
    kpts_conf = result.keypoints.conf.tolist() if result.keypoints.conf is not None else None
    for i, (box, c, kpt) in enumerate(zip(xywhn, confs, kpts_xy)):
        xc, yc, w, h = box
        tail_xy, tip_xy = kpt[0], kpt[1]
        if kpts_conf is not None:
            tail_v, tip_v = kpts_conf[i][0] >= kpt_conf_thresh, kpts_conf[i][1] >= kpt_conf_thresh
        else:
            tail_v = tip_v = True
        tail = tuple(tail_xy) if tail_v else None
        tip = tuple(tip_xy) if tip_v else None
        flows.append(Flow(id=i, bbox=BBox(xc, yc, w, h), tail=tail, tip=tip, confidence=c))
    return flows


def run(
    image_path: str,
    components_weights: str,
    flows_weights: str,
    image_name: str,
    output: str,
    rules_path: str = str(DEFAULT_RULES),
    conf: float = 0.25,
    viz_output: str | None = None,
) -> str:
    from ultralytics import YOLO

    components_model = YOLO(components_weights)
    flows_model = YOLO(flows_weights)

    components = predict_components(components_model, image_path, conf=conf)
    flows = predict_flows(flows_model, image_path, conf=conf)

    if viz_output:
        draw_annotations(image_path, components, flows, viz_output)

    graph = build_graph(components, flows)
    rules = load_rules(rules_path)
    threats = generate_threats(graph, rules)
    report = build_markdown_report(image_name, graph, threats)
    save_report(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--components-weights", default=str(DEFAULT_COMPONENTS_WEIGHTS))
    parser.add_argument("--flows-weights", default=str(DEFAULT_FLOWS_WEIGHTS))
    parser.add_argument("--image-name", default="diagrama")
    parser.add_argument("--output", default="report.md")
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    parser.add_argument("--conf", type=float, default=0.25, help="confianca minima de deteccao")
    parser.add_argument("--viz-output", default=None, help="se informado, salva a imagem anotada (bboxes + fluxos) nesse caminho")
    args = parser.parse_args()

    run(
        args.image,
        args.components_weights,
        args.flows_weights,
        args.image_name,
        args.output,
        args.rules,
        args.conf,
        args.viz_output,
    )
    print(f"Relatorio salvo em {args.output}")
    if args.viz_output:
        print(f"Imagem anotada salva em {args.viz_output}")


if __name__ == "__main__":
    main()
