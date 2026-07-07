"""Pipeline ponta a ponta: labels YOLO (components + flows) -> grafo -> ameacas
STRIDE -> relatorio Markdown.

Hoje aceita os arquivos .txt de label (ground-truth) como entrada, o que
permite testar as etapas de grafo/regras/relatorio sem depender de um modelo
treinado. Quando os modelos (stride_vision.training.train_components /
train_flows) estiverem prontos, basta gerar os mesmos dois arquivos .txt a
partir de `model.predict(...)` (formato YOLO / YOLO-pose) e reusar este
mesmo pipeline.

Uso:
    python -m stride_vision.pipeline \\
        --components-label dataset/stride-architecture-components-v1/train/labels/X.txt \\
        --flows-label dataset/stride-architecture-flows-v1/train/labels/X.txt \\
        --image-name X --output report.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .graph.reconstruct import build_graph, parse_components_label, parse_flows_label
from .knowledge_base.engine import generate_threats, load_rules
from .report.generate import build_markdown_report, save_report

DEFAULT_RULES = Path(__file__).parent / "knowledge_base" / "stride_rules.yaml"


def run(
    components_label: str,
    flows_label: str | None,
    image_name: str,
    output: str,
    rules_path: str = str(DEFAULT_RULES),
) -> str:
    components = parse_components_label(components_label)
    flows = parse_flows_label(flows_label) if flows_label else []
    graph = build_graph(components, flows)
    rules = load_rules(rules_path)
    threats = generate_threats(graph, rules)
    report = build_markdown_report(image_name, graph, threats)
    save_report(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components-label", required=True)
    parser.add_argument("--flows-label")
    parser.add_argument("--image-name", default="diagrama")
    parser.add_argument("--output", default="report.md")
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    args = parser.parse_args()

    run(args.components_label, args.flows_label, args.image_name, args.output, args.rules)
    print(f"Relatorio salvo em {args.output}")


if __name__ == "__main__":
    main()
