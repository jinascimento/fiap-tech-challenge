"""Treina o modelo de deteccao de componentes (YOLOv8, 32 classes) sobre
dataset/stride-architecture-components-v1.

Uso:
    python -m stride_vision.training.train_components --epochs 100 --imgsz 960

Antes de treinar: se for usar o split re-hidratado (ver
stride_vision.data.resplit), aponte --data para o data.yaml do dataset
re-splitado em vez do original.
"""
from __future__ import annotations

import argparse

from ..data.yaml_utils import resolve_data_yaml


def main() -> None:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/stride-architecture-components-v1/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="checkpoint base (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps", help="mps (Apple GPU) / cpu / 0 (cuda:0)")
    parser.add_argument("--patience", type=int, default=100, help="epocas sem melhora antes de early stopping")
    parser.add_argument("--project", default="runs/components")
    parser.add_argument("--name", default="yolov8n_components")
    args = parser.parse_args()

    data_path = resolve_data_yaml(args.data)

    model = YOLO(args.model)
    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
