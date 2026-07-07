"""Treina o modelo de deteccao de fluxos (YOLOv8-pose, 1 classe + 2 keypoints
tail/tip) sobre dataset/stride-architecture-flows-v1.

Uso:
    python -m stride_vision.training.train_flows --epochs 100 --imgsz 960
"""
from __future__ import annotations

import argparse

from ..data.yaml_utils import resolve_data_yaml


def main() -> None:
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/stride-architecture-flows-v1/data.yaml")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="checkpoint base (yolov8n/s/m/l/x-pose.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps", help="mps (Apple GPU) / cpu / 0 (cuda:0)")
    parser.add_argument("--patience", type=int, default=100, help="epocas sem melhora antes de early stopping")
    parser.add_argument("--project", default="runs/flows")
    parser.add_argument("--name", default="yolov8n_flows")
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
