"""Wrapper para o modelo STRIDE (deteccao YOLO + grafo + regras STRIDE).

Chama ``stride_vision.infer.run`` usando os pesos treinados versionados em
``models/`` (``components_yolov8n.pt``, ``flows_yolov8n_pose.pt``) e a base de
regras em ``stride_vision/knowledge_base/stride_rules.yaml``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from stride_vision.infer import (
    DEFAULT_COMPONENTS_WEIGHTS,
    DEFAULT_FLOWS_WEIGHTS,
    DEFAULT_RULES,
)
from stride_vision.infer import run as _run_inference


class StrideRunnerError(RuntimeError):
    """Erro elevado quando o modelo STRIDE falha ao analisar o diagrama."""


def analisar_diagrama(image_path: Path, image_name: str = "diagrama") -> str:
    """Analisa um diagrama de arquitetura e retorna o relatorio STRIDE bruto.

    Args:
        image_path: Caminho para a imagem do diagrama (PNG/JPG).
        image_name: Nome da arquitetura, usado no titulo do relatorio.

    Returns:
        Texto do relatorio STRIDE, no formato emitido por
        ``stride_vision.report.generate.build_markdown_report``.

    Raises:
        FileNotFoundError: Se a imagem nao existir.
        StrideRunnerError: Se os pesos treinados nao existirem ou o modelo falhar.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem nao encontrada: {image_path}")

    if not DEFAULT_COMPONENTS_WEIGHTS.exists() or not DEFAULT_FLOWS_WEIGHTS.exists():
        raise StrideRunnerError(
            "Pesos treinados nao encontrados em models/ "
            f"({DEFAULT_COMPONENTS_WEIGHTS.name}, {DEFAULT_FLOWS_WEIGHTS.name})."
        )

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        output_path = Path(tmp.name)
    try:
        return _run_inference(
            str(image_path),
            str(DEFAULT_COMPONENTS_WEIGHTS),
            str(DEFAULT_FLOWS_WEIGHTS),
            image_name,
            str(output_path),
            str(DEFAULT_RULES),
        )
    except Exception as exc:
        raise StrideRunnerError(f"Falha ao analisar diagrama: {exc}") from exc
    finally:
        output_path.unlink(missing_ok=True)
