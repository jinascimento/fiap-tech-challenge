"""Utilitario para contornar uma peculiaridade do Ultralytics: quando
`data.yaml` tem `path: .`, o `check_det_dataset` resolve esse "." contra o
cwd (porque `Path(".").exists()` e sempre verdadeiro), e nao contra o
diretorio do proprio arquivo yaml. Isso quebra o treino sempre que o comando
nao e rodado com cwd == diretorio do dataset.

`resolve_data_yaml` le o yaml original e escreve uma copia temporaria com
`path` apontando para o caminho absoluto do diretorio do dataset, sem
modificar o arquivo original (que pertence ao repositorio do dataset,
versionado separadamente via git-lfs).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import yaml


def resolve_data_yaml(data_yaml_path: str | Path) -> str:
    src = Path(data_yaml_path).resolve()
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    data["path"] = str(src.parent)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{src.stem}.yaml", delete=False, encoding="utf-8"
    )
    yaml.safe_dump(data, tmp)
    tmp.close()
    return tmp.name
