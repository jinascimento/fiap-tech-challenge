"""Re-splita o dataset `components` por diagrama-base (nao por arquivo) para
eliminar o vazamento train/val/test identificado na EDA
(eda/RELATORIO_EDA.md, secao 4.2: ~4-5% dos diagramas-base aparecem em mais
de um split porque o split original foi feito por arquivo, incluindo
variacoes de augmentation do mesmo diagrama).

Por padrao roda em modo dry-run (so escreve o manifesto com o plano de
realocacao). Use --apply para de fato copiar os arquivos para --out -- os
dados originais em dataset/ nunca sao modificados ou movidos.

Uso:
    python -m stride_vision.data.resplit
    python -m stride_vision.data.resplit --apply --out dataset/stride-architecture-components-v1-resplit
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

UUID_PREFIX_RE = re.compile(r"^[0-9a-f]{8}-")
AUG_SUFFIXES = ["BW", "sharp", "contrast", "gamma_hi", "gamma_lo", "jpeg50", "blur1", "noise6", "degrade80"]
AUG_RE = re.compile(r"_(" + "|".join(AUG_SUFFIXES) + r")$")


def base_name(fname: str) -> str:
    """Remove o prefixo uuid aleatorio e o sufixo de augmentation para obter o
    id do diagrama de origem (mesma logica usada em eda/analyze_datasets.py)."""
    stem = Path(fname).stem
    stem = UUID_PREFIX_RE.sub("", stem)
    stem = AUG_RE.sub("", stem)
    return stem


def collect_files(dataset_root: Path) -> dict[str, list[Path]]:
    by_base: dict[str, list[Path]] = defaultdict(list)
    for split in ("train", "val", "test"):
        for lf in (dataset_root / split / "labels").glob("*.txt"):
            by_base[base_name(lf.name)].append(lf)
    return by_base


def assign_splits(bases: list[str], ratios: tuple[float, float, float] = (0.7, 0.2, 0.1), seed: int = 42) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = bases[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    assignment: dict[str, str] = {}
    for b in shuffled[:n_train]:
        assignment[b] = "train"
    for b in shuffled[n_train : n_train + n_val]:
        assignment[b] = "val"
    for b in shuffled[n_train + n_val :]:
        assignment[b] = "test"
    return assignment


def image_path_for_label(label_path: Path) -> Path | None:
    img_dir = label_path.parent.parent / "images"
    stem = label_path.stem
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="dataset/stride-architecture-components-v1")
    parser.add_argument("--out", default="dataset/stride-architecture-components-v1-resplit")
    parser.add_argument("--apply", action="store_true", help="Copia os arquivos para --out; sem essa flag so mostra o plano (dry-run).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset)
    by_base = collect_files(root)
    assignment = assign_splits(sorted(by_base), seed=args.seed)

    counts = Counter(assignment.values())
    print(
        f"{len(by_base)} diagramas-base -> "
        f"train={counts['train']} val={counts['val']} test={counts['test']}"
    )

    plan = []
    for base, split in assignment.items():
        for label_path in by_base[base]:
            img_path = image_path_for_label(label_path)
            plan.append(
                {
                    "base": base,
                    "split": split,
                    "label": str(label_path),
                    "image": str(img_path) if img_path else None,
                }
            )
    n_missing_image = sum(1 for e in plan if e["image"] is None)
    if n_missing_image:
        print(f"aviso: {n_missing_image} label(s) sem imagem correspondente encontrada (ignoradas na copia)")

    out_root = Path(args.out)
    if args.apply:
        for split in ("train", "val", "test"):
            (out_root / split / "images").mkdir(parents=True, exist_ok=True)
            (out_root / split / "labels").mkdir(parents=True, exist_ok=True)
        for entry in plan:
            split = entry["split"]
            label_src = Path(entry["label"])
            shutil.copy2(label_src, out_root / split / "labels" / label_src.name)
            if entry["image"]:
                img_src = Path(entry["image"])
                shutil.copy2(img_src, out_root / split / "images" / img_src.name)
        manifest_path = out_root / "manifest.json"
        manifest_path.write_text(json.dumps(plan, indent=2))
        print(f"Dataset re-splitado copiado para {out_root} (manifest: {manifest_path})")
    else:
        manifest_path = out_root.with_suffix(".manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(plan, indent=2))
        print(f"Dry-run: nenhum arquivo copiado. Plano salvo em {manifest_path}. Rode com --apply para copiar.")


if __name__ == "__main__":
    main()
