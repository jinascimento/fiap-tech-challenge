"""EDA dos datasets STRIDE (components + flows) - Hackathon Fase 5.

Roda com: uv run --no-project --with numpy,pillow,matplotlib analyze_datasets.py
Gera estatisticas (stdout + eda/outputs/stats.json) e graficos/amostras em eda/outputs/.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent / "dataset"
COMP = ROOT / "stride-architecture-components-v1"
FLOW = ROOT / "stride-architecture-flows-v1"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
AUG_SUFFIXES = [
    "BW", "sharp", "contrast", "gamma_hi", "gamma_lo",
    "jpeg50", "blur1", "noise6", "degrade80",
]
AUG_RE = re.compile(r"_(" + "|".join(AUG_SUFFIXES) + r")$")
UUID_PREFIX_RE = re.compile(r"^[0-9a-f]{8}-")
MIN_REAL_FILE_SIZE = 1024  # below this -> still a git-lfs pointer stub

COMP_NAMES = {
    0: "actor_user", 1: "actor_admin", 2: "edge_ddos_protection", 3: "edge_cdn",
    4: "edge_waf", 5: "edge_gateway", 6: "edge_portal", 7: "external_entry_point",
    8: "integration_orchestrator", 9: "integration_messaging", 10: "compute_load_balancer",
    11: "compute_service", 12: "compute_worker", 13: "data_database", 14: "data_cache",
    15: "data_storage", 16: "security_identity_provider", 17: "security_key_management",
    18: "obs_monitoring", 19: "obs_audit", 20: "external_backend_service",
    21: "external_saas_service", 22: "external_web_service", 23: "communication_service",
    24: "backup_service", 25: "boundary_cloud", 26: "boundary_region",
    27: "boundary_resource_group", 28: "boundary_vpc_or_vnet", 29: "boundary_subnet_public",
    30: "boundary_subnet_private", 31: "boundary_autoscaling_group",
}


def base_name(fname: str) -> str:
    """Strip the random uuid prefix and the augmentation suffix to get the source diagram id."""
    stem = Path(fname).stem
    stem = UUID_PREFIX_RE.sub("", stem)
    stem = AUG_RE.sub("", stem)
    return stem


def is_real_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > MIN_REAL_FILE_SIZE


# ---------------------------------------------------------------------------
# Components dataset (object detection, 32 classes)
# ---------------------------------------------------------------------------

def analyze_components_labels() -> dict:
    print("\n=== COMPONENTS: labels ===")
    result = {"per_split": {}}
    base_by_split: dict[str, set[str]] = {}
    class_counts_by_split: dict[str, Counter] = {}
    box_sizes = []  # (w, h) normalized, across all splits
    boxes_per_image = []

    for split in SPLITS:
        label_files = sorted((COMP / split / "labels").glob("*.txt"))
        bases = set()
        counts = Counter()
        for lf in label_files:
            bases.add(base_name(lf.name))
            n_boxes = 0
            for line in lf.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls = int(parts[0])
                w, h = float(parts[3]), float(parts[4])
                counts[cls] += 1
                box_sizes.append((w, h))
                n_boxes += 1
            boxes_per_image.append(n_boxes)
        base_by_split[split] = bases
        class_counts_by_split[split] = counts
        missing = [COMP_NAMES[i] for i in range(32) if counts[i] == 0]
        total_boxes = sum(counts.values())
        print(f"{split}: {len(label_files)} label files, {len(bases)} diagramas-base unicos, "
              f"{total_boxes} boxes, {len(counts)}/32 classes presentes")
        if missing:
            print(f"  classes ausentes: {missing}")
        result["per_split"][split] = {
            "label_files": len(label_files),
            "unique_base_images": len(bases),
            "total_boxes": total_boxes,
            "classes_present": len(counts),
            "classes_missing": missing,
            "class_counts": {COMP_NAMES[i]: counts[i] for i in range(32)},
        }

    # leakage between splits
    leak = {
        "train_val": sorted(base_by_split["train"] & base_by_split["val"]),
        "train_test": sorted(base_by_split["train"] & base_by_split["test"]),
        "val_test": sorted(base_by_split["val"] & base_by_split["test"]),
    }
    print("Vazamento entre splits (mesmo diagrama-base):",
          {k: len(v) for k, v in leak.items()})
    result["split_leakage"] = {k: len(v) for k, v in leak.items()}
    result["all_unique_base_images"] = len(set().union(*base_by_split.values()))

    # bbox geometry chart
    box_sizes_arr = np.array(box_sizes)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(box_sizes_arr[:, 0] * box_sizes_arr[:, 1], bins=50, color="#3b6ea5")
    axes[0].set_title("Area normalizada das bounding boxes (components)")
    axes[0].set_xlabel("largura * altura (normalizado)")
    axes[1].hist(boxes_per_image, bins=range(0, max(boxes_per_image) + 2), color="#5a9367")
    axes[1].set_title("Boxes por imagem (components)")
    axes[1].set_xlabel("numero de boxes")
    fig.tight_layout()
    fig.savefig(OUT / "components_bbox_geometry.png", dpi=130)
    plt.close(fig)

    # class distribution chart (train vs val vs test)
    fig, ax = plt.subplots(figsize=(14, 7))
    idx = np.arange(32)
    width = 0.27
    for i, split in enumerate(SPLITS):
        counts = class_counts_by_split[split]
        values = [counts[c] for c in range(32)]
        ax.bar(idx + (i - 1) * width, values, width, label=split)
    ax.set_xticks(idx)
    ax.set_xticklabels([COMP_NAMES[i] for i in range(32)], rotation=90)
    ax.set_ylabel("numero de boxes")
    ax.set_title("Distribuicao de classes por split - components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "components_class_distribution.png", dpi=130)
    plt.close(fig)

    result["base_by_split"] = {k: sorted(v) for k, v in base_by_split.items()}
    return result


# ---------------------------------------------------------------------------
# Flows dataset (YOLO-pose, 1 class, 2 keypoints: tail/tip)
# ---------------------------------------------------------------------------

def analyze_flows_labels() -> dict:
    print("\n=== FLOWS: labels ===")
    result = {"per_split": {}}
    base_by_split: dict[str, set[str]] = {}
    arrows_per_image = []

    for split in SPLITS:
        label_files = sorted((FLOW / split / "labels").glob("*.txt"))
        bases = set()
        total = both_visible = one_missing = both_missing = 0
        for lf in label_files:
            bases.add(base_name(lf.name))
            n_arrows = 0
            for line in lf.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                total += 1
                n_arrows += 1
                v1, v2 = int(parts[7]), int(parts[10])
                if v1 and v2:
                    both_visible += 1
                elif v1 or v2:
                    one_missing += 1
                else:
                    both_missing += 1
            arrows_per_image.append(n_arrows)
        base_by_split[split] = bases
        print(f"{split}: {len(label_files)} label files, {len(bases)} diagramas-base unicos, "
              f"{total} setas (both_visible={both_visible}, one_missing={one_missing}, "
              f"both_missing={both_missing})")
        result["per_split"][split] = {
            "label_files": len(label_files),
            "unique_base_images": len(bases),
            "total_arrows": total,
            "both_keypoints_visible": both_visible,
            "one_keypoint_missing": one_missing,
            "both_keypoints_missing": both_missing,
        }

    leak = {
        "train_val": len(base_by_split["train"] & base_by_split["val"]),
        "train_test": len(base_by_split["train"] & base_by_split["test"]),
        "val_test": len(base_by_split["val"] & base_by_split["test"]),
    }
    print("Vazamento entre splits (flows):", leak)
    result["split_leakage"] = leak
    result["all_unique_base_images"] = len(set().union(*base_by_split.values()))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(arrows_per_image, bins=range(0, max(arrows_per_image) + 2), color="#a35b3b")
    ax.set_title("Setas (flow_arrow) por imagem")
    ax.set_xlabel("numero de setas")
    fig.tight_layout()
    fig.savefig(OUT / "flows_arrows_per_image.png", dpi=130)
    plt.close(fig)

    result["base_by_split"] = {k: sorted(v) for k, v in base_by_split.items()}
    return result


# ---------------------------------------------------------------------------
# Cross-dataset compatibility
# ---------------------------------------------------------------------------

def cross_dataset_compatibility(comp_result: dict, flow_result: dict) -> dict:
    print("\n=== Compatibilidade entre datasets ===")
    comp_all: set[str] = set().union(*[set(v) for v in comp_result["base_by_split"].values()])
    flow_all: set[str] = set().union(*[set(v) for v in flow_result["base_by_split"].values()])
    inter = comp_all & flow_all

    comp_split_of = {b: s for s, bs in comp_result["base_by_split"].items() for b in bs}
    flow_split_of = {b: s for s, bs in flow_result["base_by_split"].items() for b in bs}
    mismatches = [b for b in inter if comp_split_of[b] != flow_split_of[b]]

    print(f"components: {len(comp_all)} diagramas-base | flows: {len(flow_all)} diagramas-base")
    print(f"interseccao (tem componentes E fluxos): {len(inter)}")
    print(f"somente em components: {len(comp_all - flow_all)} | somente em flows: {len(flow_all - comp_all)}")
    print(f"diagramas com split divergente entre os dois datasets: {len(mismatches)} -> {mismatches}")

    return {
        "components_unique_base_images": len(comp_all),
        "flows_unique_base_images": len(flow_all),
        "intersection": len(inter),
        "only_in_components": len(comp_all - flow_all),
        "only_in_flows": len(flow_all - comp_all),
        "split_mismatches": mismatches,
    }


# ---------------------------------------------------------------------------
# .npy vs .jpg/.png redundancy check
# ---------------------------------------------------------------------------

def npy_redundancy_check(n_samples: int = 5) -> dict:
    print("\n=== Redundancia .npy vs imagem ===")
    img_dir = COMP / "train" / "images"
    npy_files = [p for p in img_dir.glob("*.npy") if is_real_file(p)]
    diffs = []
    checked = 0
    for npy_path in npy_files:
        stem = npy_path.stem
        candidates = list(img_dir.glob(stem + ".*"))
        img_path = next((c for c in candidates if c.suffix.lower() in (".jpg", ".jpeg", ".png") and is_real_file(c)), None)
        if img_path is None:
            continue
        arr_npy = np.load(npy_path, allow_pickle=True)
        arr_img = np.array(Image.open(img_path).convert("RGB"))
        if arr_npy.shape != arr_img.shape:
            continue
        diff = np.abs(arr_npy.astype(int) - arr_img.astype(int))
        diffs.append({
            "file": stem,
            "mean_abs_diff": float(diff.mean()),
            "pct_pixels_diff_gt_10": float((diff.max(axis=-1) > 10).mean() * 100),
        })
        checked += 1
        if checked >= n_samples:
            break
    for d in diffs:
        print(f"  {d['file']}: mean_abs_diff={d['mean_abs_diff']:.2f}, "
              f"%pixels diff>10 = {d['pct_pixels_diff_gt_10']:.1f}%")
    return {"samples_checked": checked, "samples": diffs}


# ---------------------------------------------------------------------------
# Image resolution stats (requires real downloaded images)
# ---------------------------------------------------------------------------

def image_resolution_stats() -> dict:
    print("\n=== Resolucao das imagens (components, train) ===")
    img_dir = COMP / "train" / "images"
    sizes = []
    for p in img_dir.glob("*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or not is_real_file(p):
            continue
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            continue
    if not sizes:
        print("  nenhuma imagem real encontrada (LFS ainda nao baixado?)")
        return {"count": 0}
    arr = np.array(sizes)
    print(f"  {len(sizes)} imagens | largura: min={arr[:,0].min()} max={arr[:,0].max()} "
          f"media={arr[:,0].mean():.0f} | altura: min={arr[:,1].min()} max={arr[:,1].max()} "
          f"media={arr[:,1].mean():.0f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(arr[:, 0], arr[:, 1], alpha=0.4, s=10)
    ax.set_xlabel("largura (px)")
    ax.set_ylabel("altura (px)")
    ax.set_title("Resolucao das imagens - components/train")
    fig.tight_layout()
    fig.savefig(OUT / "components_image_resolution.png", dpi=130)
    plt.close(fig)

    return {
        "count": len(sizes),
        "width_min": int(arr[:, 0].min()), "width_max": int(arr[:, 0].max()), "width_mean": float(arr[:, 0].mean()),
        "height_min": int(arr[:, 1].min()), "height_max": int(arr[:, 1].max()), "height_mean": float(arr[:, 1].mean()),
    }


# ---------------------------------------------------------------------------
# Visual sanity-check: draw boxes/keypoints on a couple of sample images
# ---------------------------------------------------------------------------

def draw_components_sample(n: int = 3) -> None:
    img_dir = COMP / "train" / "images"
    lbl_dir = COMP / "train" / "labels"
    label_files = sorted(lbl_dir.glob("*.txt"))
    drawn = 0
    for lf in label_files:
        stem = lf.stem
        candidates = list(img_dir.glob(stem + ".*"))
        img_path = next((c for c in candidates if c.suffix.lower() in (".jpg", ".jpeg", ".png") and is_real_file(c)), None)
        if img_path is None:
            continue
        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        W, H = im.size
        for line in lf.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:5])
            x0, y0 = (xc - w / 2) * W, (yc - h / 2) * H
            x1, y1 = (xc + w / 2) * W, (yc + h / 2) * H
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            draw.text((x0 + 2, max(0, y0 - 12)), COMP_NAMES[cls], fill="red")
        im.save(OUT / f"sample_components_{stem}.png")
        drawn += 1
        if drawn >= n:
            break
    print(f"\nAmostras de components com boxes desenhadas: {drawn} salvas em {OUT}")


def draw_flows_sample(n: int = 3) -> None:
    img_dir = FLOW / "train" / "images"
    lbl_dir = FLOW / "train" / "labels"
    label_files = sorted(lbl_dir.glob("*.txt"))
    drawn = 0
    for lf in label_files:
        stem = lf.stem
        candidates = list(img_dir.glob(stem + ".*"))
        img_path = next((c for c in candidates if c.suffix.lower() in (".jpg", ".jpeg", ".png") and is_real_file(c)), None)
        if img_path is None:
            continue
        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        W, H = im.size
        for line in lf.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            xc, yc, w, h = (float(v) for v in parts[1:5])
            x0, y0 = (xc - w / 2) * W, (yc - h / 2) * H
            x1, y1 = (xc + w / 2) * W, (yc + h / 2) * H
            draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
            tail_x, tail_y, tail_v = float(parts[5]) * W, float(parts[6]) * H, int(parts[7])
            tip_x, tip_y, tip_v = float(parts[8]) * W, float(parts[9]) * H, int(parts[10])
            if tail_v:
                draw.ellipse([tail_x - 4, tail_y - 4, tail_x + 4, tail_y + 4], fill="green")
            if tip_v:
                draw.ellipse([tip_x - 4, tip_y - 4, tip_x + 4, tip_y + 4], fill="orange")
            if tail_v and tip_v:
                draw.line([tail_x, tail_y, tip_x, tip_y], fill="purple", width=1)
        im.save(OUT / f"sample_flows_{stem}.png")
        drawn += 1
        if drawn >= n:
            break
    print(f"Amostras de flows com keypoints desenhados: {drawn} salvas em {OUT}")


def main() -> None:
    comp_result = analyze_components_labels()
    flow_result = analyze_flows_labels()
    cross = cross_dataset_compatibility(comp_result, flow_result)
    npy_check = npy_redundancy_check()
    res_stats = image_resolution_stats()
    draw_components_sample()
    draw_flows_sample()

    stats = {
        "components": comp_result,
        "flows": flow_result,
        "cross_dataset": cross,
        "npy_redundancy": npy_check,
        "image_resolution": res_stats,
    }
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nEstatisticas completas salvas em {OUT / 'stats.json'}")


if __name__ == "__main__":
    main()
