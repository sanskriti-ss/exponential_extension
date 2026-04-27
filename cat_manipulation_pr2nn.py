"""
Cat manipulation experiment runner.

What this script does:
1) Samples 20 STL10 cat images (natural set).
2) Loads up to 20 synthetic cat images from Peng2023Robust outputs.
3) Creates a pure-random-noise image set of equal size.
4) Applies all combinations (power set) of these manipulations:
   - add_random_noise
   - low_pass_gaussian
   - high_pass_gaussian
   - rainbow_border
5) Computes PR + 2NN in the paper-consistent mathematical form:
   - PR from covariance eigenvalues:
       d_PR = (sum lambda)^2 / sum(lambda^2)
   - 2NN MLE:
       d_2NN = 1 / mean(log(r2 / r1))
6) Saves:
   - per-image outputs organized by dataset/combo
   - summary CSV with PR/2NN metrics

Run:
  python cat_manipulation_pr2nn.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import STL10
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm


STL10_LABELS = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]
CAT_LABEL_ID = STL10_LABELS.index("cat")

MANIPULATION_NAMES = [
    "add_random_noise",
    "low_pass_gaussian",
    "high_pass_gaussian",
    "rainbow_border",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--image-side", type=int, default=96)
    parser.add_argument("--feature-side", type=int, default=24)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--gaussian-kernel", type=int, default=9)
    parser.add_argument("--gaussian-sigma", type=float, default=2.0)
    parser.add_argument("--border-width", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./outputs/cat_manipulation_pr2nn")
    parser.add_argument(
        "--peng-dir",
        default="auto",
        help="Path to Peng2023Robust cat images directory, or 'auto' to detect latest.",
    )
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_stl10_cats(data_dir: Path, num_images: int, image_side: int, seed: int, no_download: bool) -> list[torch.Tensor]:
    dataset = STL10(root=str(data_dir), split="train", download=not no_download, transform=T.ToTensor())
    labels = np.asarray(dataset.labels)
    cat_indices = np.where(labels == CAT_LABEL_ID)[0]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(cat_indices, size=min(num_images, len(cat_indices)), replace=False)
    resize = T.Resize((image_side, image_side), antialias=True)
    out = []
    for idx in chosen:
        x, _ = dataset[int(idx)]
        out.append(resize(x).clamp(0.0, 1.0))
    return out


def auto_find_peng_dir(base_dir: Path) -> Path | None:
    runs_root = base_dir / "outputs" / "gradient_diffusion_experiments"
    if not runs_root.exists():
        return None
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    run_dirs.sort(reverse=True)
    for run in run_dirs:
        candidate = run / "models" / "Peng2023Robust" / "images" / "noise_to_cat" / "cat"
        if candidate.exists():
            return candidate
    return None


def load_peng_synthetic_images(peng_dir: Path, num_images: int, image_side: int) -> list[torch.Tensor]:
    files = sorted([p for p in peng_dir.glob("cat_final_*.png") if p.is_file()])
    files = [p for p in files if "_grid" not in p.name][:num_images]
    resize = T.Resize((image_side, image_side), antialias=True)
    out = []
    for fp in files:
        img = Image.open(fp).convert("RGB")
        x = T.ToTensor()(img)
        out.append(resize(x).clamp(0.0, 1.0))
    return out


def make_random_noise_set(num_images: int, image_side: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.rand((3, image_side, image_side), generator=gen) for _ in range(num_images)]


def apply_manipulations(
    images: list[torch.Tensor],
    combo: tuple[str, ...],
    *,
    noise_sigma: float,
    gaussian_kernel: int,
    gaussian_sigma: float,
    border_width: int,
    seed: int,
) -> list[torch.Tensor]:
    out = [img.clone() for img in images]
    rng = np.random.default_rng(seed)

    if "add_random_noise" in combo:
        for i, img in enumerate(out):
            eps = torch.tensor(rng.normal(0.0, noise_sigma, size=tuple(img.shape)), dtype=torch.float32)
            out[i] = (img + eps).clamp(0.0, 1.0)

    if "low_pass_gaussian" in combo:
        blur = T.GaussianBlur(kernel_size=gaussian_kernel, sigma=gaussian_sigma)
        out = [blur(img).clamp(0.0, 1.0) for img in out]

    if "high_pass_gaussian" in combo:
        blur = T.GaussianBlur(kernel_size=gaussian_kernel, sigma=gaussian_sigma)
        hp = []
        for img in out:
            low = blur(img)
            # high-pass visualization offset keeps values in displayable range
            hp_img = (img - low + 0.5).clamp(0.0, 1.0)
            hp.append(hp_img)
        out = hp

    if "rainbow_border" in combo:
        bw = max(1, border_width)
        for i, img in enumerate(out):
            h, w = img.shape[1], img.shape[2]
            x = torch.linspace(0, 1, w)
            y = torch.linspace(0, 1, h)
            # simple deterministic rainbow gradients
            top = torch.stack([x, torch.flip(x, dims=[0]), torch.ones_like(x)], dim=0)
            left = torch.stack([torch.ones_like(y), y, torch.flip(y, dims=[0])], dim=0)
            img[:, :bw, :] = top[:, None, :].repeat(1, bw, 1)
            img[:, -bw:, :] = torch.flip(top, dims=[1])[:, None, :].repeat(1, bw, 1)
            img[:, :, :bw] = left[:, :, None].repeat(1, 1, bw)
            img[:, :, -bw:] = torch.flip(left, dims=[1])[:, :, None].repeat(1, 1, bw)
            out[i] = img.clamp(0.0, 1.0)

    return out


def flatten_with_feature_side(images: list[torch.Tensor], feature_side: int) -> np.ndarray:
    resize = T.Resize((feature_side, feature_side), antialias=True)
    rows = []
    for img in images:
        small = resize(img).clamp(0.0, 1.0)
        rows.append(small.permute(1, 2, 0).reshape(-1).numpy())
    return np.stack(rows, axis=0).astype(np.float64)


def participation_ratio(X: np.ndarray) -> tuple[float, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / X.shape[0]
    eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    pr = float((eigvals.sum() ** 2) / (np.square(eigvals).sum() + 1e-12))
    return pr, eigvals


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X * X, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * (X @ X.T), 0.0)
    return np.sqrt(d2)


def two_nn_dimension(X: np.ndarray) -> float:
    if X.shape[0] < 3:
        return float("nan")
    D = pairwise_distances(X)
    np.fill_diagonal(D, np.inf)
    sorted_d = np.sort(D, axis=1)
    r1 = np.maximum(sorted_d[:, 0], 1e-12)
    r2 = np.maximum(sorted_d[:, 1], r1 + 1e-12)
    mu = r2 / r1
    return float(1.0 / (np.log(mu).mean() + 1e-12))


def combo_name(combo: tuple[str, ...]) -> str:
    return "none" if len(combo) == 0 else "__".join(combo)


def save_image_set(images: list[torch.Tensor], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, out_dir / f"{prefix}_{i:03d}.png")
    grid = make_grid(torch.stack(images), nrow=min(5, len(images)), padding=2)
    save_image(grid, out_dir / f"{prefix}_grid.png")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_root = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    natural = sample_stl10_cats(
        data_dir=Path(args.data_dir),
        num_images=args.num_images,
        image_side=args.image_side,
        seed=args.seed,
        no_download=args.no_download,
    )

    if args.peng_dir == "auto":
        peng_dir = auto_find_peng_dir(Path(".").resolve())
    else:
        peng_dir = Path(args.peng_dir)
    synthetic = load_peng_synthetic_images(peng_dir, args.num_images, args.image_side) if peng_dir and Path(peng_dir).exists() else []
    if len(synthetic) == 0:
        synthetic = [img.clone() for img in natural]
    noise = make_random_noise_set(min(len(natural), args.num_images), args.image_side, args.seed + 99)

    dataset_sets = {
        "natural_cat": natural,
        "synthetic_cat_peng2023robust": synthetic,
        "random_noise": noise,
    }

    all_combos = []
    for r in range(0, len(MANIPULATION_NAMES) + 1):
        all_combos.extend(itertools.combinations(MANIPULATION_NAMES, r))

    metrics_rows = []
    total = len(dataset_sets) * len(all_combos)
    progress = tqdm(total=total, desc="dataset x combos")

    for dataset_name, base_images in dataset_sets.items():
        for combo in all_combos:
            cname = combo_name(combo)
            mod_images = apply_manipulations(
                base_images,
                combo,
                noise_sigma=args.noise_sigma,
                gaussian_kernel=args.gaussian_kernel,
                gaussian_sigma=args.gaussian_sigma,
                border_width=args.border_width,
                seed=args.seed + len(cname) * 31,
            )
            X = flatten_with_feature_side(mod_images, args.feature_side)
            pr, eig = participation_ratio(X)
            d2 = two_nn_dimension(X)

            save_dir = out_root / "images" / dataset_name / cname
            save_image_set(mod_images, save_dir, "sample")

            metrics_rows.append(
                {
                    "dataset_name": dataset_name,
                    "combo_name": cname,
                    "num_images": len(mod_images),
                    "feature_side": args.feature_side,
                    "ambient_dim": int(X.shape[1]),
                    "pr": pr,
                    "two_nn_mle": d2,
                    "top_eigenvalue": float(eig[0]) if eig.size > 0 else float("nan"),
                    "sum_eigenvalues": float(eig.sum()) if eig.size > 0 else float("nan"),
                }
            )
            progress.update(1)
    progress.close()

    csv_path = out_root / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    config = vars(args).copy()
    config["resolved_peng_dir"] = str(peng_dir) if peng_dir else None
    (out_root / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Saved experiment to: {out_root}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

