from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import pdist


DEFAULT_MASKS = ("kidney_left", "kidney_right")


def _load_nifti(path: Path) -> nib.Nifti1Image:
    try:
        return nib.load(str(path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"NIfTI file not found: {path}") from exc


def _assert_same_geometry(ct_img: nib.Nifti1Image, mask_img: nib.Nifti1Image, mask_name: str) -> None:
    if ct_img.shape != mask_img.shape:
        raise ValueError(f"Geometry mismatch for {mask_name}: CT shape {ct_img.shape} vs mask shape {mask_img.shape}")
    if not np.allclose(ct_img.affine, mask_img.affine, atol=1e-4):
        raise ValueError(f"Affine mismatch for {mask_name}: CT and mask are not aligned")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _largest_axis_mm(coords_xyz: np.ndarray) -> float:
    if coords_xyz.shape[0] < 2:
        return 0.0
    return float(pdist(coords_xyz, metric="euclidean").max())


def _principal_axes_mm(coords_xyz: np.ndarray) -> list[float]:
    if coords_xyz.shape[0] < 2:
        return [0.0, 0.0, 0.0]
    centered = coords_xyz - coords_xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vh.T
    extents = proj.max(axis=0) - proj.min(axis=0)
    extents = np.sort(extents)[::-1]
    padded = np.zeros(3, dtype=np.float64)
    padded[: min(3, extents.shape[0])] = extents[:3]
    return [float(v) for v in padded]


def _crop_bounds(mask: np.ndarray, margin_xyz: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    mins = np.maximum(coords.min(axis=0) - np.asarray(margin_xyz), 0)
    maxs = np.minimum(coords.max(axis=0) + np.asarray(margin_xyz), np.asarray(mask.shape) - 1)
    return mins, maxs


def _render_component_overlays(
    ct: np.ndarray,
    kidney_mask: np.ndarray,
    component_mask: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    title_prefix: str,
    output_prefix: Path,
) -> dict[str, Any]:
    coords = np.argwhere(component_mask)
    centroid_ijk = np.round(coords.mean(axis=0)).astype(int)
    axial_slice = int(centroid_ijk[2])
    coronal_slice = int(centroid_ijk[1])
    x0, y0, z0 = mins
    x1, y1, z1 = maxs

    wl = 40.0
    ww = 400.0
    vmin = wl - ww / 2.0
    vmax = wl + ww / 2.0

    views = [
        (
            "axial",
            ct[:, :, axial_slice].T[y0 : y1 + 1, x0 : x1 + 1],
            kidney_mask[:, :, axial_slice].T[y0 : y1 + 1, x0 : x1 + 1],
            component_mask[:, :, axial_slice].T[y0 : y1 + 1, x0 : x1 + 1],
            f"{title_prefix} axial z={axial_slice}",
        ),
        (
            "coronal",
            ct[:, coronal_slice, :].T[z0 : z1 + 1, x0 : x1 + 1],
            kidney_mask[:, coronal_slice, :].T[z0 : z1 + 1, x0 : x1 + 1],
            component_mask[:, coronal_slice, :].T[z0 : z1 + 1, x0 : x1 + 1],
            f"{title_prefix} coronal y={coronal_slice}",
        ),
    ]

    paths: dict[str, Any] = {
        "axial_slice": axial_slice,
        "coronal_slice": coronal_slice,
    }
    for plane, ct_plane, kidney_plane, component_plane, title in views:
        out_path = output_prefix.parent / f"{output_prefix.name}_{plane}.png"
        fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
        ax.imshow(ct_plane, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
        ax.contour(kidney_plane.astype(float), levels=[0.5], colors=["deepskyblue"], linewidths=1.0)
        masked = np.ma.masked_where(~component_plane, component_plane)
        ax.imshow(masked, cmap="autumn", alpha=0.8, origin="lower", interpolation="none")
        ax.contour(component_plane.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.2)
        ax.set_title(title)
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        paths[f"{plane}_overlay_png"] = str(out_path)
    return paths


def analyze_kidneys(
    ct_path: Path,
    mask_dir: Path,
    threshold_hu: float = 130.0,
    masks: tuple[str, ...] = DEFAULT_MASKS,
    min_voxels: int = 3,
    min_volume_mm3: float | None = None,
    render_dir: Path | None = None,
) -> dict[str, Any]:
    ct_img = _load_nifti(ct_path)
    ct = ct_img.get_fdata(dtype=np.float32)
    voxel_spacing = tuple(float(v) for v in ct_img.header.get_zooms()[:3])
    voxel_volume_mm3 = float(np.prod(voxel_spacing))
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)

    if render_dir is None:
        render_dir = mask_dir / "kidney_stone_renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    kidneys: list[dict[str, Any]] = []
    missing_masks: list[str] = []
    total_components = 0
    total_stone_volume_mm3 = 0.0

    for mask_name in masks:
        mask_path = mask_dir / f"{mask_name}.nii.gz"
        if not mask_path.exists():
            missing_masks.append(mask_name)
            continue

        mask_img = _load_nifti(mask_path)
        _assert_same_geometry(ct_img, mask_img, mask_name)
        kidney_mask = mask_img.get_fdata() > 0.5
        kidney_values = ct[kidney_mask]
        dense_mask = kidney_mask & (ct > threshold_hu)
        labels, num_components = ndimage.label(dense_mask, structure=structure)
        mins, maxs = _crop_bounds(kidney_mask, margin_xyz=(20, 20, 10))

        components: list[dict[str, Any]] = []
        for label in range(1, num_components + 1):
            component_mask = labels == label
            voxel_count = int(component_mask.sum())
            if voxel_count == 0:
                continue
            volume_mm3 = voxel_count * voxel_volume_mm3
            passes_voxel_filter = voxel_count >= min_voxels
            passes_volume_filter = min_volume_mm3 is not None and volume_mm3 >= min_volume_mm3
            if not (passes_voxel_filter or passes_volume_filter):
                continue
            component_values = ct[component_mask]
            coords_ijk = np.argwhere(component_mask)
            coords_xyz = nib.affines.apply_affine(ct_img.affine, coords_ijk)
            centroid_ijk = coords_ijk.mean(axis=0)
            centroid_xyz = coords_xyz.mean(axis=0)
            component_id = f"{mask_name}_component_{label}"
            overlay_paths = _render_component_overlays(
                ct=ct,
                kidney_mask=kidney_mask,
                component_mask=component_mask,
                mins=mins,
                maxs=maxs,
                title_prefix=component_id,
                output_prefix=render_dir / component_id,
            )
            component = {
                "component_id": component_id,
                "label": label,
                "voxel_count": voxel_count,
                "volume_mm3": volume_mm3,
                "volume_ml": volume_mm3 / 1000.0,
                "hu_mean": float(component_values.mean()),
                "hu_max": float(component_values.max()),
                "centroid_ijk": [float(v) for v in centroid_ijk],
                "centroid_xyz_mm": [float(v) for v in centroid_xyz],
                "largest_axis_mm": _largest_axis_mm(coords_xyz),
                "principal_axes_mm": _principal_axes_mm(coords_xyz),
                "passes_min_voxels": passes_voxel_filter,
                "passes_min_volume_mm3": passes_volume_filter,
                **overlay_paths,
            }
            components.append(component)

        components.sort(key=lambda item: item["volume_mm3"], reverse=True)
        total_components += len(components)
        total_stone_volume_mm3 += sum(item["volume_mm3"] for item in components)
        kidneys.append(
            {
                "mask_name": mask_name,
                "mask_path": str(mask_path),
                "kidney_voxel_count": int(kidney_mask.sum()),
                "kidney_volume_ml": float(kidney_mask.sum() * voxel_volume_mm3 / 1000.0),
                "kidney_hu_mean": float(kidney_values.mean()) if kidney_values.size else None,
                "kidney_hu_max": float(kidney_values.max()) if kidney_values.size else None,
                "stone_voxel_count": int(dense_mask.sum()),
                "stone_volume_mm3": float(dense_mask.sum() * voxel_volume_mm3),
                "stone_volume_ml": float(dense_mask.sum() * voxel_volume_mm3 / 1000.0),
                "component_count": len(components),
                "components": components,
            }
        )

    return {
        "ct_path": str(ct_path),
        "mask_dir": str(mask_dir),
        "threshold_hu": float(threshold_hu),
        "min_voxels": int(min_voxels),
        "min_volume_mm3": None if min_volume_mm3 is None else float(min_volume_mm3),
        "voxel_spacing_mm": voxel_spacing,
        "summary": {
            "kidneys_analyzed": len(kidneys),
            "total_components": total_components,
            "total_stone_volume_mm3": total_stone_volume_mm3,
        },
        "kidneys": kidneys,
        "missing_masks": missing_masks,
        "disclaimer": [
            "Heuristica de threshold em HU sobre mascaras renais do TotalSegmentator; usar para triagem e revisao, nao como confirmacao diagnostica.",
            "O threshold padrao de 130 HU segue a referencia mais recorrente na literatura para deteccao de calculos em TC sem contraste.",
            "Artefatos, volume parcial, contraste, clips, calcificacoes vasculares adjacentes e segmentacao renal imperfeita podem gerar falsos positivos.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect kidney stone candidate components inside TotalSegmentator kidney masks using a HU threshold."
    )
    parser.add_argument("--ct", required=True, type=Path, help="Path to CT NIfTI in HU.")
    parser.add_argument("--mask-dir", required=True, type=Path, help="Directory containing individual TotalSegmentator masks.")
    parser.add_argument("--output", required=True, type=Path, help="Path to output JSON report.")
    parser.add_argument(
        "--masks",
        type=str,
        default=",".join(DEFAULT_MASKS),
        help="Comma-separated kidney masks to analyze. Defaults to kidney_left,kidney_right.",
    )
    parser.add_argument("--threshold-hu", type=float, default=130.0, help="HU threshold for dense stone candidate voxels.")
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=3,
        help="Minimum connected voxels required to keep a component. Default: 3.",
    )
    parser.add_argument(
        "--min-volume-mm3",
        type=float,
        default=None,
        help="Optional minimum component volume in mm3. A component is kept if it passes min-voxels or min-volume-mm3.",
    )
    parser.add_argument(
        "--render-dir",
        type=Path,
        default=None,
        help="Optional directory to save axial/coronal PNG overlays for each component.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    masks = tuple(item.strip() for item in args.masks.split(",") if item.strip())
    report = analyze_kidneys(
        ct_path=args.ct,
        mask_dir=args.mask_dir,
        threshold_hu=args.threshold_hu,
        masks=masks,
        min_voxels=args.min_voxels,
        min_volume_mm3=args.min_volume_mm3,
        render_dir=args.render_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=_to_serializable)


if __name__ == "__main__":
    main()
