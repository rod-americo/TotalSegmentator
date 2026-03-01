import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _np() -> Any:
    import numpy as np

    return np


def _ndi() -> Any:
    from scipy import ndimage

    return ndimage


def _sample_nearest_2d(ct_slice: Any, point: Sequence[float]) -> float | None:
    np = _np()
    idx = np.rint(np.asarray(point)).astype(int)
    if idx[0] < 0 or idx[1] < 0 or idx[0] >= ct_slice.shape[0] or idx[1] >= ct_slice.shape[1]:
        return None
    return float(ct_slice[tuple(idx)])


def _trace_profile_2d(
    ct_slice: Any,
    center_xy: Sequence[float],
    surface_xy: Sequence[float],
    spacing_xy: Sequence[float],
    inside_samples: int,
    outside_samples: int,
) -> Dict[str, Any] | None:
    np = _np()

    center_xy = np.asarray(center_xy, dtype=float)
    surface_xy = np.asarray(surface_xy, dtype=float)
    vec = surface_xy - center_xy
    vec_mm = vec * np.asarray(spacing_xy, dtype=float)
    length_mm = float(np.linalg.norm(vec_mm))
    if length_mm < 1.0:
        return None

    unit = vec / max(np.linalg.norm(vec), 1e-6)
    deep_points = [center_xy + vec * (i / max(inside_samples - 1, 1)) for i in range(inside_samples)]
    outside_points = [surface_xy + unit * i for i in range(1, outside_samples + 1)]
    all_points = deep_points + outside_points

    hu_values = []
    distances = []
    for idx, point in enumerate(all_points):
        sampled = _sample_nearest_2d(ct_slice, point)
        if sampled is None:
            continue
        if idx < inside_samples:
            frac = idx / max(inside_samples - 1, 1)
            dist = -length_mm * (1.0 - frac)
        else:
            dist = float((idx - inside_samples + 1) * float(sum(spacing_xy) / len(spacing_xy)))
        hu_values.append(sampled)
        distances.append(dist)

    if len(hu_values) < 6:
        return None

    surface_index = inside_samples - 1
    hu_profile = np.asarray(hu_values, dtype=float)
    outer_window_start = max(0, surface_index - 1)
    outer_window_end = min(len(hu_profile), surface_index + 2)
    outer_peak = float(np.max(hu_profile[outer_window_start:outer_window_end]))
    deep_section = hu_profile[: max(2, surface_index - 1)]
    outside_section = hu_profile[min(surface_index + 1, len(hu_profile)) :]
    if deep_section.size == 0 or outside_section.size == 0:
        return None
    deep_median = float(np.median(deep_section))
    outside_median = float(np.median(outside_section))
    rise = outer_peak - deep_median

    return {
        "distances": distances,
        "hu_values": hu_values,
        "outer_peak_hu": round(outer_peak, 3),
        "deep_median_hu": round(deep_median, 3),
        "outside_median_hu": round(outside_median, 3),
        "rise_hu": round(rise, 3),
        "path_length_mm": round(length_mm, 3),
    }


def _expand_mask_2d(mask_slice: Any, spacing_xy: Sequence[float], expansion_factor: float) -> Any:
    np = _np()
    ndi = _ndi()
    area = int(mask_slice.sum())
    if area == 0:
        return mask_slice
    effective_radius_vox = (area / np.pi) ** 0.5
    expansion_vox = max(1, int(round(effective_radius_vox * expansion_factor)))
    expanded = ndi.binary_dilation(mask_slice, iterations=expansion_vox)
    return expanded


def _select_profiles(ct_slice: Any, mask_slice: Any, expanded_slice: Any, spacing_xy: Sequence[float]) -> List[Dict[str, Any]]:
    np = _np()
    ndi = _ndi()

    dist = ndi.distance_transform_edt(mask_slice, sampling=spacing_xy)
    center_index = np.unravel_index(int(np.argmax(dist)), dist.shape)
    surface = expanded_slice & ~ndi.binary_erosion(expanded_slice, iterations=1, border_value=0)
    coords = np.argwhere(surface)
    if len(coords) == 0:
        return []
    step = max(1, len(coords) // 32)
    profiles = []
    for x, y in coords[::step]:
        profile = _trace_profile_2d(
            ct_slice=ct_slice,
            center_xy=center_index,
            surface_xy=(x, y),
            spacing_xy=spacing_xy,
            inside_samples=12,
            outside_samples=8,
        )
        if profile is None:
            continue
        profile["surface_xy"] = [int(x), int(y)]
        profiles.append(profile)

    profiles.sort(key=lambda item: (item["rise_hu"], item["outer_peak_hu"], -item["path_length_mm"]))
    return profiles[:4]


def _plot_profiles(
    output_path: Path,
    ct_slice: Any,
    mask_slice: Any,
    expanded_slice: Any,
    profiles: List[Dict[str, Any]],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np = _np()
    ndi = _ndi()

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    ax0 = axes[0, 0]
    bbox_pts = np.argwhere(expanded_slice)
    mins = np.maximum(0, bbox_pts.min(axis=0) - 18)
    maxs = np.minimum(expanded_slice.shape, bbox_pts.max(axis=0) + 19)
    img = np.rot90(ct_slice[mins[0] : maxs[0], mins[1] : maxs[1]])
    orig = np.rot90(mask_slice[mins[0] : maxs[0], mins[1] : maxs[1]])
    expd = np.rot90(expanded_slice[mins[0] : maxs[0], mins[1] : maxs[1]])
    ax0.imshow(img, cmap="gray", vmin=-250, vmax=1200)
    ax0.contour(expd.astype(float), levels=[0.5], colors=["orange"], linewidths=0.8)
    ax0.contour(orig.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.1)
    center_dist = ndi.distance_transform_edt(mask_slice)
    center_index = np.unravel_index(int(np.argmax(center_dist)), center_dist.shape)
    cx = center_index[0] - mins[0]
    cy = center_index[1] - mins[1]
    ax0.scatter(cy, img.shape[0] - 1 - cx, c="red", s=18)
    ax0.set_title("axial ROI | cyan=mask orange=expanded")
    ax0.axis("off")

    plot_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    for ax, profile in zip(plot_axes, profiles):
        ax.plot(profile["distances"], profile["hu_values"], color="black", linewidth=1.6)
        ax.axvline(0.0, color="orange", linestyle="--", linewidth=1.0)
        ax.axhline(profile["deep_median_hu"], color="royalblue", linestyle=":", linewidth=1.0)
        ax.axhline(profile["outside_median_hu"], color="seagreen", linestyle=":", linewidth=1.0)
        ax.axhline(profile["outer_peak_hu"], color="firebrick", linestyle=":", linewidth=1.0)
        ax.set_title(f"rise={profile['rise_hu']:.1f} HU | len={profile['path_length_mm']:.1f} mm")
        ax.set_xlabel("center -> expanded surface -> outside")
        ax.set_ylabel("HU")
        ax.grid(alpha=0.25)
    for ax in plot_axes[len(profiles) :]:
        ax.axis("off")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0,
        1.0,
        "Interpretation:\n"
        "- high outer peak near 0 mm suggests preserved cortex\n"
        "- low/flattened outer peak suggests cortical failure or mask mismatch\n"
        "- orange contour is the 50% expanded mask used for the profile boundary",
        va="top",
    )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_debug(ct_path: Path, mask_dir: Path, structures: Sequence[str], slices: Sequence[int], output_dir: Path, expansion_factor: float) -> Dict[str, Any]:
    import nibabel as nib
    import numpy as np

    ct_img = nib.load(str(ct_path))
    ct = ct_img.get_fdata().astype(np.int16)
    spacing = ct_img.header.get_zooms()[:3]

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for structure in structures:
        mask_path = mask_dir / f"{structure}.nii.gz"
        if not mask_path.exists():
            continue
        mask = nib.load(str(mask_path)).get_fdata() > 0.5
        summary[structure] = {}
        for z in slices:
            mask_slice = mask[:, :, z]
            if mask_slice.sum() == 0:
                continue
            expanded_slice = _expand_mask_2d(mask_slice, spacing[:2], expansion_factor=expansion_factor)
            profiles = _select_profiles(ct[:, :, z], mask_slice, expanded_slice, spacing[:2])
            out_path = output_dir / f"{structure}_z{z:03d}.png"
            _plot_profiles(
                output_path=out_path,
                ct_slice=ct[:, :, z],
                mask_slice=mask_slice,
                expanded_slice=expanded_slice,
                profiles=profiles,
                title=f"{structure} | z={z} | expansion={int(expansion_factor * 100)}%",
            )
            summary[structure][str(z)] = {
                "plot": str(out_path),
                "profiles": profiles,
                "original_area_vox": int(mask_slice.sum()),
                "expanded_area_vox": int(expanded_slice.sum()),
            }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug axial center-to-boundary attenuation profiles using a 50% mask expansion.")
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=str, help="Comma-separated mask names.")
    parser.add_argument("--slices", required=True, type=str, help="Comma-separated slice indices and ranges like 206-218,240-252")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expansion-factor", type=float, default=0.5)
    return parser


def _parse_slices(raw: str) -> List[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(list(range(int(start), int(end) + 1)))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> None:
    args = build_parser().parse_args()
    run_debug(
        ct_path=args.ct,
        mask_dir=args.mask_dir,
        structures=[item.strip() for item in args.structures.split(",") if item.strip()],
        slices=_parse_slices(args.slices),
        output_dir=args.output_dir,
        expansion_factor=args.expansion_factor,
    )


if __name__ == "__main__":
    main()
