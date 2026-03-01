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


def _expand_mask_2d(mask_slice: Any, expansion_factor: float) -> Any:
    np = _np()
    ndi = _ndi()
    area = int(mask_slice.sum())
    if area == 0:
        return mask_slice
    effective_radius_vox = (area / np.pi) ** 0.5
    expansion_vox = max(1, int(round(effective_radius_vox * expansion_factor)))
    return ndi.binary_dilation(mask_slice, iterations=expansion_vox)


def _largest_component_2d(mask_slice: Any) -> Any:
    np = _np()
    ndi = _ndi()
    labeled, n = ndi.label(mask_slice)
    if n == 0:
        return mask_slice
    counts = np.bincount(labeled.ravel())[1:]
    label = int(np.argmax(counts)) + 1
    return labeled == label


def _radial_profile(
    ct_slice: Any,
    center_xy: Sequence[float],
    theta: float,
    inside_radius: float,
    outside_radius: float,
    step_vox: float,
) -> Dict[str, Any] | None:
    import math
    np = _np()

    direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=float)
    distances = np.arange(-inside_radius, outside_radius + step_vox, step_vox)
    hu_values = []
    used_distances = []
    for dist in distances:
        point = np.asarray(center_xy, dtype=float) + direction * dist
        sampled = _sample_nearest_2d(ct_slice, point)
        if sampled is None:
            continue
        hu_values.append(sampled)
        used_distances.append(float(dist))
    if len(hu_values) < 10:
        return None

    hu = np.asarray(hu_values, dtype=float)
    d = np.asarray(used_distances, dtype=float)
    near_surface = (d >= -1.5) & (d <= 1.5)
    deep_region = d <= -2.5
    outside_region = d >= 1.5
    if not near_surface.any() or not deep_region.any() or not outside_region.any():
        return None

    outer_peak = float(np.max(hu[near_surface]))
    deep_median = float(np.median(hu[deep_region]))
    outside_median = float(np.median(hu[outside_region]))
    rise = outer_peak - deep_median
    return {
        "theta_rad": float(theta),
        "distances": used_distances,
        "hu_values": hu_values,
        "outer_peak_hu": round(outer_peak, 3),
        "deep_median_hu": round(deep_median, 3),
        "outside_median_hu": round(outside_median, 3),
        "rise_hu": round(rise, 3),
    }


def _sector_suspicion(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    np = _np()
    if not profiles:
        return []
    profiles = sorted(profiles, key=lambda p: p["theta_rad"])
    peaks = np.asarray([p["outer_peak_hu"] for p in profiles], dtype=float)
    rises = np.asarray([p["rise_hu"] for p in profiles], dtype=float)
    n = len(profiles)
    peak_ref = float(np.percentile(peaks, 85))
    rise_ref = float(np.percentile(rises, 85))
    peak_mid = float(np.percentile(peaks, 50))
    peak_floor = float(np.percentile(peaks, 25))
    rise_floor = float(np.percentile(rises, 25))
    out = []
    raw_flags = []
    for idx, profile in enumerate(profiles):
        peak_drop = peak_ref - profile["outer_peak_hu"]
        rise_drop = rise_ref - profile["rise_hu"]
        peak_norm = peak_drop / max(1.0, peak_ref - peak_floor)
        rise_norm = rise_drop / max(1.0, rise_ref - rise_floor)
        suspicion_score = max(peak_norm, rise_norm)
        raw_flag = (
            rise_drop >= 250
            and rise_norm >= 0.40
            and profile["outer_peak_hu"] <= peak_mid + 20.0
        )
        raw_flags.append(raw_flag)
        out.append(
            {
                **profile,
                "peak_drop_hu": round(float(peak_drop), 3),
                "rise_drop_hu": round(float(rise_drop), 3),
                "suspicion_score": round(float(suspicion_score), 3),
                "suspicious": False,
            }
        )

    # Require angular continuity: at least 2 adjacent flagged sectors.
    for idx in range(n):
        if not raw_flags[idx]:
            continue
        group = [raw_flags[(idx - 1) % n], raw_flags[idx], raw_flags[(idx + 1) % n]]
        if sum(group) >= 2:
            out[idx]["suspicious"] = True
    return out


def _plot_slice(
    output_path: Path,
    ct_slice: Any,
    mask_slice: Any,
    expanded_slice: Any,
    center_xy: Sequence[float],
    profiles: List[Dict[str, Any]],
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    suspicious = [p for p in profiles if p["suspicious"]]
    normal = [p for p in profiles if not p["suspicious"]]

    bbox_pts = np.argwhere(expanded_slice)
    mins = np.maximum(0, bbox_pts.min(axis=0) - 18)
    maxs = np.minimum(expanded_slice.shape, bbox_pts.max(axis=0) + 19)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax0 = axes[0, 0]
    img = np.rot90(ct_slice[mins[0] : maxs[0], mins[1] : maxs[1]])
    orig = np.rot90(mask_slice[mins[0] : maxs[0], mins[1] : maxs[1]])
    expd = np.rot90(expanded_slice[mins[0] : maxs[0], mins[1] : maxs[1]])
    ax0.imshow(img, cmap="gray", vmin=-250, vmax=1200)
    ax0.contour(expd.astype(float), levels=[0.5], colors=["orange"], linewidths=0.8)
    ax0.contour(orig.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.1)
    cx = center_xy[0] - mins[0]
    cy = center_xy[1] - mins[1]
    ax0.scatter(cy, img.shape[0] - 1 - cx, c="red", s=24)
    for profile in suspicious:
        theta = profile["theta_rad"]
        dx = 12 * np.cos(theta)
        dy = 12 * np.sin(theta)
        ax0.arrow(
            cy,
            img.shape[0] - 1 - cx,
            dy,
            -dx,
            color="yellow",
            width=0.15,
            head_width=2.5,
            length_includes_head=True,
        )
    ax0.set_title("axial ROI | yellow=suspicious sectors")
    ax0.axis("off")

    ax1 = axes[0, 1]
    for profile in normal[:8]:
        ax1.plot(profile["distances"], profile["hu_values"], color="gray", alpha=0.35)
    for profile in suspicious:
        ax1.plot(profile["distances"], profile["hu_values"], color="red", linewidth=1.8)
    ax1.axvline(0.0, color="orange", linestyle="--", linewidth=1.0)
    ax1.set_title("all radial profiles")
    ax1.set_xlabel("center -> surface -> outside")
    ax1.set_ylabel("HU")
    ax1.grid(alpha=0.25)

    ax2 = axes[1, 0]
    angles = [p["theta_rad"] for p in profiles]
    peaks = [p["outer_peak_hu"] for p in profiles]
    rises = [p["rise_hu"] for p in profiles]
    colors = ["red" if p["suspicious"] else "black" for p in profiles]
    ax2.scatter(angles, peaks, c=colors, s=20)
    ax2.set_title("outer peak by angle")
    ax2.set_xlabel("angle (rad)")
    ax2.set_ylabel("outer peak HU")
    ax2.grid(alpha=0.25)

    ax3 = axes[1, 1]
    ax3.scatter(angles, rises, c=colors, s=20)
    ax3.set_title("rise by angle")
    ax3.set_xlabel("angle (rad)")
    ax3.set_ylabel("rise HU")
    ax3.grid(alpha=0.25)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_slice(ct_slice: Any, mask_slice: Any, expansion_factor: float) -> Dict[str, Any] | None:
    import numpy as np

    mask_slice = _largest_component_2d(mask_slice)
    if mask_slice.sum() < 20:
        return None
    expanded = _expand_mask_2d(mask_slice, expansion_factor=expansion_factor)
    coords = np.argwhere(mask_slice)
    center = coords.mean(axis=0)
    center = np.asarray(center, dtype=float)
    orig_pts = np.argwhere(mask_slice)
    dists = np.linalg.norm(orig_pts - center[None, :], axis=1)
    inside_radius = float(np.percentile(dists, 90))
    exp_pts = np.argwhere(expanded)
    exp_dists = np.linalg.norm(exp_pts - center[None, :], axis=1)
    outside_radius = float(np.percentile(exp_dists, 95)) + 6.0

    coords = np.argwhere(mask_slice)
    covariance = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    major_axis = eigvecs[:, int(np.argmax(eigvals))]
    raw_profiles = []
    for theta in np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False):
        direction = np.asarray([np.cos(theta), np.sin(theta)], dtype=float)
        # Keep only the hemicircle roughly orthogonal to the rib long axis.
        if abs(float(np.dot(direction, major_axis))) > 0.5:
            continue
        profile = _radial_profile(
            ct_slice=ct_slice,
            center_xy=center,
            theta=float(theta),
            inside_radius=inside_radius,
            outside_radius=outside_radius,
            step_vox=0.5,
        )
        if profile is not None:
            raw_profiles.append(profile)
    profiles = _sector_suspicion(raw_profiles)
    return {
        "center_xy": [round(float(center[0]), 3), round(float(center[1]), 3)],
        "profiles": profiles,
        "mask_slice": mask_slice,
        "expanded_slice": expanded,
    }


def run_debug(ct_path: Path, mask_dir: Path, structures: Sequence[str], slices: Sequence[int], output_dir: Path, expansion_factor: float) -> Dict[str, Any]:
    import nibabel as nib
    import numpy as np

    ct = nib.load(str(ct_path)).get_fdata().astype(np.int16)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for structure in structures:
        mask_path = mask_dir / f"{structure}.nii.gz"
        if not mask_path.exists():
            continue
        mask = nib.load(str(mask_path)).get_fdata() > 0.5
        summary[structure] = {}
        for z in slices:
            result = analyze_slice(ct[:, :, z], mask[:, :, z], expansion_factor=expansion_factor)
            if result is None:
                continue
            output_path = output_dir / f"{structure}_z{z:03d}.png"
            _plot_slice(
                output_path=output_path,
                ct_slice=ct[:, :, z],
                mask_slice=result["mask_slice"],
                expanded_slice=result["expanded_slice"],
                center_xy=result["center_xy"],
                profiles=result["profiles"],
                title=f"{structure} | z={z} | angular axial debug | expansion={int(expansion_factor * 100)}%",
            )
            summary[structure][str(z)] = {
                "plot": str(output_path),
                "center_xy": result["center_xy"],
                "num_profiles": len(result["profiles"]),
                "num_suspicious": sum(1 for p in result["profiles"] if p["suspicious"]),
                "top_suspicious": [p for p in result["profiles"] if p["suspicious"]][:8],
            }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _parse_slices(raw: str) -> List[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Angular axial rib profile debug with expanded masks.")
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=str)
    parser.add_argument("--slices", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expansion-factor", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_debug(
        ct_path=args.ct,
        mask_dir=args.mask_dir,
        structures=[s.strip() for s in args.structures.split(",") if s.strip()],
        slices=_parse_slices(args.slices),
        output_dir=args.output_dir,
        expansion_factor=args.expansion_factor,
    )


if __name__ == "__main__":
    main()
