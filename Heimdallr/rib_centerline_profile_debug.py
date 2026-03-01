import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _np() -> Any:
    import numpy as np

    return np


def _ndi() -> Any:
    from scipy import ndimage

    return ndimage


def _load_nifti(path: Path) -> Tuple[Any, Any]:
    import nibabel as nib

    img = nib.load(path)
    return img.get_fdata(), img


def _patient_ap_coord(affine: Any, point_xyz: Sequence[float]) -> float:
    import nibabel as nib
    import numpy as np

    world = nib.affines.apply_affine(affine, np.asarray(point_xyz, dtype=float))
    axcodes = nib.aff2axcodes(affine)
    for axis, code in enumerate(axcodes):
        if code in ("A", "P"):
            return float(world[axis] if code == "A" else -world[axis])
    return float(world[1])


def _normalize(vec: Any) -> Any:
    np = _np()
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


def _sample_trilinear(volume: Any, point_xyz: Sequence[float]) -> Optional[float]:
    np = _np()
    ndi = _ndi()
    point = np.asarray(point_xyz, dtype=float).reshape(3, 1)
    shape = np.asarray(volume.shape, dtype=float)
    if np.any(point < 0.0) or np.any(point > (shape.reshape(3, 1) - 1.0)):
        return None
    sampled = ndi.map_coordinates(volume, point, order=1, mode="nearest")
    return float(sampled[0])


def _point_inside(mask: Any, point_xyz: Sequence[float]) -> bool:
    np = _np()
    idx = np.rint(np.asarray(point_xyz, dtype=float)).astype(int)
    if np.any(idx < 0) or np.any(idx >= np.asarray(mask.shape)):
        return False
    return bool(mask[tuple(idx)])


def _expand_mask(mask: Any, factor: float) -> Any:
    np = _np()
    ndi = _ndi()
    volume = int(mask.sum())
    if volume == 0:
        return mask
    radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    iterations = max(1, int(round(radius * factor)))
    return ndi.binary_dilation(mask, iterations=iterations)


def _principal_axes(coords_phys: Any) -> Tuple[Any, Any]:
    np = _np()
    centered = coords_phys - coords_phys.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def _estimate_centerline(mask: Any, spacing_xyz: Sequence[float], step_mm: float) -> List[Dict[str, Any]]:
    np = _np()
    ndi = _ndi()
    coords = np.argwhere(mask)
    spacing = np.asarray(spacing_xyz, dtype=float)
    dist_mm = ndi.distance_transform_edt(mask, sampling=spacing_xyz)
    coords_phys = coords * spacing[None, :]
    _, eigvecs = _principal_axes(coords_phys)
    major = _normalize(eigvecs[:, 0])
    second = _normalize(eigvecs[:, 1])
    third = _normalize(eigvecs[:, 2])
    center_phys = coords_phys.mean(axis=0)
    t = (coords_phys - center_phys[None, :]) @ major
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max - t_min < step_mm:
        step_mm = max(1.0, (t_max - t_min) / 8.0)
    edges = np.arange(t_min, t_max + step_mm, step_mm)
    if len(edges) < 3:
        edges = np.linspace(t_min, t_max, 12)

    points: List[Dict[str, Any]] = []
    for idx in range(len(edges) - 1):
        lo = edges[idx]
        hi = edges[idx + 1]
        in_bin = (t >= lo) & (t < hi if idx < len(edges) - 2 else t <= hi)
        if int(in_bin.sum()) < 25:
            continue
        bin_coords = coords[in_bin]
        bin_coords_phys = coords_phys[in_bin]
        centroid_phys = bin_coords_phys.mean(axis=0)
        centroid_dist_mm = np.linalg.norm(bin_coords_phys - centroid_phys[None, :], axis=1)
        centrality = np.asarray([dist_mm[tuple(coord)] for coord in bin_coords], dtype=float)
        # Favor voxels deep inside the rib while staying close to the slab centroid.
        score = centrality - 0.2 * centroid_dist_mm
        best_idx = int(np.argmax(score))
        centroid_phys = bin_coords_phys[best_idx]
        centroid_xyz = centroid_phys / spacing
        points.append(
            {
                "t_mm": 0.5 * (lo + hi),
                "center_xyz": centroid_xyz,
                "center_phys": centroid_phys,
            }
        )

    if len(points) < 5:
        return []

    centers = np.asarray([p["center_phys"] for p in points], dtype=float)
    smoothed = centers.copy()
    for idx in range(1, len(points) - 1):
        smoothed[idx] = 0.25 * centers[idx - 1] + 0.5 * centers[idx] + 0.25 * centers[idx + 1]

    out: List[Dict[str, Any]] = []
    for idx, point in enumerate(points):
        if idx == 0:
            tangent = smoothed[idx + 1] - smoothed[idx]
        elif idx == len(points) - 1:
            tangent = smoothed[idx] - smoothed[idx - 1]
        else:
            tangent = smoothed[idx + 1] - smoothed[idx - 1]
        tangent = _normalize(tangent)
        normal_1 = second - np.dot(second, tangent) * tangent
        if np.linalg.norm(normal_1) < 1e-6:
            normal_1 = third - np.dot(third, tangent) * tangent
        normal_1 = _normalize(normal_1)
        normal_2 = _normalize(np.cross(tangent, normal_1))
        out.append(
            {
                "t_mm": round(float(point["t_mm"]), 3),
                "center_xyz": [round(float(v), 3) for v in (smoothed[idx] / spacing)],
                "center_phys": [round(float(v), 3) for v in smoothed[idx]],
                "tangent": [round(float(v), 6) for v in tangent],
                "normal_1": [round(float(v), 6) for v in normal_1],
                "normal_2": [round(float(v), 6) for v in normal_2],
            }
        )
    return out


def _surface_distance(mask: Any, center_xyz: Sequence[float], direction_phys: Sequence[float], spacing_xyz: Sequence[float], max_mm: float) -> Optional[float]:
    np = _np()
    spacing = np.asarray(spacing_xyz, dtype=float)
    center = np.asarray(center_xyz, dtype=float)
    direction_phys = _normalize(direction_phys)
    direction_xyz = direction_phys / spacing
    direction_xyz = direction_xyz / max(np.linalg.norm(direction_xyz), 1e-8)

    last_inside = None
    for dist_mm in np.arange(0.0, max_mm + 0.25, 0.25):
        point_xyz = center + direction_xyz * dist_mm
        if _point_inside(mask, point_xyz):
            last_inside = float(dist_mm)
            continue
        return last_inside
    return last_inside


def _radial_profile(
    ct: Any,
    surface_mask: Any,
    center_xyz: Sequence[float],
    direction_phys: Sequence[float],
    spacing_xyz: Sequence[float],
    outside_factor: float,
    max_mm: float,
) -> Optional[Dict[str, Any]]:
    np = _np()
    surface_mm = _surface_distance(surface_mask, center_xyz, direction_phys, spacing_xyz, max_mm=max_mm)
    if surface_mm is None or surface_mm < 1.0:
        return None

    sample_end_mm = surface_mm * (1.0 + outside_factor) + 2.0
    spacing = np.asarray(spacing_xyz, dtype=float)
    direction_phys = _normalize(direction_phys)
    direction_xyz = direction_phys / spacing
    direction_xyz = direction_xyz / max(np.linalg.norm(direction_xyz), 1e-8)
    distances = np.arange(0.0, sample_end_mm + 0.25, 0.25)
    hu_values: List[float] = []
    used_mm: List[float] = []
    for dist_mm in distances:
        point_xyz = np.asarray(center_xyz, dtype=float) + direction_xyz * dist_mm
        sampled = _sample_trilinear(ct, point_xyz)
        if sampled is None:
            continue
        hu_values.append(sampled)
        used_mm.append(float(dist_mm))

    if len(hu_values) < 12:
        return None

    d = np.asarray(used_mm, dtype=float)
    hu = np.asarray(hu_values, dtype=float)
    deep_region = d <= max(0.8, 0.45 * surface_mm)
    near_surface = (d >= max(0.0, surface_mm - 1.0)) & (d <= surface_mm + 1.0)
    outside_region = d >= surface_mm + 0.75
    if not deep_region.any() or not near_surface.any() or not outside_region.any():
        return None

    deep_median = float(np.median(hu[deep_region]))
    outer_peak = float(np.max(hu[near_surface]))
    outside_median = float(np.median(hu[outside_region]))
    rise = outer_peak - deep_median
    drop_to_outside = outer_peak - outside_median
    return {
        "surface_mm": round(float(surface_mm), 3),
        "distances_mm": [round(float(v), 3) for v in used_mm],
        "hu_values": [round(float(v), 3) for v in hu_values],
        "deep_median_hu": round(deep_median, 3),
        "outer_peak_hu": round(outer_peak, 3),
        "outside_median_hu": round(outside_median, 3),
        "rise_hu": round(rise, 3),
        "drop_to_outside_hu": round(drop_to_outside, 3),
    }


def _analyze_rib(
    ct: Any,
    mask: Any,
    spacing_xyz: Sequence[float],
    centerline: List[Dict[str, Any]],
    expansion_factor: float,
    angle_step_deg: float,
) -> Dict[str, Any]:
    import numpy as np

    expanded = _expand_mask(mask, expansion_factor)
    angle_values = np.arange(0.0, 360.0, angle_step_deg)
    outer_peak_map = np.full((len(centerline), len(angle_values)), np.nan, dtype=float)
    rise_map = np.full_like(outer_peak_map, np.nan)
    drop_map = np.full_like(outer_peak_map, np.nan)
    surface_map = np.full_like(outer_peak_map, np.nan)
    top_profiles: List[Dict[str, Any]] = []

    for i, point in enumerate(centerline):
        center_xyz = np.asarray(point["center_xyz"], dtype=float)
        n1 = np.asarray(point["normal_1"], dtype=float)
        n2 = np.asarray(point["normal_2"], dtype=float)
        for j, angle_deg in enumerate(angle_values):
            theta = math.radians(float(angle_deg))
            direction_phys = math.cos(theta) * n1 + math.sin(theta) * n2
            profile = _radial_profile(
                ct=ct,
                surface_mask=mask,
                center_xyz=center_xyz,
                direction_phys=direction_phys,
                spacing_xyz=spacing_xyz,
                outside_factor=expansion_factor,
                max_mm=20.0,
            )
            if profile is None:
                continue
            outer_peak_map[i, j] = float(profile["outer_peak_hu"])
            rise_map[i, j] = float(profile["rise_hu"])
            drop_map[i, j] = float(profile["drop_to_outside_hu"])
            surface_map[i, j] = float(profile["surface_mm"])
            top_profiles.append(
                {
                    "position_index": i,
                    "t_mm": point["t_mm"],
                    "angle_deg": round(float(angle_deg), 3),
                    **profile,
                }
            )

    valid_rise = rise_map[np.isfinite(rise_map)]
    valid_peak = outer_peak_map[np.isfinite(outer_peak_map)]
    valid_drop = drop_map[np.isfinite(drop_map)]
    if valid_rise.size:
        rise_ref = float(np.percentile(valid_rise, 85))
        peak_ref = float(np.percentile(valid_peak, 85))
        drop_ref = float(np.percentile(valid_drop, 85))
        suspicious = (
            (rise_map <= rise_ref - 220.0)
            & (outer_peak_map <= peak_ref - 80.0)
            & (drop_map <= drop_ref - 120.0)
        )
    else:
        rise_ref = peak_ref = drop_ref = 0.0
        suspicious = np.zeros_like(rise_map, dtype=bool)

    suspicious_components = []
    if suspicious.any():
        ndi = _ndi()
        labeled, n_labels = ndi.label(suspicious)
        for label in range(1, n_labels + 1):
            coords = np.argwhere(labeled == label)
            suspicious_components.append(
                {
                    "component": label,
                    "num_cells": int(coords.shape[0]),
                    "position_index_min": int(coords[:, 0].min()),
                    "position_index_max": int(coords[:, 0].max()),
                    "angle_index_min": int(coords[:, 1].min()),
                    "angle_index_max": int(coords[:, 1].max()),
                }
            )
        suspicious_components.sort(key=lambda item: item["num_cells"], reverse=True)

    top_profiles.sort(key=lambda item: (item["rise_hu"], item["outer_peak_hu"], item["drop_to_outside_hu"]))
    return {
        "expanded_voxels": int(expanded.sum()),
        "angle_values_deg": [round(float(v), 3) for v in angle_values],
        "rise_ref_hu": round(rise_ref, 3),
        "peak_ref_hu": round(peak_ref, 3),
        "drop_ref_hu": round(drop_ref, 3),
        "outer_peak_map": outer_peak_map,
        "rise_map": rise_map,
        "drop_map": drop_map,
        "surface_map": surface_map,
        "suspicious_map": suspicious,
        "components": suspicious_components[:20],
        "top_profiles": top_profiles[:25],
    }


def _orient_centerline_posterior_to_anterior(centerline: List[Dict[str, Any]], affine: Any) -> List[Dict[str, Any]]:
    import numpy as np

    if len(centerline) < 2:
        return centerline
    first_ap = _patient_ap_coord(affine, centerline[0]["center_xyz"])
    last_ap = _patient_ap_coord(affine, centerline[-1]["center_xyz"])
    ordered = list(centerline if first_ap <= last_ap else list(reversed(centerline)))

    cumulative_mm = 0.0
    prev = None
    for point in ordered:
        center_phys = np.asarray(point["center_phys"], dtype=float)
        if prev is not None:
            cumulative_mm += float(np.linalg.norm(center_phys - prev))
        point["distance_cm_pa"] = round(cumulative_mm / 10.0, 3)
        prev = center_phys
    return ordered


def _plot_maps(output_path: Path, structure: str, centerline: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x_values = [p["distance_cm_pa"] for p in centerline]
    angle_values = np.asarray(analysis["angle_values_deg"], dtype=float)
    rise_map = np.asarray(analysis["rise_map"], dtype=float)
    peak_map = np.asarray(analysis["outer_peak_map"], dtype=float)
    suspicious = np.asarray(analysis["suspicious_map"], dtype=bool)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    im0 = axes[0].imshow(
        peak_map.T,
        origin="lower",
        aspect="auto",
        extent=[x_values[0], x_values[-1], angle_values[0], angle_values[-1]],
        cmap="magma",
    )
    axes[0].set_title("Outer Peak HU")
    axes[0].set_ylabel("Angle (deg)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(
        rise_map.T,
        origin="lower",
        aspect="auto",
        extent=[x_values[0], x_values[-1], angle_values[0], angle_values[-1]],
        cmap="viridis",
    )
    axes[1].set_title("Rise HU (center -> cortical)")
    axes[1].set_ylabel("Angle (deg)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(
        suspicious.T.astype(float),
        origin="lower",
        aspect="auto",
        extent=[x_values[0], x_values[-1], angle_values[0], angle_values[-1]],
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
    )
    axes[2].set_title("Suspicious sectors")
    axes[2].set_xlabel("Posterior -> Anterior Distance (cm)")
    axes[2].set_ylabel("Angle (deg)")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    fig.suptitle(f"{structure} | centerline 3D rib profile map", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_position_summary(
    output_path: Path,
    structure: str,
    centerline: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    segments: Sequence[Dict[str, Any]] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x_values = np.asarray([p["distance_cm_pa"] for p in centerline], dtype=float)
    rise_map = np.asarray(analysis["rise_map"], dtype=float)
    peak_map = np.asarray(analysis["outer_peak_map"], dtype=float)
    suspicious = np.asarray(analysis["suspicious_map"], dtype=bool)

    rise_p10 = np.nanpercentile(rise_map, 10, axis=1)
    rise_med = np.nanpercentile(rise_map, 50, axis=1)
    peak_p10 = np.nanpercentile(peak_map, 10, axis=1)
    peak_med = np.nanpercentile(peak_map, 50, axis=1)
    suspicious_frac = np.nanmean(suspicious.astype(float), axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(x_values, rise_med, color="steelblue", linewidth=1.5, label="median rise")
    axes[0].plot(x_values, rise_p10, color="firebrick", linewidth=1.5, label="10th percentile rise")
    axes[0].set_ylabel("Rise HU")
    axes[0].set_title("Center-to-cortical rise along rib")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(x_values, peak_med, color="darkgreen", linewidth=1.5, label="median outer peak")
    axes[1].plot(x_values, peak_p10, color="darkorange", linewidth=1.5, label="10th percentile outer peak")
    axes[1].set_ylabel("Outer Peak HU")
    axes[1].set_title("Cortical peak along rib")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(x_values, suspicious_frac, color="black", linewidth=1.7)
    axes[2].set_ylabel("Suspicious Sector Fraction")
    axes[2].set_xlabel("Posterior -> Anterior Distance (cm)")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_title("Fraction of angular sectors flagged at each rib position")
    axes[2].grid(alpha=0.25)

    for axis in axes:
        for segment in segments or []:
            color = "gold" if segment.get("segment_type") == "core_suspicious" else "lightskyblue"
            axis.axvspan(segment["start_cm_pa"], segment["end_cm_pa"], color=color, alpha=0.2)

    fig.suptitle(f"{structure} | posterior-anterior rib summary", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _detect_position_segments(centerline: List[Dict[str, Any]], analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    import numpy as np

    x_values = np.asarray([p["distance_cm_pa"] for p in centerline], dtype=float)
    rise_map = np.asarray(analysis["rise_map"], dtype=float)
    peak_map = np.asarray(analysis["outer_peak_map"], dtype=float)
    suspicious = np.asarray(analysis["suspicious_map"], dtype=bool)

    rise_p10 = np.nanpercentile(rise_map, 10, axis=1)
    rise_med = np.nanpercentile(rise_map, 50, axis=1)
    peak_p10 = np.nanpercentile(peak_map, 10, axis=1)
    peak_med = np.nanpercentile(peak_map, 50, axis=1)
    suspicious_frac = np.nanmean(suspicious.astype(float), axis=1)

    rise_p10_ref = float(np.nanpercentile(rise_p10, 70))
    peak_p10_ref = float(np.nanpercentile(peak_p10, 70))
    suspicious_ref = float(np.nanpercentile(suspicious_frac, 70))

    total_length_cm = float(x_values[-1] - x_values[0]) if len(x_values) > 1 else 0.0
    posterior_guard_cm = 0.8
    anterior_guard_cm = 1.5
    core_region = (x_values >= x_values[0] + posterior_guard_cm) & (x_values <= x_values[-1] - anterior_guard_cm)
    anterior_transition = (x_values > x_values[-1] - anterior_guard_cm) & (x_values <= x_values[-1] - 0.2)
    posterior_transition = (x_values >= x_values[0]) & (x_values < x_values[0] + posterior_guard_cm)

    drop_score = np.zeros_like(x_values, dtype=float)
    drop_score += np.clip((rise_p10_ref - rise_p10) / 180.0, 0.0, None)
    drop_score += np.clip((peak_p10_ref - peak_p10) / 140.0, 0.0, None)
    drop_score += np.clip((suspicious_frac - suspicious_ref) / 0.25, 0.0, None)
    drop_score[anterior_transition | posterior_transition] *= 0.55

    def _build_segments(flags: Any, minimum_length_cm: float, segment_type: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        start = None
        for idx, flag in enumerate(flags):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                end = idx - 1
                start_cm = float(x_values[start])
                end_cm = float(x_values[end])
                length_cm = end_cm - start_cm
                if length_cm >= minimum_length_cm:
                    segment_score = float(np.nanmax(drop_score[start : end + 1]))
                    out.append(
                        {
                            "segment_type": segment_type,
                            "start_cm_pa": round(start_cm, 3),
                            "end_cm_pa": round(end_cm, 3),
                            "length_cm": round(length_cm, 3),
                            "peak_drop_score": round(segment_score, 3),
                            "min_rise_p10_hu": round(float(np.nanmin(rise_p10[start : end + 1])), 3),
                            "min_peak_p10_hu": round(float(np.nanmin(peak_p10[start : end + 1])), 3),
                            "max_suspicious_fraction": round(float(np.nanmax(suspicious_frac[start : end + 1])), 3),
                        }
                    )
                start = None
        if start is not None:
            end = len(flags) - 1
            start_cm = float(x_values[start])
            end_cm = float(x_values[end])
            length_cm = end_cm - start_cm
            if length_cm >= minimum_length_cm:
                segment_score = float(np.nanmax(drop_score[start : end + 1]))
                out.append(
                    {
                        "segment_type": segment_type,
                        "start_cm_pa": round(start_cm, 3),
                        "end_cm_pa": round(end_cm, 3),
                        "length_cm": round(length_cm, 3),
                        "peak_drop_score": round(segment_score, 3),
                        "min_rise_p10_hu": round(float(np.nanmin(rise_p10[start : end + 1])), 3),
                        "min_peak_p10_hu": round(float(np.nanmin(peak_p10[start : end + 1])), 3),
                        "max_suspicious_fraction": round(float(np.nanmax(suspicious_frac[start : end + 1])), 3),
                    }
                )
        return out

    core_flags = (
        core_region
        & (drop_score >= 1.15)
        & (rise_p10 <= rise_p10_ref - 60.0)
        & (peak_p10 <= peak_p10_ref - 40.0)
    )
    anterior_flags = (
        anterior_transition
        & (drop_score >= 0.95)
        & (rise_p10 <= rise_p10_ref - 40.0)
        & (peak_p10 <= peak_p10_ref - 25.0)
    )

    segments: List[Dict[str, Any]] = []
    segments.extend(_build_segments(core_flags, minimum_length_cm=0.6, segment_type="core_suspicious"))
    segments.extend(_build_segments(anterior_flags, minimum_length_cm=0.5, segment_type="anterior_low_confidence"))
    segments.sort(key=lambda item: (item["segment_type"] != "core_suspicious", -item["peak_drop_score"], -item["length_cm"]))
    return segments


def _plot_top_profiles(output_path: Path, structure: str, analysis: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_profiles = analysis["top_profiles"][:6]
    if not top_profiles:
        return

    fig, axes = plt.subplots(len(top_profiles), 1, figsize=(10, 2.6 * len(top_profiles)), sharex=False)
    if len(top_profiles) == 1:
        axes = [axes]
    for ax, profile in zip(axes, top_profiles):
        ax.plot(profile["distances_mm"], profile["hu_values"], color="firebrick", linewidth=1.5)
        ax.axvline(profile["surface_mm"], color="orange", linestyle="--", linewidth=1.0)
        ax.axhline(profile["outer_peak_hu"], color="dimgray", linestyle=":", linewidth=1.0)
        ax.set_title(
            f"pos={profile['position_index']} | PA={profile.get('distance_cm_pa', float('nan')):.1f} cm | angle={profile['angle_deg']:.0f} deg | "
            f"rise={profile['rise_hu']:.1f} | peak={profile['outer_peak_hu']:.1f}"
        )
        ax.set_xlabel("Distance from center (mm)")
        ax.set_ylabel("HU")
        ax.grid(alpha=0.25)
    fig.suptitle(f"{structure} | lowest-profile sectors", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_debug(
    ct_path: Path,
    mask_dir: Path,
    structures: Sequence[str],
    output_dir: Path,
    expansion_factor: float,
    centerline_step_mm: float,
    angle_step_deg: float,
) -> Dict[str, Any]:
    ct, ct_img = _load_nifti(ct_path)
    ct = ct.astype("float32")
    spacing_xyz = ct_img.header.get_zooms()[:3]
    affine = ct_img.affine
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    for structure in structures:
        mask_path = mask_dir / f"{structure}.nii.gz"
        if not mask_path.exists():
            continue
        mask, _ = _load_nifti(mask_path)
        mask = mask > 0.5
        centerline = _estimate_centerline(mask, spacing_xyz=spacing_xyz, step_mm=centerline_step_mm)
        if not centerline:
            summary[structure] = {"error": "failed_to_estimate_centerline"}
            continue
        centerline = _orient_centerline_posterior_to_anterior(centerline, affine)
        analysis = _analyze_rib(
            ct=ct,
            mask=mask,
            spacing_xyz=spacing_xyz,
            centerline=centerline,
            expansion_factor=expansion_factor,
            angle_step_deg=angle_step_deg,
        )
        map_path = output_dir / f"{structure}_maps.png"
        summary_path = output_dir / f"{structure}_pa_summary.png"
        profiles_path = output_dir / f"{structure}_profiles.png"
        distance_by_index = {idx: point["distance_cm_pa"] for idx, point in enumerate(centerline)}
        for profile in analysis["top_profiles"]:
            profile["distance_cm_pa"] = distance_by_index.get(profile["position_index"])
        segments = _detect_position_segments(centerline, analysis)
        _plot_maps(map_path, structure, centerline, analysis)
        _plot_position_summary(summary_path, structure, centerline, analysis, segments=segments)
        _plot_top_profiles(profiles_path, structure, analysis)
        summary[structure] = {
            "centerline_points": len(centerline),
            "map_plot": str(map_path),
            "pa_summary_plot": str(summary_path),
            "profiles_plot": str(profiles_path),
            "rise_ref_hu": analysis["rise_ref_hu"],
            "peak_ref_hu": analysis["peak_ref_hu"],
            "drop_ref_hu": analysis["drop_ref_hu"],
            "num_suspicious_cells": int(analysis["suspicious_map"].sum()),
            "suspected_segments_pa_cm": segments,
            "components": analysis["components"],
            "top_profiles": analysis["top_profiles"],
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D rib centerline cortical profile debug.")
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--structures", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expansion-factor", type=float, default=0.5)
    parser.add_argument("--centerline-step-mm", type=float, default=2.0)
    parser.add_argument("--angle-step-deg", type=float, default=10.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_debug(
        ct_path=args.ct,
        mask_dir=args.mask_dir,
        structures=[s.strip() for s in args.structures.split(",") if s.strip()],
        output_dir=args.output_dir,
        expansion_factor=args.expansion_factor,
        centerline_step_mm=args.centerline_step_mm,
        angle_step_deg=args.angle_step_deg,
    )


if __name__ == "__main__":
    main()
