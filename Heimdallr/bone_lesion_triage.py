import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VERTEBRAE = [
    "vertebrae_C1",
    "vertebrae_C2",
    "vertebrae_C3",
    "vertebrae_C4",
    "vertebrae_C5",
    "vertebrae_C6",
    "vertebrae_C7",
    "vertebrae_T1",
    "vertebrae_T2",
    "vertebrae_T3",
    "vertebrae_T4",
    "vertebrae_T5",
    "vertebrae_T6",
    "vertebrae_T7",
    "vertebrae_T8",
    "vertebrae_T9",
    "vertebrae_T10",
    "vertebrae_T11",
    "vertebrae_T12",
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
    "sacrum",
]
RIBS = [f"rib_left_{idx}" for idx in range(1, 13)] + [f"rib_right_{idx}" for idx in range(1, 13)]
PELVIS = ["hip_left", "hip_right"]
FEMURS = ["femur_left", "femur_right"]
DEFAULT_STRUCTURES = VERTEBRAE + RIBS + PELVIS + FEMURS

PAIRINGS = {
    **{f"rib_left_{idx}": f"rib_right_{idx}" for idx in range(1, 13)},
    **{f"rib_right_{idx}": f"rib_left_{idx}" for idx in range(1, 13)},
    "hip_left": "hip_right",
    "hip_right": "hip_left",
    "femur_left": "femur_right",
    "femur_right": "femur_left",
}

GROUP_THRESHOLDS = {
    "rib": {"dense_mm3": 25.0, "lytic_mm3": 120.0, "cortical_mm3": 80.0, "shell_drop": 150.0, "core_drop": 135.0},
    "vertebra": {"dense_mm3": 140.0, "lytic_mm3": 450.0, "cortical_mm3": 120.0, "shell_drop": 135.0, "core_drop": 110.0},
    "pelvis": {"dense_mm3": 300.0, "lytic_mm3": 1200.0, "cortical_mm3": 300.0, "shell_drop": 170.0, "core_drop": 120.0},
    "femur": {"dense_mm3": 500.0, "lytic_mm3": 1500.0, "cortical_mm3": 350.0, "shell_drop": 190.0, "core_drop": 160.0},
}
PROFILE_THRESHOLDS = {
    "rib": {"min_rise_hu": 140.0, "surface_drop_hu": 120.0, "max_steps": 8, "min_depth_mm": 1.8, "outside_steps": 5},
    "vertebra": {"min_rise_hu": 110.0, "surface_drop_hu": 90.0, "max_steps": 12, "min_depth_mm": 3.0, "outside_steps": 6},
}


@dataclass
class StructureStats:
    name: str
    group: str
    voxel_count: int
    volume_mm3: float
    hu_median: float
    hu_p95: float
    hu_p99: float
    cortical_shell_hu: float
    medullary_hu: float
    cortical_medullary_gap: float
    dense_core_threshold: float
    lytic_core_threshold: float
    cortical_defect_threshold: float
    dense_core_volume_mm3: float
    dense_core_total_mm3: float
    lytic_core_volume_mm3: float
    lytic_core_total_mm3: float
    cortical_defect_volume_mm3: float
    cortical_defect_total_mm3: float
    cortical_defect_extent_mm: float
    dense_core_fraction: float
    lytic_core_fraction: float
    cortical_defect_fraction: float
    shell_low_fraction: float
    axis_spread_mm: Tuple[float, float, float]
    z_center_mm: float


def _np() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("numpy is required for bone lesion triage. Install project dependencies first.") from exc
    return np


def _ndi() -> Any:
    try:
        from scipy import ndimage
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scipy is required for bone lesion triage. Install project dependencies first.") from exc
    return ndimage


def _load_nifti(path: Path) -> Tuple[Any, Any]:
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("nibabel is required to read NIfTI files. Install project dependencies first.") from exc
    img = nib.load(path)
    return img.get_fdata(), img


def _group_for_structure(name: str) -> str:
    if name in RIBS:
        return "rib"
    if name in FEMURS:
        return "femur"
    if name in PELVIS:
        return "pelvis"
    return "vertebra"


def _mask_path(mask_dir: Path, name: str) -> Path:
    return mask_dir / f"{name}.nii.gz"


def _iter_existing_masks(mask_dir: Path, structures: Sequence[str]) -> Iterable[Tuple[str, Path]]:
    for structure in structures:
        path = _mask_path(mask_dir, structure)
        if path.exists():
            yield structure, path


def _bounding_box(mask: Any, margin: int | Sequence[int] = 0) -> Tuple[slice, slice, slice]:
    np = _np()
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    if margin:
        if isinstance(margin, int):
            margin = np.array([margin, margin, margin], dtype=int)
        else:
            margin = np.asarray(list(margin), dtype=int)
        shape = np.array(mask.shape)
        mins = np.maximum(0, mins - margin)
        maxs = np.minimum(shape, maxs + margin)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))


def _largest_component(binary: Any) -> Tuple[int, int]:
    ndi = _ndi()
    np = _np()
    if not binary.any():
        return 0, 0
    labeled, n_components = ndi.label(binary)
    if n_components == 0:
        return 0, 0
    counts = np.bincount(labeled.ravel())[1:]
    return int(counts.max()), int(counts.sum())


def _largest_component_extent_mm(binary: Any, spacing: Sequence[float]) -> float:
    ndi = _ndi()
    np = _np()
    if not binary.any():
        return 0.0
    labeled, n_components = ndi.label(binary)
    if n_components == 0:
        return 0.0
    counts = np.bincount(labeled.ravel())[1:]
    largest_label = int(np.argmax(counts)) + 1
    coords = np.argwhere(labeled == largest_label)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = (maxs - mins + 1) * np.asarray(spacing)
    return round(float(extents.max()), 3)


def _sample_nearest(ct_crop: Any, point: Sequence[float]) -> Optional[float]:
    np = _np()
    idx = np.rint(np.asarray(point)).astype(int)
    if np.any(idx < 0) or np.any(idx >= np.asarray(ct_crop.shape)):
        return None
    return float(ct_crop[tuple(idx)])


def _sample_nearest_2d(ct_slice: Any, point: Sequence[float]) -> Optional[float]:
    np = _np()
    idx = np.rint(np.asarray(point)).astype(int)
    if np.any(idx < 0) or idx[0] >= ct_slice.shape[0] or idx[1] >= ct_slice.shape[1]:
        return None
    return float(ct_slice[tuple(idx)])


def _trace_profile_2d(
    ct_slice: Any,
    center_xy: Sequence[float],
    surface_xy: Sequence[float],
    spacing_xy: Sequence[float],
    inside_samples: int,
    outside_samples: int,
) -> Optional[Dict[str, Any]]:
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
    outside_step_vox = unit
    outside_points = [surface_xy + outside_step_vox * i for i in range(1, outside_samples + 1)]
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
            dist = float((idx - inside_samples + 1) * np.mean(spacing_xy))
        hu_values.append(sampled)
        distances.append(dist)

    if len(hu_values) < 6:
        return None

    surface_index = inside_samples - 1
    return {
        "hu_values": hu_values,
        "distances": distances,
        "path_length_mm": length_mm,
        "surface_index": surface_index,
    }


def _trace_profile(ct_crop: Any, dist_mm: Any, mask_crop: Any, start: Tuple[int, int, int], max_steps: int, outside_steps: int) -> Optional[Dict[str, Any]]:
    np = _np()

    shape = mask_crop.shape
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == dy == dz == 0)]
    current = tuple(int(v) for v in start)
    path = [current]
    current_dist = float(dist_mm[current])

    for _ in range(max_steps):
        best = current
        best_dist = current_dist
        for dx, dy, dz in offsets:
            nx, ny, nz = current[0] + dx, current[1] + dy, current[2] + dz
            if nx < 0 or ny < 0 or nz < 0 or nx >= shape[0] or ny >= shape[1] or nz >= shape[2]:
                continue
            if not mask_crop[nx, ny, nz]:
                continue
            cand_dist = float(dist_mm[nx, ny, nz])
            if cand_dist > best_dist + 0.05:
                best = (nx, ny, nz)
                best_dist = cand_dist
        if best == current:
            break
        current = best
        current_dist = best_dist
        path.append(current)

    if len(path) < 2:
        return None

    inward_vec = np.asarray(path[1], dtype=float) - np.asarray(path[0], dtype=float)
    norm = np.linalg.norm(inward_vec)
    if norm == 0:
        return None
    inward_unit = inward_vec / norm

    outside_points = []
    for step in range(1, outside_steps + 1):
        outside_point = np.asarray(path[0], dtype=float) - inward_unit * step
        outside_points.append(outside_point)

    full_points = list(reversed(path)) + [np.asarray(path[0], dtype=float)] + outside_points
    hu_values = []
    distances = []
    for idx, point in enumerate(full_points):
        sampled = _sample_nearest(ct_crop, point)
        if sampled is None:
            continue
        hu_values.append(sampled)
        distances.append(float(idx - (len(path) - 1)))

    if len(hu_values) < 4:
        return None

    return {
        "hu_values": hu_values,
        "distances": distances,
        "path_length_mm": float(dist_mm[start]),
        "surface_index": len(path) - 1,
    }


def _axial_rib_profile_failure(ct_crop: Any, mask_crop: Any, spacing: Sequence[float]) -> Tuple[Any, List[Dict[str, Any]]]:
    np = _np()
    ndi = _ndi()

    params = PROFILE_THRESHOLDS["rib"]
    failure = np.zeros_like(mask_crop, dtype=bool)
    failed_profiles = []
    candidate_profiles = []

    for z in range(mask_crop.shape[2]):
        mask_slice = mask_crop[:, :, z]
        if mask_slice.sum() < 20:
            continue
        ct_slice = ct_crop[:, :, z]
        dist_2d = ndi.distance_transform_edt(mask_slice, sampling=spacing[:2])
        center_index = np.unravel_index(int(np.argmax(dist_2d)), dist_2d.shape)
        center_depth_mm = float(dist_2d[center_index])
        if center_depth_mm < params["min_depth_mm"]:
            continue

        surface_2d = mask_slice & ~ndi.binary_erosion(mask_slice, iterations=1, border_value=0)
        if not surface_2d.any():
            continue
        surface_vals = ct_slice[surface_2d]
        preserved_surface_hu = float(np.percentile(surface_vals, 75))
        high_floor_hu = preserved_surface_hu - params["surface_drop_hu"]

        coords = np.argwhere(surface_2d)
        step = max(1, len(coords) // 40)
        for x, y in coords[::step]:
            profile = _trace_profile_2d(
                ct_slice=ct_slice,
                center_xy=center_index,
                surface_xy=(x, y),
                spacing_xy=spacing[:2],
                inside_samples=10,
                outside_samples=params["outside_steps"],
            )
            if profile is None:
                continue

            hu_profile = np.asarray(profile["hu_values"], dtype=float)
            surface_index = int(profile["surface_index"])
            outer_window_start = max(0, surface_index - 1)
            outer_window_end = min(len(hu_profile), surface_index + 2)
            outer_peak = float(np.max(hu_profile[outer_window_start:outer_window_end]))
            deep_section = hu_profile[: max(2, surface_index - 1)]
            outside_section = hu_profile[min(surface_index + 1, len(hu_profile)) :]
            if deep_section.size == 0 or outside_section.size == 0:
                continue
            deep_median = float(np.median(deep_section))
            outside_median = float(np.median(outside_section))
            rise = outer_peak - deep_median
            peak_gap = outer_peak - high_floor_hu

            profile_record = {
                "surface_voxel": [int(x), int(y), int(z)],
                "distances": profile["distances"],
                "hu_values": profile["hu_values"],
                "outer_peak_hu": round(outer_peak, 3),
                "deep_median_hu": round(deep_median, 3),
                "outside_median_hu": round(outside_median, 3),
                "rise_hu": round(rise, 3),
                "path_length_mm": round(float(profile["path_length_mm"]), 3),
                "peak_gap_hu": round(peak_gap, 3),
            }
            candidate_profiles.append(profile_record)

            if outer_peak < high_floor_hu and rise < params["min_rise_hu"] and outer_peak < outside_median + params["surface_drop_hu"] * 0.5:
                failure[x, y, z] = True
                failed_profiles.append(profile_record)

    failed_profiles.sort(key=lambda item: (item["rise_hu"], item["peak_gap_hu"], -item["path_length_mm"]))
    candidate_profiles.sort(key=lambda item: (abs(item["peak_gap_hu"]) + abs(item["rise_hu"]), -item["path_length_mm"]))
    selected = failed_profiles[:8]
    if len(selected) < 8:
        seen = {tuple(item["surface_voxel"]) for item in selected}
        for item in candidate_profiles:
            key = tuple(item["surface_voxel"])
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            if len(selected) >= 8:
                break

    return failure, selected


def _ascending_profile_failure(ct_crop: Any, mask_crop: Any, group: str, spacing: Sequence[float]) -> Tuple[Any, List[Dict[str, Any]]]:
    np = _np()
    ndi = _ndi()

    if group not in PROFILE_THRESHOLDS:
        return np.zeros_like(mask_crop, dtype=bool), []
    if group == "rib":
        return _axial_rib_profile_failure(ct_crop, mask_crop, spacing)

    params = PROFILE_THRESHOLDS[group]
    surface = mask_crop & ~ndi.binary_erosion(mask_crop, iterations=1, border_value=0)
    if not surface.any():
        return np.zeros_like(mask_crop, dtype=bool), []

    dist_mm = ndi.distance_transform_edt(mask_crop, sampling=spacing)
    surface_vals = ct_crop[surface]
    if surface_vals.size == 0:
        return np.zeros_like(mask_crop, dtype=bool), []
    preserved_surface_hu = float(np.percentile(surface_vals, 75))
    high_floor_hu = preserved_surface_hu - params["surface_drop_hu"]
    failure = np.zeros_like(mask_crop, dtype=bool)
    failed_profiles = []
    candidate_profiles = []

    for x, y, z in np.argwhere(surface):
        start = (int(x), int(y), int(z))
        if float(dist_mm[start]) < params["min_depth_mm"]:
            continue
        profile = _trace_profile(
            ct_crop=ct_crop,
            dist_mm=dist_mm,
            mask_crop=mask_crop,
            start=start,
            max_steps=params["max_steps"],
            outside_steps=params["outside_steps"],
        )
        if profile is None:
            continue
        hu_profile = np.asarray(profile["hu_values"], dtype=float)
        surface_index = int(profile["surface_index"])
        outer_window_start = max(0, surface_index - 1)
        outer_window_end = min(len(hu_profile), surface_index + 2)
        outer_peak = float(np.max(hu_profile[outer_window_start:outer_window_end]))
        deep_section = hu_profile[: max(2, surface_index - 1)]
        outside_section = hu_profile[min(surface_index + 1, len(hu_profile)) :]
        if deep_section.size == 0 or outside_section.size == 0:
            continue
        deep_median = float(np.median(deep_section))
        outside_median = float(np.median(outside_section))
        rise = outer_peak - deep_median

        profile_record = {
            "surface_voxel": [x, y, z],
            "distances": profile["distances"],
            "hu_values": profile["hu_values"],
            "outer_peak_hu": round(outer_peak, 3),
            "deep_median_hu": round(deep_median, 3),
            "outside_median_hu": round(outside_median, 3),
            "rise_hu": round(rise, 3),
            "path_length_mm": round(float(profile["path_length_mm"]), 3),
            "peak_gap_hu": round(outer_peak - high_floor_hu, 3),
        }
        candidate_profiles.append(profile_record)

        if outer_peak < high_floor_hu and rise < params["min_rise_hu"] and outer_peak < outside_median + params["surface_drop_hu"] * 0.5:
            failure[x, y, z] = True
            failed_profiles.append(profile_record)

    failed_profiles.sort(key=lambda item: (item["rise_hu"], item["peak_gap_hu"], -item["path_length_mm"]))
    candidate_profiles.sort(key=lambda item: (abs(item["peak_gap_hu"]) + abs(item["rise_hu"]), -item["path_length_mm"]))
    selected = failed_profiles[:8]
    if len(selected) < 8:
        seen = {tuple(item["surface_voxel"]) for item in selected}
        for item in candidate_profiles:
            key = tuple(item["surface_voxel"])
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
            if len(selected) >= 8:
                break

    return failure, selected


def _axis_spread_mm(mask: Any, spacing: Sequence[float]) -> Tuple[float, float, float]:
    np = _np()
    coords = np.argwhere(mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return tuple(round(float((maxs[idx] - mins[idx] + 1) * spacing[idx]), 3) for idx in range(3))


def _rib_segment_analysis(ct: Any, mask: Any, affine: Any, spacing: Sequence[float]) -> Dict[str, Any]:
    from heimdallr import rib_centerline_profile_debug as rib_debug

    centerline = rib_debug._estimate_centerline(mask, spacing_xyz=spacing, step_mm=2.0)
    if not centerline:
        return {"segments": [], "centerline_points": 0, "analysis": None}
    centerline = rib_debug._orient_centerline_posterior_to_anterior(centerline, affine)
    analysis = rib_debug._analyze_rib(
        ct=ct,
        mask=mask,
        spacing_xyz=spacing,
        centerline=centerline,
        expansion_factor=0.5,
        angle_step_deg=10.0,
    )
    segments = rib_debug._detect_position_segments(centerline, analysis)
    return {
        "segments": segments,
        "centerline_points": len(centerline),
        "centerline": centerline,
        "analysis": analysis,
    }


def _structure_payload(ct: Any, mask: Any, affine: Any, spacing: Sequence[float], name: str, high_hu: int, low_hu: int) -> Dict[str, Any]:
    np = _np()
    ndi = _ndi()

    margin_vox = [max(2, int(round(8.0 / float(sp)))) for sp in spacing]
    bbox = _bounding_box(mask, margin=margin_vox)
    ct_crop = ct[bbox]
    mask_crop = mask[bbox]
    voxel_volume = float(np.prod(spacing))
    voxel_count = int(mask_crop.sum())
    volume_mm3 = voxel_count * voxel_volume
    group = _group_for_structure(name)
    thresholds = GROUP_THRESHOLDS[group]

    erosion_iterations = 1 if group == "rib" else 2
    eroded = ndi.binary_erosion(mask_crop, iterations=erosion_iterations, border_value=0)
    if eroded.sum() == 0:
        eroded = ndi.binary_erosion(mask_crop, iterations=1, border_value=0)
    core = eroded if eroded.sum() > 0 else mask_crop
    shell = mask_crop & ~core

    voxels = ct_crop[mask_crop]
    shell_voxels = ct_crop[shell]
    core_voxels = ct_crop[core]
    shell_hu = float(np.median(shell_voxels)) if shell_voxels.size else float(np.median(voxels))
    core_hu = float(np.median(core_voxels)) if core_voxels.size else float(np.median(voxels))
    preserved_cortex_hu = float(np.percentile(shell_voxels, 75)) if shell_voxels.size else shell_hu

    dense_core_threshold = max(high_hu, float(np.percentile(core_voxels, 99.5)) if core_voxels.size else high_hu, core_hu + 320.0)
    lytic_core_threshold = max(-250.0, min(float(low_hu), core_hu - thresholds["core_drop"]))
    cortical_defect_threshold = max(-250.0, min(preserved_cortex_hu - thresholds["shell_drop"], core_hu - 35.0))

    dense_core = core & (ct_crop >= dense_core_threshold)
    lytic_core = core & (ct_crop <= lytic_core_threshold)
    cortical_defect = shell & (ct_crop <= cortical_defect_threshold)
    if group in PROFILE_THRESHOLDS:
        profile_failure, attenuation_profiles = _ascending_profile_failure(ct_crop, mask_crop, group, spacing)
        cortical_defect = profile_failure
    else:
        profile_failure = np.zeros_like(mask_crop, dtype=bool)
        attenuation_profiles = []

    dense_max, dense_total = _largest_component(dense_core)
    lytic_max, lytic_total = _largest_component(lytic_core)
    cortical_max, cortical_total = _largest_component(cortical_defect)
    cortical_extent_mm = _largest_component_extent_mm(cortical_defect, spacing)

    z_coords = np.argwhere(mask_crop)[:, 2]
    stats = StructureStats(
        name=name,
        group=group,
        voxel_count=voxel_count,
        volume_mm3=round(volume_mm3, 3),
        hu_median=round(float(np.median(voxels)), 3),
        hu_p95=round(float(np.percentile(voxels, 95)), 3),
        hu_p99=round(float(np.percentile(voxels, 99)), 3),
        cortical_shell_hu=round(shell_hu, 3),
        medullary_hu=round(core_hu, 3),
        cortical_medullary_gap=round(shell_hu - core_hu, 3),
        dense_core_threshold=round(float(dense_core_threshold), 3),
        lytic_core_threshold=round(float(lytic_core_threshold), 3),
        cortical_defect_threshold=round(float(cortical_defect_threshold), 3),
        dense_core_volume_mm3=round(dense_max * voxel_volume, 3),
        dense_core_total_mm3=round(dense_total * voxel_volume, 3),
        lytic_core_volume_mm3=round(lytic_max * voxel_volume, 3),
        lytic_core_total_mm3=round(lytic_total * voxel_volume, 3),
        cortical_defect_volume_mm3=round(cortical_max * voxel_volume, 3),
        cortical_defect_total_mm3=round(cortical_total * voxel_volume, 3),
        cortical_defect_extent_mm=cortical_extent_mm,
        dense_core_fraction=round(float(dense_total / max(core.sum(), 1)), 5),
        lytic_core_fraction=round(float(lytic_total / max(core.sum(), 1)), 5),
        cortical_defect_fraction=round(float(cortical_total / max(shell.sum(), 1)), 5),
        shell_low_fraction=round(float((shell & (ct_crop <= shell_hu - 90.0)).sum() / max(shell.sum(), 1)), 5),
        axis_spread_mm=_axis_spread_mm(mask_crop, spacing),
        z_center_mm=round(float(z_coords.mean() * spacing[2]), 3),
    )

    rib_segment_analysis = {"segments": [], "centerline_points": 0, "analysis": None}
    if group == "rib":
        rib_segment_analysis = _rib_segment_analysis(ct=ct, mask=mask, affine=affine, spacing=spacing)

    return {
        "stats": stats,
        "bbox": bbox,
        "mask": mask_crop,
        "dense_core": dense_core,
        "lytic_core": lytic_core,
        "cortical_defect": cortical_defect,
        "profile_failure": profile_failure,
        "attenuation_profiles": attenuation_profiles,
        "rib_segment_analysis": rib_segment_analysis,
        "ct_crop": ct_crop,
    }


def _vertebra_neighbor_delta(name: str, stats_map: Dict[str, StructureStats], field: str) -> Optional[float]:
    if name not in VERTEBRAE:
        return None
    idx = VERTEBRAE.index(name)
    neighbor_values = []
    if idx > 0 and VERTEBRAE[idx - 1] in stats_map:
        neighbor_values.append(getattr(stats_map[VERTEBRAE[idx - 1]], field))
    if idx + 1 < len(VERTEBRAE) and VERTEBRAE[idx + 1] in stats_map:
        neighbor_values.append(getattr(stats_map[VERTEBRAE[idx + 1]], field))
    if not neighbor_values:
        return None
    np = _np()
    return round(float(getattr(stats_map[name], field) - np.median(neighbor_values)), 5)


def _pair_delta(name: str, stats_map: Dict[str, StructureStats], field: str) -> Optional[float]:
    peer = PAIRINGS.get(name)
    if not peer or peer not in stats_map:
        return None
    return round(float(getattr(stats_map[name], field) - getattr(stats_map[peer], field)), 5)


def _score_structure(stats: StructureStats, stats_map: Dict[str, StructureStats], payload: Dict[str, Any]) -> Dict[str, object]:
    thresholds = GROUP_THRESHOLDS[stats.group]
    pair_hu_delta = _pair_delta(stats.name, stats_map, "hu_median")
    pair_dense_delta = _pair_delta(stats.name, stats_map, "dense_core_fraction")
    pair_cortical_delta = _pair_delta(stats.name, stats_map, "cortical_defect_fraction")
    pair_volume_delta = _pair_delta(stats.name, stats_map, "volume_mm3")
    neighbor_hu_delta = _vertebra_neighbor_delta(stats.name, stats_map, "hu_median")
    neighbor_cortical_delta = _vertebra_neighbor_delta(stats.name, stats_map, "cortical_defect_fraction")

    score = 0.0
    reasons: List[str] = []
    context_support = False
    rib_segments = payload.get("rib_segment_analysis", {}).get("segments", []) if stats.group == "rib" else []

    if stats.dense_core_volume_mm3 >= thresholds["dense_mm3"]:
        score += 2.0
        reasons.append(f"foco hiperdenso focal ({stats.dense_core_volume_mm3:.1f} mm3)")
    elif stats.dense_core_volume_mm3 >= thresholds["dense_mm3"] * 0.55:
        score += 1.0
        reasons.append(f"foco hiperdenso menor ({stats.dense_core_volume_mm3:.1f} mm3)")

    if stats.cortical_defect_volume_mm3 >= thresholds["cortical_mm3"]:
        score += 2.5
        reasons.append(f"defeito cortical focal ({stats.cortical_defect_volume_mm3:.1f} mm3)")
    elif stats.cortical_defect_volume_mm3 >= thresholds["cortical_mm3"] * 0.5:
        score += 1.0
        reasons.append(f"defeito cortical menor ({stats.cortical_defect_volume_mm3:.1f} mm3)")
    if stats.cortical_defect_extent_mm >= 6.0:
        score += 1.5
        reasons.append(f"interrupcao cortical >= 6 mm ({stats.cortical_defect_extent_mm:.1f} mm)")
    elif stats.cortical_defect_extent_mm >= 4.0:
        score += 0.5
        reasons.append(f"interrupcao cortical menor ({stats.cortical_defect_extent_mm:.1f} mm)")

    if stats.group != "femur":
        if stats.lytic_core_volume_mm3 >= thresholds["lytic_mm3"]:
            score += 1.5
            reasons.append(f"componente litico focal ({stats.lytic_core_volume_mm3:.1f} mm3)")
        elif stats.lytic_core_volume_mm3 >= thresholds["lytic_mm3"] * 0.5 and stats.group != "rib":
            score += 0.75
            reasons.append(f"componente litico menor ({stats.lytic_core_volume_mm3:.1f} mm3)")

    if stats.group == "rib" and stats.cortical_defect_volume_mm3 >= thresholds["cortical_mm3"] * 1.5 and stats.lytic_core_volume_mm3 >= thresholds["lytic_mm3"] * 3.0:
        score += 1.5
        reasons.append("combinacao costal de defeito cortical e rarefacao focal")
    if stats.group == "rib" and rib_segments:
        core_segments = [seg for seg in rib_segments if seg.get("segment_type") == "core_suspicious"]
        anterior_segments = [seg for seg in rib_segments if seg.get("segment_type") == "anterior_low_confidence"]
        if core_segments:
            best_core = max(core_segments, key=lambda seg: (seg["peak_drop_score"], seg["length_cm"]))
            score += 2.0
            context_support = True
            reasons.append(
                f"segmento costal 3D posterior-anterior ({best_core['start_cm_pa']:.1f}-{best_core['end_cm_pa']:.1f} cm)"
            )
            if best_core["length_cm"] >= 1.0:
                score += 0.75
                reasons.append(f"segmento costal extenso ({best_core['length_cm']:.1f} cm)")
        if anterior_segments:
            best_ant = max(anterior_segments, key=lambda seg: (seg["peak_drop_score"], seg["length_cm"]))
            score += 0.5
            reasons.append(
                f"segmento anterior de baixa confianca ({best_ant['start_cm_pa']:.1f}-{best_ant['end_cm_pa']:.1f} cm)"
            )
    if stats.group == "vertebra" and (stats.cortical_defect_volume_mm3 >= thresholds["cortical_mm3"] * 3.0 or stats.cortical_defect_extent_mm >= 6.0):
        score += 1.5
        reasons.append("defeito cortical vertebral volumoso")

    if pair_cortical_delta is not None and abs(pair_cortical_delta) >= 0.018:
        score += 1.5
        context_support = True
        reasons.append(f"assimetria cortical contralateral ({pair_cortical_delta:+.3f})")
    if pair_dense_delta is not None and abs(pair_dense_delta) >= 0.006:
        score += 1.0
        context_support = True
        reasons.append(f"assimetria de densidade focal ({pair_dense_delta:+.3f})")
    if pair_hu_delta is not None and abs(pair_hu_delta) >= 45:
        score += 0.75
        context_support = True
        reasons.append(f"assimetria global de HU ({pair_hu_delta:+.1f})")
    if pair_volume_delta is not None:
        peer = PAIRINGS.get(stats.name)
        peer_volume = stats_map[peer].volume_mm3 if peer in stats_map else None
        if peer_volume and abs(pair_volume_delta) / max(peer_volume, 1.0) >= 0.12:
            score += 0.75
            context_support = True
            reasons.append(f"assimetria volumetrica ({pair_volume_delta:+.1f} mm3)")

    if neighbor_hu_delta is not None and abs(neighbor_hu_delta) >= 55:
        score += 1.0
        context_support = True
        reasons.append(f"desvio em relacao a vertebras vizinhas ({neighbor_hu_delta:+.1f} HU)")
    if neighbor_cortical_delta is not None and abs(neighbor_cortical_delta) >= 0.012:
        score += 1.0
        context_support = True
        reasons.append(f"desvio cortical em relacao a vizinhas ({neighbor_cortical_delta:+.3f})")

    if stats.group == "femur" and not context_support:
        score *= 0.35
        reasons.append("rebaixado por ausencia de assimetria convincente em femur")
    if stats.group == "rib" and not context_support:
        score *= 0.65
        reasons.append("rebaixado por ausencia de assimetria convincente em costela")
    if stats.group == "femur":
        peer = PAIRINGS.get(stats.name)
        if peer in stats_map:
            peer_stats = stats_map[peer]
            peer_volume = max(peer_stats.volume_mm3, 1.0)
            bilateral_symmetric = (
                abs(pair_hu_delta or 0.0) < 20.0
                and abs(pair_dense_delta or 0.0) < 0.002
                and abs(pair_volume_delta or 0.0) / peer_volume < 0.15
                and stats.cortical_defect_fraction > 0.16
                and peer_stats.cortical_defect_fraction > 0.16
            )
            if bilateral_symmetric:
                score *= 0.4
                reasons.append("rebaixado por padrao femoral bilateral simetrico")

    suspicion = "low"
    if score >= 5.5 and (context_support or stats.cortical_defect_volume_mm3 >= thresholds["cortical_mm3"] * 1.6):
        suspicion = "high"
    elif score >= 3.0:
        suspicion = "medium"

    pattern = "indeterminate"
    if stats.dense_core_volume_mm3 >= max(thresholds["dense_mm3"], stats.cortical_defect_volume_mm3 * 1.5):
        pattern = "sclerotic_or_blastic"
    elif stats.cortical_defect_volume_mm3 >= max(thresholds["cortical_mm3"], stats.dense_core_volume_mm3 * 1.25):
        pattern = "cortical_destruction_or_lytic"
    elif stats.dense_core_volume_mm3 > 0 and stats.cortical_defect_volume_mm3 > 0:
        pattern = "mixed"

    return {
        "structure": stats.name,
        "group": stats.group,
        "suspicion": suspicion,
        "score": round(score, 3),
        "pattern": pattern,
        "reasons": reasons,
        "metrics": {
            "volume_mm3": stats.volume_mm3,
            "hu_median": stats.hu_median,
            "hu_p95": stats.hu_p95,
            "hu_p99": stats.hu_p99,
            "cortical_shell_hu": stats.cortical_shell_hu,
            "medullary_hu": stats.medullary_hu,
            "cortical_medullary_gap": stats.cortical_medullary_gap,
            "dense_core_threshold": stats.dense_core_threshold,
            "lytic_core_threshold": stats.lytic_core_threshold,
            "cortical_defect_threshold": stats.cortical_defect_threshold,
            "dense_core_volume_mm3": stats.dense_core_volume_mm3,
            "dense_core_total_mm3": stats.dense_core_total_mm3,
            "lytic_core_volume_mm3": stats.lytic_core_volume_mm3,
            "lytic_core_total_mm3": stats.lytic_core_total_mm3,
            "cortical_defect_volume_mm3": stats.cortical_defect_volume_mm3,
            "cortical_defect_total_mm3": stats.cortical_defect_total_mm3,
            "cortical_defect_extent_mm": stats.cortical_defect_extent_mm,
            "dense_core_fraction": stats.dense_core_fraction,
            "lytic_core_fraction": stats.lytic_core_fraction,
            "cortical_defect_fraction": stats.cortical_defect_fraction,
            "shell_low_fraction": stats.shell_low_fraction,
            "axis_spread_mm": stats.axis_spread_mm,
            "pair_delta_hu": pair_hu_delta,
            "pair_delta_dense_fraction": pair_dense_delta,
            "pair_delta_cortical_fraction": pair_cortical_delta,
            "pair_delta_volume_mm3": pair_volume_delta,
            "vertebra_neighbor_delta_hu": neighbor_hu_delta,
            "vertebra_neighbor_delta_cortical_fraction": neighbor_cortical_delta,
            "z_center_mm": stats.z_center_mm,
            "rib_centerline_points": payload.get("rib_segment_analysis", {}).get("centerline_points"),
            "rib_suspected_segments_pa_cm": rib_segments,
        },
    }


def _render_structure(path: Path, name: str, payload: Dict[str, Any], finding: Dict[str, Any]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("matplotlib is required to render review PNGs.") from exc

    profiles = payload.get("attenuation_profiles", [])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    fig.suptitle(f"{name} | {finding['suspicion']} | score={finding['score']:.2f} | {finding['pattern']}", fontsize=11)

    if not profiles:
        for ax in axes:
            ax.text(0.5, 0.5, "No attenuation profiles saved", ha="center", va="center")
            ax.axis("off")
    else:
        for ax, profile in zip(axes, profiles[:4]):
            ax.plot(profile["distances"], profile["hu_values"], color="black", linewidth=1.6)
            ax.axvline(0.0, color="orange", linestyle="--", linewidth=1.0)
            ax.axhline(profile["deep_median_hu"], color="royalblue", linestyle=":", linewidth=1.0, label="deep median")
            ax.axhline(profile["outside_median_hu"], color="seagreen", linestyle=":", linewidth=1.0, label="outside median")
            ax.axhline(profile["outer_peak_hu"], color="firebrick", linestyle=":", linewidth=1.0, label="outer peak")
            ax.set_title(f"rise={profile['rise_hu']:.1f} HU | gap={profile['peak_gap_hu']:.1f} | len={profile['path_length_mm']:.1f} mm")
            ax.set_xlabel("center -> surface -> outside")
            ax.set_ylabel("HU")
            ax.grid(alpha=0.25)
        for ax in axes[len(profiles[:4]) :]:
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def analyze_bone_lesions(
    ct_path: Path,
    mask_dir: Path,
    structures: Sequence[str] = DEFAULT_STRUCTURES,
    high_hu: int = 900,
    low_hu: int = 80,
    render_dir: Optional[Path] = None,
    render_top_k: int = 10,
) -> Dict[str, object]:
    np = _np()
    ct_data, ct_img = _load_nifti(ct_path)
    ct = np.asarray(ct_data, dtype=np.int16)
    spacing = ct_img.header.get_zooms()[:3]
    affine = ct_img.affine

    payloads: Dict[str, Dict[str, Any]] = {}
    stats_map: Dict[str, StructureStats] = {}
    missing = []
    for name, path in _iter_existing_masks(mask_dir, structures):
        mask_data, _ = _load_nifti(path)
        mask = mask_data > 0.5
        if mask.sum() == 0:
            continue
        payload = _structure_payload(ct, mask, affine, spacing, name, high_hu=high_hu, low_hu=low_hu)
        payloads[name] = payload
        stats_map[name] = payload["stats"]

    for name in structures:
        if name not in stats_map:
            missing.append(name)

    findings = [_score_structure(stats_map[name], stats_map, payloads[name]) for name in stats_map]
    findings.sort(key=lambda item: (-item["score"], item["structure"]))

    render_paths = {}
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)
        for item in findings[: max(render_top_k, 0)]:
            png_path = render_dir / f"{item['structure']}.png"
            _render_structure(png_path, item["structure"], payloads[item["structure"]], item)
            item["review_png"] = str(png_path)
            render_paths[item["structure"]] = str(png_path)

    summary = {
        "structures_analyzed": len(stats_map),
        "high_suspicion_count": sum(item["suspicion"] == "high" for item in findings),
        "medium_suspicion_count": sum(item["suspicion"] == "medium" for item in findings),
        "top_structures": [item["structure"] for item in findings[:10]],
        "thresholds": {
            "high_hu": high_hu,
            "low_hu": low_hu,
        },
    }

    return {
        "ct_path": str(ct_path),
        "mask_dir": str(mask_dir),
        "summary": summary,
        "findings": findings,
        "missing_masks": missing,
        "rendered_pngs": render_paths,
        "disclaimer": [
            "Heuristica sem retreino: usar para triagem e priorizacao, nao para confirmacao diagnostica.",
            "Destruicao cortical grosseira tende a ser mais visivel do que infiltracao medular sutil, mas ainda pode ser subestimada por segmentacao anatomica imperfeita.",
            "Artefatos, fraturas, esclerose degenerativa, ilhas osseas e hardware podem gerar falsos positivos.",
        ],
    }


def _parse_structures(raw: Optional[str]) -> Sequence[str]:
    if not raw:
        return DEFAULT_STRUCTURES
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Heuristic triage for large/blastic bone metastasis candidates using TotalSegmentator masks.")
    parser.add_argument("--ct", required=True, type=Path, help="Path to CT NIfTI in HU.")
    parser.add_argument("--mask-dir", required=True, type=Path, help="Directory containing individual TotalSegmentator masks.")
    parser.add_argument("--output", required=True, type=Path, help="Path to output JSON report.")
    parser.add_argument("--structures", type=str, default=None, help="Comma-separated list of structures to analyze. Defaults to vertebrae, ribs and pelvis-related masks.")
    parser.add_argument("--high-hu", type=int, default=900, help="HU threshold for dense/blastic candidate clusters.")
    parser.add_argument("--low-hu", type=int, default=80, help="HU threshold for lytic/cortical-destruction candidate clusters.")
    parser.add_argument("--render-dir", type=Path, default=None, help="Optional directory to save PNG review slices for the top findings.")
    parser.add_argument("--render-top-k", type=int, default=10, help="How many top findings to render as PNG review slices.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = analyze_bone_lesions(
        ct_path=args.ct,
        mask_dir=args.mask_dir,
        structures=_parse_structures(args.structures),
        high_hu=args.high_hu,
        low_hu=args.low_hu,
        render_dir=args.render_dir,
        render_top_k=args.render_top_k,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
