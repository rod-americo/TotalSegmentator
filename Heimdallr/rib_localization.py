import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence


RIBS = [f"rib_left_{idx}" for idx in range(1, 13)] + [f"rib_right_{idx}" for idx in range(1, 13)]


def _parse_structures(raw: str | None) -> List[str]:
    if not raw:
        return list(RIBS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_localization(ct_path: Path, mask_dir: Path, structures: Sequence[str]) -> Dict[str, Any]:
    import numpy as np
    from heimdallr import rib_centerline_profile_debug as rib_debug

    ct_data, ct_img = rib_debug._load_nifti(ct_path)
    ct = np.asarray(ct_data, dtype=np.float32)
    spacing = ct_img.header.get_zooms()[:3]
    affine = ct_img.affine

    findings: List[Dict[str, Any]] = []
    missing: List[str] = []
    started = time.perf_counter()

    for structure in structures:
        mask_path = mask_dir / f"{structure}.nii.gz"
        if not mask_path.exists():
            missing.append(structure)
            continue
        mask_data, _ = rib_debug._load_nifti(mask_path)
        mask = mask_data > 0.5
        if int(mask.sum()) == 0:
            missing.append(structure)
            continue

        t0 = time.perf_counter()
        centerline = rib_debug._estimate_centerline(mask, spacing_xyz=spacing, step_mm=2.0)
        if not centerline:
            findings.append(
                {
                    "structure": structure,
                    "status": "failed_centerline",
                    "elapsed_s": round(time.perf_counter() - t0, 3),
                    "segments": [],
                }
            )
            continue
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
        findings.append(
            {
                "structure": structure,
                "status": "ok",
                "elapsed_s": round(time.perf_counter() - t0, 3),
                "centerline_points": len(centerline),
                "segments": segments,
            }
        )

    findings.sort(key=lambda item: item["structure"])
    return {
        "ct_path": str(ct_path),
        "mask_dir": str(mask_dir),
        "structures": list(structures),
        "total_elapsed_s": round(time.perf_counter() - started, 3),
        "findings": findings,
        "missing_masks": missing,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rib-only localization using 3D posterior-anterior segment analysis.")
    parser.add_argument("--ct", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--structures", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_localization(
        ct_path=args.ct,
        mask_dir=args.mask_dir,
        structures=_parse_structures(args.structures),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
