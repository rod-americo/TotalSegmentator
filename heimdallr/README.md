# Heimdallr

Heuristic triage utilities that run on top of TotalSegmentator outputs without retraining.

## DICOM to NIfTI

The `dicom_to_nifti.py` script converts one DICOM series into a `.nii.gz` volume.

### Usage

```bash
python3 -m heimdallr.dicom_to_nifti \
  --input /path/to/dicom_series \
  --output /path/to/ct.nii.gz
```

You can also pass a `.zip` as input. In that case, set `--tmp-dir` if you want to control where the archive is extracted.

## Bone lesion triage

The `bone_lesion_triage.py` script ranks vertebrae, ribs and pelvis-related structures by suspicion for large bone lesions that are already structurally visible on CT, especially:

- blastic / sclerotic foci
- large lytic defects
- approximate cortical disruption patterns

It uses only the CT volume plus existing TotalSegmentator masks.

### Usage

```bash
python -m heimdallr.bone_lesion_triage \
  --ct /path/to/ct.nii.gz \
  --mask-dir /path/to/segmentations \
  --output /path/to/heimdallr_bone_report.json
```

Optional parameters:

- `--structures`: comma-separated mask names
- `--high-hu`: dense cluster threshold, default `900`
- `--low-hu`: low-density cluster threshold, default `80`
- `--render-dir`: output directory for PNG attenuation-profile graphs
- `--render-top-k`: how many ranked structures to render

### Output

The JSON report contains:

- `summary`: counts and top-ranked structures
- `findings`: one entry per analyzed structure with score, suspicion level, pattern and metrics
- `missing_masks`: structures expected but not found

If `--render-dir` is used, the script saves attenuation-profile graphs sampled from suspicious center-to-surface trajectories.

This is a triage heuristic, not a diagnostic model.

## Kidney stone triage

The `kidney_stone_triage.py` script detects dense intrarenal connected components inside the `kidney_left` and `kidney_right` masks from TotalSegmentator.

It uses only the CT volume plus the renal masks and defaults to a `130 HU` threshold, which is the working threshold adopted here because it is the most recurrent cutoff in the kidney-stone CT literature.

### Usage

```bash
python -m heimdallr.kidney_stone_triage \
  --ct /path/to/ct.nii.gz \
  --mask-dir /path/to/segmentations \
  --output /path/to/kidney_stone_report.json
```

Optional parameters:

- `--masks`: comma-separated mask names, default `kidney_left,kidney_right`
- `--threshold-hu`: dense voxel threshold, default `130`
- `--min-voxels`: minimum connected voxels to retain a component, default `3`
- `--min-volume-mm3`: optional minimum component volume in `mm3`; a component is kept if it passes `min-voxels` or `min-volume-mm3`
- `--render-dir`: optional directory for axial/coronal PNG overlays per component

### Output

The JSON report contains:

- `summary`: number of kidneys analyzed, connected components and total dense volume
- `kidneys`: one entry per kidney mask with renal summary and per-component metrics
- `missing_masks`: masks requested but not found

Per-component metrics include:

- voxel count
- volume in `mm3` and `mL`
- mean HU and max HU
- centroid in voxel and physical coordinates
- largest 3D axis and PCA-based principal axes
- axial and coronal overlay PNG paths

The method summary is documented in `KIDNEY_STONE_METHOD.md`.

## Experimental rib tools

These scripts are kept as `debug / experimental` tools for directed investigation of rib findings.

They are useful for:

- exploring a known suspicious rib
- measuring posterior-to-anterior segments in cm
- comparing core rib segments against the anterior costochondral transition

They are **not** good enough for automatic rib metastasis detection or reliable rib ranking across all ribs.

### `rib_centerline_profile_debug.py`

This is a 3D proof of concept for ribs:

- estimate a rib centerline from the 3D mask
- expand the rib ROI outward
- build local orthogonal cross-sections along the centerline
- sample radial attenuation profiles over `360 degrees`
- save `position along rib x angle` heatmaps for `outer_peak` and `rise`

### Usage

```bash
python -m heimdallr.rib_centerline_profile_debug \
  --ct /path/to/ct.nii.gz \
  --mask-dir /path/to/segmentations \
  --structures rib_right_6,rib_left_7 \
  --output-dir /path/to/rib_centerline_debug
```

Outputs:

- `*_maps.png`: 3D heatmaps by rib position and angular sector
- `*_profiles.png`: lowest-profile radial curves
- `summary.json`: centerline size, suspicious connected components and top profiles

### `rib_localization.py`

This is a stripped-down rib-only localization runner based on the same 3D centerline method.

It outputs rib segments in `cm posterior -> anterior`, split into:

- `core_suspicious`
- `anterior_low_confidence`

Use it when you want rib localization only, without the broader bone triage report.

### `rib_localization_aggregate.py`

This merges multiple rib-localization shards into one consolidated JSON.

It is intended for parallel runs such as:

- right even ribs
- right odd ribs
- left even ribs
- left odd ribs

## Status note

At the current state of the repository:

- rib localization works as an investigation aid on selected ribs
- it remains too slow and too non-specific for automatic whole-rib detection
- it should be treated as an exploratory workflow, not as a production detector
