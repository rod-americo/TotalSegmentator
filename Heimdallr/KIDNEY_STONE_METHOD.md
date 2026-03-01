# Kidney Stone Detection Method

This workflow uses the CT volume in HU plus the `kidney_left` and `kidney_right` masks from TotalSegmentator to localize dense intrarenal components compatible with stone candidates.

## Threshold

- Default threshold: `130 HU`
- Rationale: this is the threshold most commonly used in the non-contrast CT stone literature and is the working default for this repository.

## Pipeline

1. Load the CT NIfTI and the renal masks.
2. Verify shape and affine alignment between CT and each mask.
3. Restrict analysis to voxels inside `kidney_left` and `kidney_right`.
4. Mark voxels with attenuation strictly above the selected threshold.
5. Split the thresholded map into 3D connected components.
6. Keep components that satisfy either a minimum connected-voxel count or a minimum volume.
7. Compute per-component metrics and save axial/coronal review overlays.

## Minimum size filter

- Default connected-voxel filter: `>= 3 voxels`
- Optional volume filter: approximately `3-5 mm3`, configurable with `--min-volume-mm3`
- Retention rule: a component is kept if it passes the voxel filter or the volume filter

## Per-component metrics

- `voxel_count`
- `volume_mm3`
- `volume_ml`
- `hu_mean`
- `hu_max`
- `centroid_ijk`
- `centroid_xyz_mm`
- `largest_axis_mm`
- `principal_axes_mm`

The core metrics requested for review are voxel count, volume in `mm3`, mean HU, max HU and centroid. The axis measurements are included because they are useful when approximating real specimen size.

## Overlay review images

For each connected component, the script renders:

- one axial slice through the component centroid
- one coronal slice through the component centroid

Display conventions:

- grayscale CT with abdominal window
- kidney contour in blue
- thresholded component overlay in yellow/orange

## Notes

- This is a heuristic post-processing step, not a trained detector.
- Performance depends on mask quality, CT calibration, acquisition phase and partial-volume effects.
- Dense vascular or collecting-system calcifications near the kidney can still appear inside the mask if the segmentation leaks.
