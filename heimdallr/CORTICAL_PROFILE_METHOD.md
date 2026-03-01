# Cortical Profile Method

## Motivation

For gross bone metastases with cortical destruction, a purely morphological shell estimate can be too crude.

The alternative implemented here is a `center-to-cortex attenuation profile` idea:

- stay inside the segmented bone mask
- start from the surface voxel
- walk inward along the local thickness gradient until reaching a deeper local center
- read the CT attenuation profile along that inward path
- expect a cortical peak near the surface
- mark a cortical failure when that peak is missing and the rise from inner bone to outer cortex is insufficient

This is closer to the clinical intuition:

`inner trabecular bone -> progressive increase -> cortical high attenuation peak`

If the cortical peak disappears over a continuous physical extent, that region becomes suspicious for cortical destruction.

## Current implementation

For `ribs` and `vertebrae`, the cortical defect map is derived from attenuation profiles instead of from erosion-based shell thresholding.

The algorithm is:

1. Compute the one-voxel bone surface.
2. Compute the in-mask Euclidean distance transform in millimeters.
3. For each surface voxel, walk inward by repeatedly moving to the neighboring voxel with greater in-mask distance.
4. Extract a short attenuation profile along that path.
5. Compare:
   - `outer_peak`: maximum HU in the first few samples near the surface
   - `deep_median`: median HU deeper in the path
6. Mark a failure when:
   - the outer peak is lower than the expected preserved cortical range, and
   - the rise from deep bone to outer cortex is below the minimum expected rise
7. Aggregate connected failures and measure their physical extent.

The scoring layer currently gives extra weight to cortical interruptions with extent `>= 6 mm`.

## Rib version

Ribs are thin, so the profile is short and the local center is shallow.

The rib-specific settings are more conservative in depth and more demanding for cortical rise:

- minimum inward depth around `1.8 mm`
- minimum cortical rise around `140 HU`
- stricter surface drop requirement

This is intended to prefer true loss of cortical rim over ordinary thin-rib partial volume.

## Vertebra version

Vertebrae are thicker, so the inward path is longer and the cortical rise can be evaluated over a more stable profile.

The vertebra-specific settings use:

- minimum inward depth around `3.0 mm`
- minimum cortical rise around `110 HU`

Important limitation:

the current mask is the whole vertebra, not an explicit posterior-element mask. So posterior-element detection is still approximate and depends on the vertebral segmentation quality.

## Relation to literature

I did not find a paper that exactly matches this implementation for metastatic bone detection on CT.

Closest adjacent literature found:

- CT CAD for sclerotic spinal lesions uses attenuation, connected components, anatomy-aware false-positive reduction, but not this exact center-to-cortex profile rule:
  [Automated detection and segmentation of sclerotic spinal lesions on body CTs](https://pubmed.ncbi.nlm.nih.gov/34291325/)
- Clinical CT methods for cortical bone analysis estimate cortical properties from blurred clinical CT data using model-based intensity ideas at the cortex boundary, which is conceptually adjacent:
  [Model-based measurement of cortical bone thickness and density in CT](https://pubmed.ncbi.nlm.nih.gov/21497500/)
- Radiology reviews of bone metastases describe cortical destruction on CT as a key structural sign, which motivates this heuristic:
  [Imaging of bone metastasis: An update](https://pmc.ncbi.nlm.nih.gov/articles/PMC4553252/)

So this method should be treated as:

- inspired by imaging physics and radiologic appearance
- adjacent to quantitative cortical analysis literature
- not a copied or validated published metastasis detector

## Practical limitations

- Depends strongly on mask quality from TotalSegmentator.
- Sensitive to partial-volume effects in thin ribs.
- Whole-vertebra masks blur the distinction between vertebral body and posterior elements.
- Degenerative change, fractures, and motion can mimic cortical failure.
- The 3D rib centerline variant is still too slow and too non-specific for whole-rib automatic detection.

## Current status

This method should currently be treated as:

- a research/debug heuristic
- useful for directed review of known suspicious ribs
- useful for expressing rib findings as posterior-to-anterior segments in cm
- not suitable yet as a production rib detector or reliable rib ranker

## Why keep it

Even with these limitations, this method is attractive because it encodes a clinically meaningful rule:

`expected cortical peak absent over a continuous physical span`

That is a stronger target than simply saying:

`the shell is hypodense`

## Intuition sketches

### Cross-section idea

The target is not a purely geometric shell. The target is the expected attenuation behavior from the inside of the bone toward the cortex.

```text
normal cross-section

      outside
        .
    #############        <- cortical rim, high HU
   ###         ###
  ##   .........  ##     <- trabecular / medullary region, lower HU
  ##    ...C...   ##     C = local inner center / medial region
  ##   .........  ##
   ###         ###
    #############
        .
      outside
```

Expected interpretation:

- inner bone: lower attenuation
- toward the boundary: attenuation rises
- near the boundary: a cortical peak appears

If the cortical peak is absent over a continuous span, that span becomes suspicious.

### Normal attenuation profile

```text
HU
^
|                          /\
|                        /   \        <- cortical peak
|                     __/     \__
|                  __/
|              ___/
|_____________/____________________> distance
     deep center      surface   outside
```

### Suspicious attenuation profile

```text
HU
^
|
|                     __
|                  __/  \__       <- no convincing cortical peak
|               __/
|           ___/
|__________/______________________> distance
     deep center      surface   outside
```

### Multi-direction intuition

The same local center can be probed in several directions.

```text
             boundary
         \      |      /
           \    |    /
boundary ----  C  ---- boundary
           /    |X   \
         /      |      \
             boundary
```

`X` marks a direction where the expected cortical peak fails.

The current heuristic approximates this by tracing from candidate surface voxels inward toward a local center and then slightly outward beyond the current mask boundary.
