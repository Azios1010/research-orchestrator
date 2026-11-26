# Pre-registration: ISLES'24 pilot, raw 4D CTP vs perfusion maps

Direction `isles24-ctp`, drafted 2026-11-26 from `reference/isles24_raw_vs_maps_pilot.ipynb`.
**Status: DRAFT, pending owner approval. Not locked.**

Claims are labelled FACT / INTERPRETATION / HYPOTHESIS / UNKNOWN.

## Purpose

A cheap pilot to get rough numbers, not a result for publication:

1. check that the pipeline runs end to end and that CTP, maps and lesion mask share one voxel grid;
2. estimate the per-patient spread of the Dice difference, to plan a full-size experiment;
3. get a first look at whether raw beats maps, and whether the controls N1 and N2 remove that gain.

## Hypotheses

- HYPOTHESIS H1: raw has a higher mean per-patient Dice than maps. Falsified if `ci_hi(raw-maps) <= 0`.
- HYPOTHESIS H2: a raw advantage needs temporal information, so raw beats N1. Falsified if
  `ci_hi(raw-N1) <= 0`.
- HYPOTHESIS H3 (secondary, not gated): it needs temporal order, so raw beats N2. Falsified if
  `ci_hi(raw-N2) <= 0`.

## Data

- FACT (from the notebook): the source is the Hugging Face dataset `hugging-science/isles24-stroke`,
  149 parquet shards, one patient each, licence CC BY-NC-SA 4.0.
- UNKNOWN: that the mirror matches the official ISLES'24 release, and that all modalities share the
  lesion-mask grid.
- TODO (owner): add the ISLES'24 dataset and challenge citations before using any result. None are
  written here because none have been checked.

## Design (fixed by `configs/exp_001_pilot.yaml`)

- 40 patients drawn without replacement with seed 2027.
- Arms, same 2D U-Net and recipe, only the first convolution differs:
  - `maps`: NCCT, CTA, CBF, CBV, MTT, Tmax
  - `raw`: NCCT, CTA, 16 CTP frames (window starting 5 frames before the enhancement peak,
    baseline of 3 frames subtracted)
  - `N1`: NCCT, CTA, temporal mean and temporal maximum of those frames
  - `N2`: NCCT, CTA, the same 16 frames in a random order fixed per patient
- In-plane size 128 x 128, 5 patient-level folds identical for all arms, 40 epochs, batch 16, AdamW
  with one-cycle learning rate 1e-3, base width 32, half of each batch from lesion slices, flips for
  augmentation and test-time augmentation, threshold 0.5 inside the NCCT brain mask. The same
  training seed per fold for every arm.
- Non-finite voxels in the four maps and in every CTP frame are replaced before any other step: NaN
  becomes 0; infinities are clipped to the largest or smallest finite value of the same map or frame.
  The count per case is recorded in `cases.csv`. NCCT, CTA and the lesion mask are not altered.
- In-plane downsampling, the same for every arm and mask: block averaging when height and width are
  whole multiples of 128, otherwise a Gaussian prefilter (sigma = (factor - 1) / 2) and linear zoom.
  A downsampled mask voxel is foreground if the resampled foreground fraction is at least 0.5.

## Exclusions (decided before seeing results)

- A patient is excluded if any modality is missing, if its grid differs from the lesion mask, if no
  slice has at least 2% brain, or if downloading or preprocessing raises an error.
- A run is invalid (no verdict) if fewer than 80% of requested patients are usable, if more than 20%
  show a grid mismatch, if any arm x fold is missing, or if an arm has Dice 0 for every patient.

## Analysis

- Per-patient Dice. Contrasts, in this order and from one bootstrap stream (seed 2030):
  raw - maps, raw - N1, raw - N2, maps - N1.
- Percentile bootstrap over patients, 10000 resamples, 95% interval.
- `mde149 = 2.8 * sd(delta) / sqrt(149)`: the smallest mean difference a 149-patient study could
  detect (normal approximation, two-sided 0.05, 80% power).
- No multiplicity correction. N2, maps - N1, lesion-size strata and volume errors are descriptive.

## Gates (copied from `direction.yaml`, which is authoritative) — DRAFT

`mean(raw-maps)` is the mean per-patient Dice difference, raw minus maps.

- GO: `mean(raw-maps) > 0.02 and ci_lo(raw-N1) > 0 and mde149(raw-maps) <= 0.03`
- PARK: `ci_hi(raw-maps) < 0 or (mean(raw-maps) <= 0 and mde149(raw-maps) > 0.03)`
- REVISE: everything else

### Open questions for the owner before approval

1. GO does not require the raw - maps interval to exclude 0. FACT (arithmetic): with 40 patients,
   sd 0.13 and mean 0.021, the normal-approximation 95% interval is about -0.019 to 0.061, and
   `mde149` is 0.030, so the first and third GO conditions hold. Confirm that this is intended.
2. A precise result with no raw advantage (`mean(raw-maps) <= 0`, `mde149 <= 0.03`, interval
   reaching 0 or above) gives REVISE, while a noisy one (`mde149 > 0.03`) gives PARK. Confirm the
   direction of the second PARK condition.
3. INTERPRETATION: the GO threshold 0.02 is below the allowed `mde149` of 0.03. If the true
   difference were 0.02 and `mde149` were 0.03, a 149-patient study would have about 46% power
   (normal approximation, two-sided 0.05).
4. Each round allows REVISE once more (`max_l1_rounds: 2`). Decide in advance what may change in
   round 2.

## Known limitations

- FACT: not nnU-Net, 2D, in-plane downsampling, fixed epochs, no hyperparameter search, about 40
  patients.
- FACT: preprocessing differs from the notebook in two places (non-finite voxels, anti-aliased
  downsampling), so numbers are not expected to equal the notebook's.
- HYPOTHESIS: requiring half of a block to be foreground removes small or thin lesions from the
  downsampled mask; a case whose mask becomes empty scores Dice 1 if the prediction is empty too.
  Frequency not checked.
- INTERPRETATION: absolute Dice values are not comparable with ISLES'24 leaderboard numbers.
- HYPOTHESIS: replacing a NaN in a CTP frame by 0 before baseline subtraction creates an artificial
  dip in that voxel's time curve. How often non-finite voxels occur is UNKNOWN.
