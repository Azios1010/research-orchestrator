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

- HYPOTHESIS H1: raw has a higher mean per-patient Dice than maps. Falsified if the 95% interval of
  raw - maps lies below the smallest effect of interest, `ci_hi(raw-maps) < 0.02`.
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
- A usable patient whose lesion mask is empty after downsampling is trained and evaluated but left out
  of every paired contrast, the strata and the arm means (`gt_empty` in `per_patient.csv`). Reason
  (FACT, arithmetic): with an empty mask Dice is 1 for an empty prediction and 0 for any other, which
  says nothing about segmentation quality. They are reported as `n_excluded_empty_gt`; `n_primary` is the
  number left. The run stops before training if more than 25% of the usable patients are empty.
- A run is invalid (no verdict) if fewer than 80% of requested patients are usable, if more than 20%
  show a grid mismatch, if more than 25% of the usable patients have an empty mask, if fewer than
  2 patients remain in the paired statistics, if any arm x fold is missing, or if an arm has
  Dice 0 for every patient.

## Analysis

- Per-patient Dice. Contrasts, in this order and from one bootstrap stream (seed 2030):
  raw - maps, raw - N1, raw - N2, maps - N1.
- Percentile bootstrap over patients, 10000 resamples, 95% interval.
- `mde149 = 2.8 * sd(delta) / sqrt(149)`: the smallest mean difference a 149-patient study could
  detect (normal approximation, two-sided 0.05, 80% power).
- `mde149_ucl`: the same with `sd(delta)` replaced by its one-sided 80% upper confidence bound,
  `sd * sqrt((n - 1) / chi2.ppf(0.20, n - 1))` with n the number of patients in the contrast (assumes
  roughly normal differences). A pilot sd is itself noisy; this guards against an optimistic one.
- Smallest effect of interest (SESOI) for the mean Dice difference: 0.02. Proposed by Claude, not yet
  chosen by the owner (see the open questions).
- No multiplicity correction. N2, maps - N1, lesion-size strata and volume errors are descriptive.

## Gates (copied from `direction.yaml`, which is authoritative) — DRAFT

`mean(raw-maps)` is the mean per-patient Dice difference, raw minus maps, over the patients with a
non-empty lesion mask. SESOI = 0.02.

- GO: `mde149_ucl(raw-maps) <= 0.02 and ci_hi(raw-maps) >= 0.02 and mean(raw-maps) > 0 and ci_lo(raw-N1) > 0`
- PARK: `mde149_ucl(raw-maps) > 0.02 or ci_hi(raw-maps) < 0.02`
- REVISE: everything else

INTERPRETATION of what the verdicts mean:

- GO means "the full 149-patient study is worth running and would be at least 80% powered for 0.02".
  It does not mean H1 is supported: the raw - maps interval may well include 0.
- PARK means either the pilot's whole interval lies below 0.02 or a 149-patient study could not detect
  0.02 even with a pessimistic sd.
- REVISE: powered and not ruled out, but the mean is not positive or raw does not beat N1 (the
  temporal information is not what helps).
- GO and PARK cannot both be true: GO needs `ci_hi >= 0.02` and `mde149_ucl <= 0.02`, PARK needs one of
  them to fail. A test checks this on a grid.

### Open questions for the owner before approval

1. **SESOI.** FACT (arithmetic, normal approximation, 40 patients in the pilot): `mde149_ucl <= 0.02`
   needs a pilot sd(delta) of at most 0.078. For 0.03 the limit is 0.117, for 0.04 it is 0.157, for 0.05
   it is 0.196. HYPOTHESIS: the per-patient Dice difference between two 2D U-Nets has an sd around 0.13
   (an earlier planning figure, not measured here). If so, SESOI 0.02 will return PARK whatever the
   means are, and 0.03 would not be reached either (`mde149_ucl` about 0.033). Decide the smallest
   difference you would care about *before* the run; changing it afterwards is moving the goalposts.
2. GO does not require the raw - maps interval to exclude 0 (see above). Confirm that this is intended.
3. A pilot of 40 patients gives a wide interval for sd(delta); the 80% upper bound is about 11% above the
   sample sd at n = 40 (FACT, arithmetic).
4. Each round allows REVISE once more (`max_l1_rounds: 2`). Decide in advance what may change in
   round 2. Since the lock now covers everything under `src/`, any code change needs a new round.

## Known limitations

- FACT: not nnU-Net, 2D, in-plane downsampling, fixed epochs, no hyperparameter search, about 40
  patients.
- FACT: preprocessing differs from the notebook in two places (non-finite voxels, anti-aliased
  downsampling), so numbers are not expected to equal the notebook's.
- HYPOTHESIS: requiring half of a block to be foreground removes small or thin lesions from the
  downsampled mask. Such cases are now excluded from the contrasts rather than scored 1 or 0 (see
  Exclusions); how often it happens is UNKNOWN until the first run, and a share above 25% stops the run.
- INTERPRETATION: excluding empty-mask patients selects patients with a visible follow-up infarct, so the
  result does not apply to patients without one.
- INTERPRETATION: absolute Dice values are not comparable with ISLES'24 leaderboard numbers.
- HYPOTHESIS: replacing a NaN in a CTP frame by 0 before baseline subtraction creates an artificial
  dip in that voxel's time curve. How often non-finite voxels occur is UNKNOWN.
