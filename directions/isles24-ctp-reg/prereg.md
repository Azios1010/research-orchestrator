# Pre-registration: ISLES'24 pilot, raw 4D CTP vs perfusion maps, NCCT-registered CTP

Direction `isles24-ctp-reg`, drafted 2026-11-27.
**Status: DRAFT, pending owner approval. Not locked.**

Claims are labelled FACT / INTERPRETATION / HYPOTHESIS / UNKNOWN.

## Why this direction exists

It replaces `isles24-ctp`, which asked the same question.

- FACT: that direction's only run, `isles24-ctp-exp-001-pilot-20261127t042526z-a860b3` (commit
  e7d7732), stopped before any training with "grid mismatch in 40/40 cases": in the Hugging Face
  mirror the CTA and CTP are in their native space, not on the lesion-mask grid.
- FACT: it produced no Dice value and no contrast. The hypotheses, arms, settings, analysis and gate
  expressions below are carried over from it, so none of them was chosen after seeing an outcome.
- What changed: the source of CTA, CTP, maps and lesion mask; how the 40 patients are chosen; the
  grid check (now shape and affine) and a new source check; when the run stops while fetching.
- FACT: `src/arms.py`, `src/model.py`, `src/train.py` and `src/evaluate.py` are byte-identical to
  those of `isles24-ctp`. The preprocessing arithmetic from the loaded volumes on is the same.

## Purpose

A cheap pilot to get rough numbers, not a result for publication:

1. check that the pipeline runs end to end on the registered volumes;
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

- CTA, CTP, CBF, CBV, MTT, Tmax and the lesion mask: the `derivatives/*space-ncct*` files of the
  ISLES'24 training release, Zenodo record 17652035, `train.7z` (99,022,114,670 bytes), licence
  CC BY-NC-SA 4.0. FACT (archive index): 149 subjects, each with all seven files.
- NCCT: the Hugging Face mirror `hugging-science/isles24-stroke`, one parquet shard per patient.
- Integrity: every file taken from the archive must match the CRC32 listed in
  `src/zenodo_manifest.json`, which the lock covers. The checksum of the whole archive is not
  verified, because the whole archive is never downloaded.
- FACT (checked for sub-stroke0001 only): the registered CTP is 512 x 595 x 75 x 55 float32 at
  0.469 x 0.469 x 2.0 mm; its header says "Warped image using NiftyReg (reg_resample)"; CTA, the four
  maps, the lesion mask and the mirror's NCCT have the same shape and affine; the lesion mask is equal
  in both sources.
- FACT (same subject): the CTP covers only part of the NCCT volume. Of 65 slices kept for training,
  50 had any CTP signal and 47 any CBF signal; no lesion voxel (after downsampling) lay on a slice
  without CTP signal. Outside its field of view the CTP is -1000 HU.
- UNKNOWN: the same for the other 148 subjects (the run checks shape, affine and mask equality for
  every patient); the CTP frame interval (the header records none); the software that computed the
  maps.
- TODO (owner): add the ISLES'24 dataset and challenge citations before using any result. None are
  written here because none have been checked.

## Design (fixed by `configs/exp_001_pilot.yaml`)

- 40 patients, chosen by a fixed rule that looks only at where files sit in the archive
  (`select_subjects` in `src/zenodo.py`). The archive is made of separately compressed blocks, and
  reading a file means reading its block from the start. Subjects whose seven files sit in one block
  come first (128 of 149): the one ending earliest in every block, then the second of every block.
  This streams about 33 GB.
  - FACT: the archive stores subjects in name order, so the rule takes the one or two lowest-numbered
    whole subjects of each of 22 blocks. The numbers (`sub-strokeNNNN`) are 1, 2, 9, 10, 15, 16, 22,
    25, 36, 37, 45, 47, 55, 57, 73, 74, 80, 81, 86, 87, 93, 94, 99, 100, 106, 107, 112, 113, 118, 119,
    138, 139, 146, 147, 154, 155, 164, 172, 181, 188. A test pins this list.
  - INTERPRETATION: this is not a random sample. It spans the whole range of subject numbers, but
    whether position in a block is related to image content or outcome is UNKNOWN.
- Arms, same 2D U-Net and recipe, only the first convolution differs:
  - `maps`: NCCT, CTA, CBF, CBV, MTT, Tmax
  - `raw`: NCCT, CTA, 16 CTP frames (window starting 5 frames before the enhancement peak,
    baseline of the first 3 frames subtracted)
  - `N1`: NCCT, CTA, temporal mean and temporal maximum of those frames
  - `N2`: NCCT, CTA, the same 16 frames in a random order fixed per patient
- In-plane size 128 x 128, 5 patient-level folds identical for all arms, 40 epochs, batch 16, AdamW
  with one-cycle learning rate 1e-3, base width 32, half of each batch from lesion slices, flips for
  augmentation and test-time augmentation, threshold 0.5 inside the NCCT brain mask. The same
  training seed per fold for every arm.
- Slices are kept by their NCCT brain fraction (at least 2%), whether or not the CTP covers them. On
  an uncovered slice `maps` and `raw` receive no perfusion signal.
- Non-finite voxels in the four maps and in every CTP frame are replaced before any other step: NaN
  becomes 0; infinities are clipped to the largest or smallest finite value of the same map or frame.
  The count per case is recorded in `cases.csv`. NCCT, CTA and the lesion mask are not altered.
- In-plane downsampling, the same for every arm and mask: block averaging when height and width are
  whole multiples of 128, otherwise a Gaussian prefilter (sigma = (factor - 1) / 2) and linear zoom.
  A downsampled mask voxel is foreground if the resampled foreground fraction is at least 0.5.

## Exclusions (decided before seeing results)

- A patient is excluded, and not replaced, if a file cannot be downloaded or fails its CRC32, if the
  shape or affine of any volume differs from the lesion mask (affine tolerance 0.001), if the mirror's
  shard holds another subject or a different lesion mask, if no slice has at least 2% brain, or if
  preprocessing raises an error. The reason is recorded in `cases.csv`.
- A usable patient whose lesion mask is empty after downsampling is trained and evaluated but left out
  of every paired contrast, the strata and the arm means (`gt_empty` in `per_patient.csv`). Reason
  (FACT, arithmetic): with an empty mask Dice is 1 for an empty prediction and 0 for any other, which
  says nothing about segmentation quality. They are reported as `n_excluded_empty_gt`; `n_primary` is the
  number left. The run stops before training if more than 25% of the usable patients are empty.
- The run stops while fetching, with no result, if more than 20% of the requested patients show a grid
  mismatch or are unusable, if fetching passes 2.5 h, or if after 10 minutes the measured speed implies
  more than 2.5 h. These rules look at no outcome.
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
- Smallest effect of interest (SESOI) for the mean Dice difference: 0.02. Proposed by Claude; confirm
  or change it before approving (open question 1).
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
   round 2. Since the lock covers everything under `src/`, any code change needs a new round.
5. **CTP coverage.** Uncovered slices stay in training and in the per-patient Dice, as in
   `isles24-ctp`. INTERPRETATION: lesion on uncovered slices can only be found from NCCT and CTA, which
   all arms share, so it pulls raw - maps toward 0. Restricting to covered slices is the alternative;
   it would have to be decided and coded before the lock.

## Known limitations

- FACT: not nnU-Net, 2D, in-plane downsampling, fixed epochs, no hyperparameter search, about 40
  patients, not drawn at random (see Design).
- FACT: preprocessing differs from the notebook in two places (non-finite voxels, anti-aliased
  downsampling), and the CTP here is the release's registered volume, so numbers are not expected to
  equal the notebook's.
- HYPOTHESIS: resampling the CTP onto the NCCT grid smooths it in space, which could change what `raw`
  can use compared with the native CTP.
- UNKNOWN: whether 16 frames around the peak contain the late part of the curve in hypoperfused
  tissue; it depends on the frame interval, which is not recorded.
- HYPOTHESIS: requiring half of a block to be foreground removes small or thin lesions from the
  downsampled mask. Such cases are excluded from the contrasts rather than scored 1 or 0 (see
  Exclusions); how often it happens is UNKNOWN until the first run, and a share above 25% stops the run.
- INTERPRETATION: excluding empty-mask patients selects patients with a visible follow-up infarct, so the
  result does not apply to patients without one.
- INTERPRETATION: absolute Dice values are not comparable with ISLES'24 leaderboard numbers.
- HYPOTHESIS: replacing a NaN in a CTP frame by 0 before baseline subtraction creates an artificial
  dip in that voxel's time curve. FACT (sub-stroke0001): no CTP voxel and 2 MTT voxels were non-finite.
