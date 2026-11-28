# Verdict: ISLES'24: raw 4D CTP vs perfusion maps, with the CTP registered to NCCT

<!-- VERDICT:AUTO:BEGIN (written by tools/verdict.py; do not edit by hand) -->

## Verdict: PARK

- Run: `isles24-ctp-reg-exp-001-pilot-20261127t071728z-f2e4d4` (experiment `exp_001_pilot`)
- Code: commit `649669230d98bbf5a391a0a37babc4a3480c729d`, config `sha256:2e60f14c2702`
- Pre-registration lock: `sha256:e3de04a77e74b155c17035df899ff6ed38a3af30919a089059ada322b59b830e`, L1 round: 1
- Subjects usable / requested: 40 / 40
- GPU hours: 1.90, computed 2026-11-28T07:22:41Z

| Gate | Expression | Result |
|---|---|---|
| GO | `mde149_ucl(raw-maps) <= 0.02 and ci_hi(raw-maps) >= 0.02 and mean(raw-maps) > 0 and ci_lo(raw-N1) > 0` | false |
| PARK | `mde149_ucl(raw-maps) > 0.02 or ci_hi(raw-maps) < 0.02` | true |
| REVISE | everything else | false |

| Scorecard field | Value |
|---|---|
| `ci_hi(raw-maps)` | 0.0144566 |
| `ci_lo(raw-N1)` | -0.0278539 |
| `mde149_ucl(raw-maps)` | 0.0300681 |
| `mean(raw-maps)` | -0.0217417 |


<!-- VERDICT:AUTO:END -->

<!-- VERDICT:INTERPRETATION (human or Claude; tools never overwrite below this line) -->

## Interpretation

Numbers are from `scorecard.json`, `summary.json`, `paired_stats.csv` and `strata.csv` of the run.
Items marked "counted" are simple counts over its `per_patient.csv`.

**What PARK means here**

- INTERPRETATION: PARK means that, under the locked gate, this study is not scaled up. It does not
  show that maps are better than raw CTP: the interval of raw - maps includes 0.
- FACT: both halves of the PARK expression hold: `mde149_ucl(raw-maps)` is 0.0300681, above 0.02, and
  `ci_hi(raw-maps)` is 0.0144566, below 0.02.

**Dice is very low in every arm**

- FACT: mean per-patient Dice over 40 patients: maps 0.0831, raw 0.0613, N1 0.0593, N2 0.037.
- FACT (counted): median Dice is 0.004 (maps), 0.0027 (raw), 0.0038 (N1) and 0.0 (N2). Dice is exactly
  0 in 17, 19, 17 and 25 patients; in 10 patients it is 0 in all four arms.
- FACT (counted): the predicted volume is below the reference volume in 29, 30, 34 and 31 of 40
  patients (maps, raw, N1, N2). The median reference volume is 12.77 mL; the median predicted volume is
  4.02, 0.84, 2.37 and 0.7 mL. The prediction is empty in 9, 10, 6 and 9 patients.
- FACT: mean Dice is higher in larger lesions for maps and raw: below 5 mL (n = 14) 0.016 and 0.021,
  5-30 mL (n = 17) 0.087 and 0.068, above 30 mL (n = 9) 0.179 and 0.11.
- INTERPRETATION: no arm segments the follow-up infarct usefully. The pilot therefore says little about
  which input would be better for a model that does.
- HYPOTHESIS: contributors are the small training set (32 patients per fold), the 2D network at 128 x 128
  with a fixed 40 epochs and no tuning, and the target itself (a follow-up lesion predicted from
  admission imaging).
- UNKNOWN: which of these, if any, explains the low Dice, and whether a defect in the pipeline
  contributes. The run's checks passed, but the results hold no training curve, no training-set Dice and
  no image overlay.

**Contrasts** (mean per-patient Dice difference, 95% patient bootstrap)

- FACT: raw - maps -0.0217, [-0.0574, 0.0145]. FACT (counted): raw is higher in 12 patients, maps in 14,
  and 14 are tied, all of them with both Dice values 0.
- FACT: H1's pre-registered falsification rule, `ci_hi(raw-maps) < 0.02`, is met.
- INTERPRETATION: for these patients and this recipe, the patient bootstrap excludes a mean raw advantage
  of 0.02 or more. At this level of Dice, that is not evidence about a well-performing pipeline.
- FACT: raw - N1 0.0021, [-0.0279, 0.033]. H2 is not falsified by its rule (`ci_hi(raw-N1) <= 0`), and
  the interval includes 0.
- INTERPRETATION: with no raw advantage over maps, this run cannot say whether temporal information
  would explain one.
- FACT: raw - N2 0.0243, [-0.0082, 0.0609] (H3 not falsified by its rule); maps - N1 0.0238,
  [-0.0086, 0.0557]. Both intervals include 0.

**Planning figures**

- FACT: the sd of the per-patient raw - maps difference is 0.1177; the minimum detectable difference at
  149 patients is 0.027 with that sd and 0.0301 with its 80% upper bound.
- INTERPRETATION (normal approximation): a 149-patient study with this spread would not have 80% power
  for a difference of 0.02.
- HYPOTHESIS: the spread would differ for a model with usable Dice, so these figures may not carry over to
  another design.

## Threats to validity

- **Sample size and interval width.** FACT: 40 patients; the raw - maps interval is 0.0719 wide
  (0.0145 minus -0.0574), more than three times the smallest effect of interest.
- **Selection.** FACT: the 40 patients follow a deterministic rule based on where files sit in the
  archive, not random sampling. UNKNOWN: whether that position relates to image content or outcome.
- **Exclusions and failures.** FACT: 40 of 40 patients usable, 0 empty lesion masks, 0 grid mismatches,
  and `checks.json` lists no problem.
- **Training randomness.** FACT: one training seed per fold. The bootstrap resamples patients only, so
  the intervals do not include variation between training runs.
- **Deviations from the pre-registration.** FACT: the parameters of `configs/exp_001_pilot.yaml` appear
  unchanged in the run's `summary.json`. UNKNOWN: any deviation the outputs do not record.
- **Leakage.** FACT: five folds split by patient, the same for every arm, with fixed epochs and threshold.
  UNKNOWN: whether the recipe carried over from the earlier notebook was tuned on any of these patients.
- **Differences between arms besides the input's content.** FACT: the first convolution has a different
  number of input channels (maps 6, raw 18, N1 4, N2 18). The non-finite voxels recorded in `cases.csv`
  are all in the MTT map, which only `maps` receives (up to 225045 voxels in one patient).
- **Acquisition and preprocessing.** FACT: in these 40 patients the number of CTP frames ranges from 44
  to 59 and the slice spacing from 2.0 to 4.0 mm. UNKNOWN: the CTP frame interval, the software that
  computed the maps, and how much of each volume the CTP covers (checked before the run for one subject
  only). HYPOTHESIS: resampling the CTP onto the NCCT grid smooths it in a way that matters for `raw`.
- **Other runs of this round.** FACT: the verdict block lists no other run. The run of the earlier
  direction `isles24-ctp` stopped before training and produced no Dice.
- **Multiplicity.** FACT: four contrasts, no correction; the gates use raw - maps and the lower bound of
  raw - N1.
- **What the data cannot say.** INTERPRETATION: nothing about all 149 patients, about 3D or nnU-Net-style
  models, about CTP in its native space, or about patients without a follow-up lesion.
