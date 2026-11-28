# Pre-registration: same-patient cells and a linear probe on confusable white cells

Direction `wbc-patient-gap`, drafted 2026-11-28. **Status: DRAFT, pending owner approval. Not locked.**
Once locked (`tools/lock_prereg.py`), this file, the gates, `scorecard_map`, `checks.py`, every file
under `src/` and the labcore version cannot change.

Claims are labelled FACT / INTERPRETATION / HYPOTHESIS / UNKNOWN.

## Question

With frozen features, does a linear probe classify a patient's confusable neutrophil maturation cells
better when other cells of that patient are in its train set than when no cell of that patient is?

This only checks whether there is a patient-specific gap worth solving. It tests no method.

## Hypotheses

- HYPOTHESIS H1: replacing train cells with other cells of the evaluated patients, class by class,
  raises macro-F1 over segmented, band, metamyelocyte, myelocyte and promyelocyte by 0.02 or more.
  Falsified if the 95% interval of shared - disjoint lies below 0.02, `ci_hi(shared-disjoint) < 0.02`.

## Data

All "checked" statements below refer to files downloaded or read on 2026-11-28.

### DeepAPL (the experiment)

- FACT (figshare record 14294675, version 1): peripheral blood smear images of 106 de-identified
  patients with AML or APL seen at Johns Hopkins Hospital, JPG, one folder per patient. The record's
  licence field is CC0, and the owner confirmed it on 2026-11-28. File used:
  `blood smear images_Patient00-105.zip`, 539,130,788 bytes. Its md5 equals the record's; its sha256 is
  pinned in `src/data.py`.
- FACT (same record): `Signed slides` folders are organised by CellaVision cell type; `Unsigned slides`
  are not used here.
- FACT (Sidhom et al., npj Precision Oncology 2021, full text at PMC8121867): cells were acquired and
  pre-classified by a CellaVision DM100, and "the operator subsequently verifies and modifies the
  classification, if necessary"; films were Wright stained; the discussion says one smear per patient,
  with at most 200 cells per smear; all individuals collected before 2018 had APL.
- UNKNOWN: the operator's role, what "signed" means exactly, and how reliable the labels are, in
  particular between neighbouring maturation stages.
- FACT (counted in the archive): one patient has two `Signed slides` folders, and 12 of the 107 such
  folders hold more than 200 images over all classes. UNKNOWN: why this differs from the paper.
- The record's metadata table (diagnosis, age, gender) is a separate file that is never downloaded.

Rules (`src/data.py`): an image is a JPG in a class folder under a `Signed slides` folder. Images with
equal bytes under different classes are all dropped; equal bytes within one class are kept once. Then the
twelve classes below are used.

FACT (checked with that code on the archive): 15,632 signed images; 33 groups of equal bytes under
different classes (66 images); no repeat within a class; 12,364 images in the twelve classes, 106
patients:

| Class folder | Images | Patients |
|---|---|---|
| Blast, no lineage spec | 3295 | 92 |
| Promyelocyte | 685 | 47 |
| Myelocyte | 180 | 40 |
| Metamyelocyte | 73 | 29 |
| Band neutrophils | 169 | 50 |
| Segmented neutrophils | 2081 | 101 |
| Lymphocyte | 3427 | 105 |
| Lymphocyte, variant | 469 | 65 |
| Monocyte | 1308 | 84 |
| Eosinophils | 108 | 40 |
| Basophil | 50 | 30 |
| Erythroblast | 519 | 60 |

The run counts again and stops before any feature is computed if a count differs.

### Acevedo (external description only)

- FACT (Mendeley Data 10.17632/snkd93bnjr.1 and its Data in Brief article, PMC7182702): 17,092 JPG
  images of peripheral blood cells in eight groups, CC BY 4.0. The archive's sha256 is pinned and equals
  the one the repository states; its eight folder counts equal the article's table.
- Seven folders are used (14,744 images); platelets have no counterpart among the twelve classes.

### DinoBloom-S (features)

- FACT: weights from Zenodo record 10908163 (record licence CC BY 4.0), sha256 pinned in
  `src/backbone.py`; model code `facebookresearch/dinov2` at commit
  `e1277af2ba9496fbadf7aec6eba56e8d882d1e35` (Apache-2.0); loaded as in `marrlab/DinoBloom`
  `dinov2/eval/models/return_model.py` (commit 6c5e86d). Checked locally: the strict load succeeds and
  the features are 384-dimensional.
- FACT (arXiv 2404.05022 v1): DinoBloom was pretrained on its 13 datasets except Acevedo, which is kept
  as an external test set. The training data include a bone marrow set, the AML sets of Matek and of
  Hehr, Raabin, and an unnamed set ("LabAnonymous") of over 40,000 peripheral blood images. Neither the
  text nor the references mention Sidhom's APL data.

## Design (`configs/exp_001_pilot.yaml`)

- Features: RGB, bilinear resize to 224 x 224 with Pillow, ImageNet normalisation, the class token of the
  frozen network, scaled to unit length. Nothing is pretrained or fine-tuned.
- Five folds by patient: a seeded order (4101) of the folder names, dealt round the folds.
- Within a fold, each test patient's cells are halved into `eval` and `donor`, class by class; a class
  with an odd count gives the extra cell to each side in turn, from a random first side per patient. This
  is repeated with split seeds 11, 12 and 13.
- Arms:
  - `disjoint`: every cell of the patients outside the test fold;
  - `shared`: the same set, but for each class n of its cells are replaced by n donor cells of that
    class, with n the number of donor cells (or the number of train cells of the class, if smaller).
- Both arms have the same train size and the same count per class, and both are scored on the same `eval`
  cells.
- INTERPRETATION (arithmetic, if the folds hold equal numbers of cells): about one train cell in eight is
  a donor cell in `shared`.
- Probe: multinomial logistic regression (scikit-learn, L-BFGS) over the twelve classes, C = 1.0,
  `class_weight` balanced, at most 5000 iterations, tolerance 1e-4. Fixed before any result; nothing is
  tuned. INTERPRETATION: balanced weights match a primary metric that weighs rare classes equally.

## Primary outcome and analysis

- For one split seed: true positives, false positives and false negatives summed over all evaluated
  patients, F1 per class, macro-F1 over segmented neutrophils, band neutrophils, metamyelocyte, myelocyte
  and promyelocyte (a false positive can come from any of the twelve classes). An arm's value is the mean
  over the three seeds.
- Contrast: shared - disjoint, called `mean(shared-disjoint)` in the gates.
- 95% percentile bootstrap over patients, 10,000 resamples (seed 4102), the same resample for both arms
  and all seeds. The probes are not refitted inside the bootstrap.
- Smallest effect of interest: 0.02. Proposed by Claude, kept by the owner on 2026-11-28.

## Descriptive only (not gated)

- The same contrast per primary class and over all twelve classes; per-class F1 of both arms.
- kNN (k = 20, cosine) for both arms.
- In `shared`: the share of an evaluated cell's 20 neighbours that come from the same patient, against
  the patient's share of the train set, and against that share within each neighbour's class.
- External: the `disjoint` probes on Acevedo, in seven groups (neutrophil = segmented + band;
  eosinophil; basophil; lymphocyte = both lymphocyte classes; monocyte; immature granulocyte =
  promyelocyte + myelocyte + metamyelocyte; erythroblast). The twelve probabilities are summed per group
  and the largest group wins. The same for the held-out DeepAPL `eval` cells of those seven groups
  (blasts are left out). For each seed the counts of the five folds are pooled. INTERPRETATION: the
  difference mixes patient groups, laboratory procedures and class definitions.

## Gates (copied from `direction.yaml`, which is authoritative) — DRAFT

- GO: `ci_lo(shared-disjoint) > 0 and mean(shared-disjoint) >= 0.02`
- PARK: `ci_hi(shared-disjoint) < 0.02`
- REVISE: everything else

INTERPRETATION of what the verdicts mean:

- GO: for these features and this probe, seeing a patient's other cells raises macro-F1 by an estimated
  0.02 or more and the interval excludes 0. The next step compares a method with ContextViT-style context
  conditioning and with colour augmentation.
- PARK: the whole interval lies below 0.02. It does not show that a fine-tuned network has no patient
  gap.
- REVISE: neither.
- FACT: GO and PARK cannot both hold while the estimate is not above the upper end of its interval. A
  percentile bootstrap does not guarantee that; if both held, the lab's verdict code returns REVISE with a
  note.

## Validity requirements (`checks.py`)

A run is invalid if: an evaluated cell, or an image with the same bytes, is in a train set; a `disjoint`
train set holds a cell of a test patient; the arms differ in train size, in a class count of the train or
eval set, or in the eval cells; a fold x seed x arm is missing; a probe did not converge; a class count,
the number of patients or of conflicting duplicate groups differs from the expected one; fewer than 80% of
the 106 patients have a cell to evaluate under every seed; the written tables disagree with each other; a
run on downloaded data did not use DinoBloom-S; or an arm's primary macro-F1 is 0.

## What the results hold

Aggregates only. `patient_counts.csv` has true positives, false positives and false negatives per
patient, class, arm and seed, under a `patient_index` drawn at run time from the operating system's
entropy; the mapping is not saved. INTERPRETATION: because the archive is public, a patient's class counts
could still be matched to a folder, so the index hides the folder name but is not anonymisation.

## Prior art

- FACT (arXiv 2305.19402 v2): ContextViT adds a context token computed from the images of one group
  within a batch (for example a hospital or a plate) and uses it during DINO self-supervised pretraining.
- UNKNOWN: other prior work on patient context for single blood cells; no systematic search was made.

## Known limitations

- FACT: every patient has AML or APL.
- INTERPRETATION: with one smear per patient (as the paper states), "same patient" also means the same
  film, staining and scanning session, and possibly the same operator's labelling habits. The contrast
  cannot separate these from the patient's biology.
- UNKNOWN: whether near-identical images (not byte-equal) exist within a patient. Such a pair split into
  `eval` and `donor` would raise `shared`.
- FACT: metamyelocyte has 73 images over 29 patients. HYPOTHESIS: the interval is wide and the result may
  be REVISE.
- INTERPRETATION: a linear probe on frozen features may not show the whole effect, so PARK does not show
  that a fine-tuned backbone is unaffected.
- FACT: the bootstrap resamples evaluated patients only; it does not include the variation from which
  patients trained the probe.

## Open questions for the owner before approval

1. Probe settings: C = 1.0 and balanced class weights were chosen by Claude without looking at any
   result. Confirm or change them before the first run.
2. To make GO and PARK exclusive in every case, GO could also require `ci_hi(shared-disjoint) >= 0.02`.
   Not applied: the gate expressions are the ones the owner gave.
3. `max_l1_rounds` is 2. Decide in advance what a second round may change after REVISE; any change under
   `src/` needs a new round.
