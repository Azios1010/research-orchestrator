# Bootstrap decisions, assumptions and bug fixes (2026-11-26)

Where the specification was silent or ambiguous, the simplest option was taken and recorded here.

## Open points for the owner

1. **ISLES gates are DRAFT and unlocked.** They were replaced before the first commit (below). Answer the
   open questions in `prereg.md` before approving.
2. **The lock does not cover everything.** The rest of `src/` (including `data.py`, `arms.py`, `train.py`
   and `main.py`, which writes `summary.json`), configs, and `labcore` edits without a version bump can
   change after locking without CI noticing. `CLAUDE.md` forbids it.
3. **Per-subject results will be committed.** `results/` will hold dataset subject IDs and per-subject
   metrics, volumes and image geometry. Phenotype columns are refused; confirm the rest is allowed before
   the first ISLES run.

## Environment

- The target folder sits inside another git repository (the drive root), so `git init -b main` was run in
  it. Nothing was committed.
- Built and tested on Windows with Python 3.11. The workflows run on Linux, which was not tested.
- `kaggle` 2.2.5 imports `six` without declaring it: install `kaggle==2.2.5 six`.
- `.gitattributes` forces LF so hashes agree across machines.

## Lifecycle, gates and verdicts

- Gates use a small parser (`and`, `or`, `not`, comparisons, numbers, field names; no `eval`, no chained
  comparisons). `ci_lo(raw-maps)` is a literal field name, not a function call.
- Every field named by a gate must exist and be finite. If GO and PARK are both true, the verdict is REVISE
  with a note.
- `gates.approval` (`DRAFT` or `APPROVED`) was added. Locking refuses DRAFT; a new round resets to DRAFT.
- Kaggle dispatch requires `L1_RUNNING`, APPROVED gates and a matching hash. Dry runs do not.
- YAML blocks are hashed as canonical JSON and files with LF line endings, so comments and formatting of
  `direction.yaml` do not matter. (What the hash covers: see the changes below.)
- `registry.yaml` is authoritative; `direction.yaml.status` mirrors it.
- An L0 pass is bound to the hashes of the direction's code and of `labcore`; locking refuses if either
  changed since. Code changes after the lock do not unlock.
- A round is counted when locked. `max_l1_rounds` defaults to 2.
- The watcher writes only `state/` and `results/`, so its `verdict.json` is provisional. The verdict is
  recorded by running `tools/verdict.py` in a pull request.
- `verdict.py` uses the latest complete Kaggle run of the current lock and lists the other usable runs.
  Local runs are ignored unless `--allow-local` (testing only).
- No tool archives a direction; it is a manual status edit.

## Dispatch, budget and watcher

- One active run per kernel. An identical completed or active run is skipped, so repeated triggers do not
  spend twice (`--force` repeats a completed one).
- Enqueue applies only to the weekly cap. A direction-cap refusal is never queued. The queue is drained by
  hand (workflow input `drain_queue`).
- The direction cap counts all of its runs in all rounds. A kernel that timed out while still queued counts
  0 GPU hours.
- The watcher evaluates and retries with the commit the run was submitted from when git can provide it. A
  retry is submitted only if the bundle is byte-identical (same hash) and the budget allows.
- Failure classification is conservative: unknown means `code` (no retry). The final exception outweighs
  earlier log noise. 401/403/gated errors are `code`. Cancelled runs are never retried.
- A TIMEOUT that never started is `infra`. The old kernel version may still be queued on Kaggle.
- Outputs or logs belonging to an earlier kernel version (different `run_id`) are not used.
- Only declared outputs plus `status.json` and `env.json` are stored; data suffixes and files over 5 MB are
  refused. `sanity.png` (patient images) stays on Kaggle.
- Issues: one `[tracking] <slug>` issue per direction for successes; one `run-failed` issue per failed run,
  with its retry commenting on the same issue.
- DeepSeek receives a redacted log tail and five fields of `status.json`. Model `deepseek-flash`, thinking
  disabled for the official endpoint (both unverified).

## Kernel bundle

- Ships `labcore`, the direction and only the selected config. Leaves out `notebooks/`, `tests/`,
  `results/`, data-like and secret-named files. Deterministic.

## Workflows

- No `labcore/data` package was created: "labcore/data-style adapters" is read as adapters like
  `src/data.py` inside each direction.
- CI runs L0 for changed directions, and for every direction when `labcore/` changed. The root test job
  has no torch, so its ISLES loop is skipped there.
- CI diffs against the merge commit's first parent, so bot commits on `main` do not trip the
  protected-path check.
- `dispatch.yml` runs `main`'s code. A `run-l1` pull request is only read and bundled
  (`dispatch --source-root`), never executed next to the Kaggle credentials. Its changed configs are
  dispatched; if there are none the job fails. A push skips unlocked directions; the first push of the
  branch dispatches nothing.
- `dispatch.yml` and `watcher.yml` share concurrency group `lab-state` with `queue: max`. GitHub documents
  the key; actionlint 1.7.12 does not know it, so that one message is ignored.
- `actions/checkout` and `actions/setup-python` are pinned by commit. `anthropics/claude-code-action@v1`
  is a moving tag, as specified; pin a commit if you prefer.
- `claude.yml` has no `prompt` input, because a prompt makes the action ignore the `@claude` comment.
- `HF_TOKEN` as a GitHub secret is unused: secrets are never written into a kernel.
- The owner-only checks (`github.actor == github.repository_owner`) assume a personal repository.
- A bot push uses `pull --rebase --autostash` and retries; if it still fails the job fails loudly, since a
  kernel may then be running without a record.

## Graduation

- `git filter-repo` if installed, else `git subtree split` through a temporary branch that is deleted.
- Requires a committed direction, `labcore` and registry. Nothing is committed in the new folder.
- Adds the vendored `labcore`, `README.md`, `VENDORED.md`, `pyproject.toml` and `evidence/` (scorecards and
  the registry entry). No import rewriting was needed; imports of `tools` or `directions` are refused.
- The registry stores only the folder name in `graduated_to`, not a local path.

## ISLES'24 port

Unchanged from the notebook: patient choice, intensity scaling and the CTP window, the four arms, the N2
permutation, folds, 2D U-Net, loss, optimiser and schedule, per-fold seeds, flip test-time augmentation, threshold and brain mask, contrasts,
bootstrap stream and `MDE = 2.8*sd/sqrt(149)`. A test compares the statistics with the notebook's
computation. Preprocessing differs in the two places described under the changes below.

### Bug fixes and behaviour changes

1. A downloaded shard is deleted even if preprocessing fails (it stayed on disk before).
2. Resume reuses `per_patient.csv` only if settings and patient list are identical; before, any leftover
   file was trusted.
3. Gates use unrounded statistics (`summary.json` → `contrasts`); the rounded table is unchanged.
4. A present but empty `subject_id` falls back to the file name.
5. Temporary NIfTI files are released before deletion and a failed delete no longer fails the case.
6. A failed download is reported as `download failed: ...`, separate from preprocessing errors, so the
   watcher can tell infrastructure from code.
7. `assert` on the number of usable cases became `RuntimeError` with the reason.
8. The run stops before training if more than 20% of cases have a grid mismatch (also in `checks.py`).
9. New outputs: `strata.csv`, and in `summary.json` requested and mismatch counts, device, per-arm values
   and full-precision contrasts.
10. Notebook configuration and `pip` cells became the YAML config and `requirements.txt`; unknown
    parameters are errors.

### Noted, deliberately not changed

- GPU training is not fully deterministic; how much results vary between identical runs is UNKNOWN.
- `exp_001_pilot` estimates 1.5 GPU hours from the notebook's own estimate; UNKNOWN until measured.

## Changes before the first commit (2026-11-26)

Requested by the owner. Nothing had been locked or run, so no result is affected.

### 1. Non-finite voxels (isles24-ctp)

- In the four maps and in every CTP frame, before any other step: NaN becomes 0; +inf and -inf are clipped
  to the largest and smallest finite value of the same map or frame (0 if there is none). "Clip" was read
  this way because numpy's default replacement (the largest float) overflows the baseline mean.
- `cases.csv` gains `nonfinite_ctp`, `nonfinite_cbf`, `nonfinite_cbv`, `nonfinite_mtt`, `nonfinite_tmax` and
  `nonfinite_voxels`, counted over the whole volume. They are empty for cases excluded before
  preprocessing finished.
- NCCT, CTA and the lesion mask are not altered (not requested).
- A NaN CTP voxel becomes 0 before baseline subtraction, so it shows as a dip in that voxel's curve.

### 2. In-plane downsampling (isles24-ctp)

- `area_down` takes the block mean when height and width are whole multiples of `img_size`. Otherwise it
  applies a Gaussian prefilter (sigma = (factor - 1) / 2 per axis) and then linear zoom with `grid_mode`, so
  that pixel areas are aligned as in the block mean.
- Brain and lesion masks: foreground where the resampled fraction is at least 0.5, on both paths.
- NCCT, CTA, maps, CTP frames and masks all go through this one function, so every arm is treated alike.
- Small or thin lesions can vanish from the downsampled mask. `gt_ml`, the lesion-size strata and the
  Dice-1-for-two-empty-masks rule all use that mask. The frequency is UNKNOWN; `gt_ml_full` is unaffected.

### 3. Pre-registration lock

- One sha256 over per-part digests of `prereg.md`, the `gates` and `scorecard_map` blocks, `checks.py`,
  `src/evaluate.py`, and the version string. No lock made with the earlier formula existed.
- "`labcore.__version__` of stats" was read as that string, standing for the statistics code; there is no
  second version number. A root test fails if `labcore/stats` changes without a bump.
- The version is read from the checkout being hashed by parsing the file, so pull-request code is still not
  executed during dispatch.
- `L1_RUNNING` directions compare the current version. Finished directions compare the version recorded in
  their lock; otherwise the first later release would break CI for every finished direction for good. One
  pull request cannot both record a verdict and bump the version.
- While any direction is `L1_RUNNING`, a pull request that bumps the version fails CI.
- Every direction now needs `src/evaluate.py`. The template moved its scoring and statistics there.

### 4. L0 on library changes

- `tools/ci/changed.py` (used by `ci-l0.yml`) selects every direction folder with tests, archived ones
  included, when any file under `labcore/` was added, changed or deleted. `_template` stays covered by the
  root tests.

### 5. Phenotype columns

- `labcore/privacy.py` matches names: CSV/TSV header cells, JSON object keys and listed column names,
  against words such as `age`, `sex`, `nihss`, `mrs` and other demographic, clinical and identifying fields.
  A checked file that cannot be read fails.
- Applied in CI to everything under `results/`; by the watcher, which does not store such a file and marks
  the run invalid; to scorecard field names; in L0, so it fails before GPU time is spent; and to metric names
  in `direction.yaml`.
- CI runs on pull requests only. The bot's direct commits to `main` rely on the watcher's refusal.
- Not caught: a phenotype under a neutral name or as a value, anything below the CSV header row, and text in
  `error_log.txt`, `diagnosis.md` or issues (do not print patient fields in a kernel).

### 6. ISLES gates (still DRAFT)

- GO: `mean(raw-maps) > 0.02 and ci_lo(raw-N1) > 0 and mde149(raw-maps) <= 0.03`
- PARK: `ci_hi(raw-maps) < 0 or (mean(raw-maps) <= 0 and mde149(raw-maps) > 0.03)`
- Otherwise REVISE (the verdict script's default).
- So that the gates read as written, the four ISLES scorecard keys `mean_delta(x-y)` were renamed
  `mean(x-y)`; they still read `contrasts.x-y.mean_delta`.
- Consequences are listed as open questions in `prereg.md`. H1's `falsified_if` still says
  `ci_hi(raw-maps) <= 0`, while PARK uses `< 0`.
