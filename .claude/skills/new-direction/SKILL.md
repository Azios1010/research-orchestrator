---
name: new-direction
description: Add a research direction to the lab. Interviews the owner, scaffolds it with draft gates, fake data and an L0 test, and opens a pull request. Use when the owner has a new idea to screen.
---

# New direction

## 1. Interview

Ask for what is missing, briefly. Do not fill gaps with guesses; write UNKNOWN.

- The single question the smoke test answers.
- Hypotheses, each with what result would falsify it.
- Data: source, licence, how a Kaggle kernel obtains it, whether it contains patient data, and whether
  pseudonymous subject IDs and per-subject metrics may be committed to this private repository.
- Budget: GPU hours cap for L1 and number of L1 rounds (default 2).
- What result would make the owner continue, and what would make them stop.

## 2. Scaffold

```bash
git switch -c direction/<slug>
python tools/new_direction.py <slug> --title "<one line>"
```

## 3. Draft

- `direction.yaml`: question, hypotheses, data, compute, outputs, `scorecard_map`, gates. Gates are
  boolean expressions over scorecard fields (`and`, `or`, `not`, comparisons). **Keep
  `approval: DRAFT`. Never write APPROVED.**
- `prereg.md`: hypotheses, data, design, exclusions, analysis, gates, open questions. Label claims
  FACT / INTERPRETATION / HYPOTHESIS / UNKNOWN. No invented citations.
- `src/`: the pipeline behind `entrypoint` (`run(config_path, out_dir)`).
- `src/evaluate.py` (required): everything that turns outputs into a gated number. All of `src/` is
  hash-locked together with `checks.py`.
- `configs/exp_001.yaml`: `compute.est_gpu_hours` and `params`.
- `checks.py`: what makes a finished run invalid. It may import only `labcore`'s dependencies.
- Outputs must not have phenotype columns (`labcore/privacy.py`); L0 fails if they do.
- `requirements.txt`: everything the kernel needs beyond the standard library.
- `tests/fake_data.py`: synthetic data in the **real file format**, with `make_dataset(...)` and
  `make_dryrun_config(work_dir, base_config)`; include the known failure cases.
- `tests/test_l0_smoke.py`: tiny settings through the local backend; assert outputs and scorecard.

## 4. Verify, then open the pull request

```bash
ruff check .
python tools/validate.py --run-l0 <slug>
python -m labcore.runs.dispatch <slug> exp_001 --dry-run
python -m labcore.watcher.watcher --dry-run
python tools/verdict.py <slug> --dry-run
```

All must pass. Open a pull request that states: the question, the draft gates and what each outcome
means, the GPU estimate and how it was obtained, open questions for the owner, and that the gates await
approval.

After merging, the owner approves the gates (`approval: APPROVED`), then L0 is re-run if code changed and
`python tools/lock_prereg.py <slug>` is committed through another pull request.
