---
name: implement
description: Implement or modify the code and config of one experiment (arguments slug and exp), run L0 locally and open a pull request with the GPU estimate.
---

# Implement `<slug> <exp>`

1. Read `registry.yaml`, then `directions/<slug>/direction.yaml`, `prereg.md`, the config and `src/`.
2. **Check the status.** At `L1_RUNNING` or later, `prereg.md`, gates, evaluation code (`checks.py`,
   `src/evaluate.py`, `scorecard_map`, metrics, statistics, preprocessing, folds, seeds, arms,
   exclusions), `labcore.__version__` and configs with a run are frozen. Any edit under `src/` breaks the lock (hash scheme 2 covers all of it, even logging), so it
   needs a new round. Dependencies in `requirements.txt` and a new config are outside the lock; say why such
   a change cannot affect results. Anything else: stop and propose a new round or a new direction.
3. Work on a branch. Implement the change and `configs/<exp>.yaml` with `compute.est_gpu_hours`.
   The estimate may not exceed `compute.l1_gpu_hours_cap`; the kernel is timed out at twice the estimate.
4. Extend the L0 test and fake data for what changed.
5. Verify:

   ```bash
   ruff check .
   python -m pytest
   python tools/validate.py --run-l0 <slug>
   python -m labcore.runs.dispatch <slug> <exp> --dry-run
   python -m labcore.watcher.watcher --dry-run
   ```

6. Open a pull request. State what changed, the L0 result, the GPU estimate and its basis (measured,
   scaled from a run, or a guess), hours left under the direction cap and this week's cap (read
   `state/runs.json` and `state/gpu_budget.json`; do not edit them), and that merging a config change
   to `main` **dispatches it automatically** if the direction is locked.
