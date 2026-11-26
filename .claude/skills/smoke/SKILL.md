---
name: smoke
description: Check that an experiment (arguments slug and exp) is ready for a real GPU run - L0 green, lock in place, budget available - and explain how to trigger L1. Does not submit anything.
---

# Smoke `<slug> <exp>`

**Never push to Kaggle from a local session unless the owner explicitly asks.** This skill only checks.

1. L0 and the offline loop:

   ```bash
   python tools/validate.py --run-l0 <slug> --no-record
   python -m labcore.runs.dispatch <slug> <exp> --dry-run
   python -m labcore.watcher.watcher --dry-run
   ```

2. Lock: `registry.yaml` must show `L1_RUNNING` with a `prereg_lock`, and
   `python tools/ci/prereg_lock.py` must pass. If not: the owner approves the gates, then
   `python tools/lock_prereg.py <slug>` goes through a pull request.
3. Budget (after `git pull`; read only):
   - direction: hours of its runs in `state/runs.json` + `est_gpu_hours` <= `l1_gpu_hours_cap`;
   - week: `used_hours` in `state/gpu_budget.json` + estimate <= `weekly_cap_hours`;
   - no active run of the same experiment, and no identical completed run.
4. Report each item as pass or fail with the number behind it. If all pass, tell the owner how to start:
   - add the label `run-l1` to the pull request that changes the config (CI runs first, then dispatch);
   - or merge that pull request into `main` (dispatches changed configs of locked directions);
   - or Actions, workflow `dispatch`, *Run workflow* with `slug` and `exp`.
