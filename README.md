# research-lab

A private screening lab for research directions. Many directions sit side by side, each gets a cheap
smoke test, and only those that pass **pre-registered gates** move out into their own repository.

```
IDEA -> SCAFFOLDED -> L0_PASS -> L1_RUNNING -> VERDICT(GO | REVISE | PARK) -> GRADUATED | ARCHIVED
```

| Level | What runs | Where | Cost |
|---|---|---|---|
| L0 | the whole kernel on synthetic data in the real file format | CPU, CI on pull requests | Actions minutes |
| L1 | a small run on real data, producing a scorecard | Kaggle GPU, internet on | capped GPU hours |
| Verdict | gates from `direction.yaml`, applied by script | your machine, via pull request | none |

When a direction enters L1, `prereg.md`, the gates, the scorecard map, `checks.py`, everything under `src/`
and the labcore version are hash-locked. An LLM may write interpretation; it never decides a verdict.

## Quick start (offline, no accounts)

Python 3.11 or newer.

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip install -r directions/isles24-ctp-reg/requirements.txt

ruff check .
python -m pytest                     # includes the offline end-to-end loops
python tools/validate.py

# the loop by hand, on fake data; everything goes to .dryrun/
python -m labcore.runs.dispatch isles24-ctp-reg exp_001_pilot --dry-run
python -m labcore.watcher.watcher --dry-run
python tools/verdict.py isles24-ctp-reg --dry-run
```

Connecting Kaggle, GitHub and the optional services: [SETUP.md](SETUP.md).

## Everyday commands

| Task | Command |
|---|---|
| New direction | `python tools/new_direction.py <slug> --title "..."` |
| L0 (records `L0_PASS`) | `python tools/validate.py --run-l0 <slug>` |
| Enter L1 (after you approve the gates) | `python tools/lock_prereg.py <slug>` |
| Start a real run | label a pull request `run-l1`, merge a config change, or run the `dispatch` workflow |
| Record a verdict | `python tools/verdict.py <slug>` |
| Second round after REVISE | `python tools/lock_prereg.py <slug> --new-round` |
| Graduate after GO | `python tools/graduate.py <slug> --out ../<slug>` |

## Directions

| Slug | Question | Status |
|---|---|---|
| `isles24-ctp-reg` | Raw 4D CTP vs perfusion maps for follow-up infarct segmentation (ISLES'24 pilot), CTP registered to NCCT | see `registry.yaml`; **gates are DRAFT** |
| `isles24-ctp` | The same question from the mirror's files | `ARCHIVED`: its run stopped before training, the CTP was not on the lesion-mask grid |

## Layout

```
labcore/       library: bundle, Kaggle client, dispatch, watcher, statistics, verdict
directions/    one folder per direction; _template/ is copied for new ones
tools/         new_direction, validate, lock_prereg, verdict, graduate, ci/
state/         runs, queue, GPU budget (written by bot workflows only)
results/       <slug>/<exp>/<run_id>/ (written by bot workflows only)
registry.yaml  lifecycle of every direction
docs/          ARCHITECTURE.md, DECISIONS-bootstrap.md
reference/     inputs that are only read
.claude/       skills for Claude Code; rules in CLAUDE.md
```

No secrets and no data belong in this repository.
