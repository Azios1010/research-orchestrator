# Pre-registration: __TITLE__

Direction `__SLUG__`, drafted __DATE__. **Status: DRAFT, pending owner approval.**
Once locked (`tools/lock_prereg.py`), this file, the gates, `scorecard_map`, `checks.py`,
`src/evaluate.py` and the labcore version cannot change.
Label every claim FACT / INTERPRETATION / HYPOTHESIS / UNKNOWN. Cite nothing you have not checked.

## Question

TODO

## Hypotheses

- HYPOTHESIS H1: TODO. Falsified if TODO.

## Data

- UNKNOWN: source, version, licence, number of subjects, inclusion and exclusion rules.

## Design

- Arms or methods compared:
- What is held identical between them:
- Splits (at the subject level):
- Seed:

## Primary outcome and analysis

- Per-subject metric:
- Contrast and uncertainty: paired bootstrap over subjects, 95% CI.

## Gates (copied from direction.yaml, which is authoritative)

- GO: `ci_lo(a-b) > 0`
- PARK: `ci_hi(a-b) <= 0`
- REVISE: everything else

## Validity requirements

A run counts only if the sanity checks in `checks.py` pass.

## Known limitations

- UNKNOWN
