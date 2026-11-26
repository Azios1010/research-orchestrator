---
name: verdict
description: Record a direction's verdict (argument slug) with tools/verdict.py, then write the interpretation and threats to validity below the marker. Never decides the verdict itself.
---

# Verdict `<slug>`

1. `git pull`, then work on a branch.
2. Run `python tools/verdict.py <slug>` (add `--run-id <id>` only if the owner chooses a run; say why).
   The script applies the locked gates and writes the top of `VERDICT.md` and the registry. **You do not
   decide, adjust or second-guess GO / REVISE / PARK.** If it refuses, report the message. Never use
   `--allow-local` for a real verdict.
3. Read the scorecard, `checks.json` and the other files of that run, and `prereg.md`.
4. Edit only below `<!-- VERDICT:INTERPRETATION ... -->`:
   - **Interpretation**: what the numbers do and do not show, every claim labelled FACT /
     INTERPRETATION / HYPOTHESIS / UNKNOWN. Quote numbers exactly as the files give them.
   - **Threats to validity**: at least sample size and interval width, exclusions and failed subjects,
     deviations from the pre-registration, leakage risks, confounds between arms, other runs of the same
     round (they are listed in the block), multiplicity, and what the data cannot say.
5. Open a pull request. For REVISE, name what a second round could change, without editing anything yet.
