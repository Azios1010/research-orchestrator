---
name: graduate
description: Split a direction with a GO verdict (argument slug) into its own standalone folder with history, and list the manual steps. Creates no remote or GitHub repository.
---

# Graduate `<slug>`

1. Only for a direction whose registry status is `VERDICT` with `verdict: GO`. Otherwise stop and say why.
2. On a branch, with `main` merged and the direction fully committed:

   ```bash
   python tools/graduate.py <slug> --out ../<slug>
   ```

3. Check the new folder: `README.md`, `VENDORED.md`, history (`git -C ../<slug> log --oneline`), and in a
   fresh environment `python -m pip install -e ".[dev]" -r requirements.txt && python -m pytest`.
4. Open a pull request with the registry change (status `GRADUATED`).
5. Print the remaining manual steps exactly as the tool printed them (review, commit in the new folder,
   `gh repo create`). **Do not run `gh repo create`, add remotes or push** unless the owner asks.
