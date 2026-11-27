# Setup

About 15 minutes of your time, plus waiting for the first runs. Commands are for the repository root.
Anything marked **UNVERIFIED** could not be checked offline; the first-run checklist (step 7) checks it.

You need: git, Python 3.11 or newer, the [GitHub CLI](https://cli.github.com/) logged in (`gh auth login`),
a Kaggle account, and optionally `git-filter-repo` (for graduation; `git subtree` is the fallback).

## 1. Check it locally (3 min)

```bash
python -m venv .venv
. .venv/bin/activate                       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pip install -r directions/isles24-ctp-reg/requirements.txt
ruff check . && python -m pytest           # PowerShell 5.1: run the two commands separately
```

## 2. Set the GPU budget, commit, create the private repository (2 min)

Install the CLI (`kaggle` 2.2.5 needs `six` but does not declare it) and, after step 3, run `kaggle quota`
to see your weekly GPU allowance and its refresh time. Put matching `weekly_cap_hours` (default 25) and
`reset_weekday` (default `saturday`, **UNVERIFIED**) into `state/gpu_budget.json`.

```bash
python -m pip install "kaggle==2.2.5" six
git status                                 # read what will be committed
git add -A
git commit -m "Bootstrap research-lab"
gh repo create research-lab --private --source=. --push
```

The first push dispatches nothing. Later changes to `state/` cannot go through pull requests (CI blocks
them): change the cap with a direct commit to `main`.

If this folder sits inside another git repository, add it to that repository's ignore list.

## 3. Kaggle (3 min)

1. In Kaggle, verify your phone number if asked: GPU and internet access may require it (**UNVERIFIED**).
2. *Settings → API*: create a credential. Either kind works (**UNVERIFIED** which your account offers):
   - an API token → secret `KAGGLE_API_TOKEN`, plus `KAGGLE_USERNAME` (it names the kernel);
   - a legacy key (`kaggle.json`) → secrets `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. Never copy `kaggle.json` or a token into this folder.
4. Optional, only if the Hugging Face mirror needs a login: add `HF_TOKEN` as a **Kaggle** secret
   (notebook editor → *Add-ons → Secrets*). **UNVERIFIED:** whether a secret is attached to a kernel
   pushed through the API. You may have to attach it once in the web editor and push again.

## 4. GitHub secrets (3 min)

Each command prompts for the value, so it does not land in your shell history.

```bash
gh secret set KAGGLE_USERNAME
gh secret set KAGGLE_KEY                  # or: gh secret set KAGGLE_API_TOKEN
gh secret set DEEPSEEK_API_KEY            # optional: without it issues say "diagnosis unavailable"
gh secret set CLAUDE_CODE_OAUTH_TOKEN     # from `claude setup-token`; or ANTHROPIC_API_KEY instead
gh secret set TELEGRAM_BOT_TOKEN          # optional
gh secret set TELEGRAM_CHAT_ID            # optional
```

- Optional variables: `gh variable set DEEPSEEK_MODEL` (default `deepseek-flash`, **UNVERIFIED**) and
  `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com`).
- `HF_TOKEN`: you may store it with `gh secret set HF_TOKEN`, but **no workflow uses it** and it is never
  put into a kernel. The kernel can only read the Kaggle secret from step 3.

## 5. Actions permissions and the Claude app (2 min)

1. *Repository → Settings → Actions → General → Workflow permissions*: choose **Read and write
   permissions**. (The workflows also declare the permissions each job needs.)
2. Install the Claude GitHub app on this repository: <https://github.com/apps/claude>.
3. The bot workflows push `state/` and `results/` straight to `main`. If you protect `main` with a rule
   that requires pull requests, allow GitHub Actions to bypass it, or those commits fail (**UNVERIFIED**
   which bypass your plan offers). Without a rule, nothing technical stops a direct push: `CLAUDE.md`
   forbids it, you enforce it.

## 6. Labels (1 min)

```bash
gh label create run-l1 --color 0e8a16 --description "Run L1 on Kaggle after CI passes" --force
gh label create run-failed --color d73a4a --description "A Kaggle run failed" --force
```

**Review a pull request before labelling it `run-l1`.** Its code runs on Kaggle with your GPU quota, and
only the repository owner's label counts. (That check, and the one in `claude.yml`, assume a personal
repository: in an organisation no user equals the owner, so both would never run.)

## 7. First-run checklist

### 7a. Deliberately fail one kernel (almost no GPU time)

This checks the whole loop and, above all, **that logs of a failed kernel can be retrieved**.

```bash
git switch -c kaggle-check
python tools/new_direction.py kaggle-check --title "First-run check, expected to fail"
```

The template pipeline points at data that does not exist on Kaggle, so it fails within seconds. In
`directions/kaggle-check/direction.yaml` set `approval: APPROVED` yourself, then:

```bash
python tools/validate.py --run-l0 kaggle-check
python tools/lock_prereg.py kaggle-check
git add -A
git commit -m "kaggle-check: deliberate failure"
git push -u origin kaggle-check
gh pr create --fill
gh pr checks --watch                       # wait for CI to pass
gh pr merge --squash
```

Merging changes a config, so `dispatch` starts by itself. When it has finished, run the watcher (do not
wait for the hourly schedule): `gh workflow run watcher`. Repeat until the run is finished.

Expected: an issue labelled `run-failed`; class **code**; no retry; `error_log.txt` containing
`too few usable subjects`; `status.json` and `diagnosis.md` in `results/kaggle-check/...`.
Afterwards set `status: ARCHIVED` in `registry.yaml` and in its `direction.yaml` through a pull request.

### 7b. Verify every UNVERIFIED item

| # | Item | Where | How to verify |
|---|---|---|---|
| 1 | CLI authentication from the secrets | `client.py`, workflows | 7a: the dispatch log says `submitted`, no 401 |
| 2 | Text printed by `kernels push` | `client.py` | 7a: `state/runs.json` has a `kernel_version` |
| 3 | Accelerator `NvidiaTeslaT4` | `metadata.py` | 7a: `env.json` `gpu_name` contains `T4` (`T4 x2` on the kernel page) |
| 4 | Kernel names up to 50 characters | `metadata.py` | 7a: push accepted; try a long experiment name if you use one |
| 5 | `-t` stops the session after twice the estimate | `client.py` | kernel page shows the limit, or watch the first long run |
| 6 | Status texts and names | `client.py` | watcher log shows `QUEUED`/`RUNNING`, then `ERROR` |
| 7 | Version-suffixed ref `owner/slug/N` | `client.py` | same: no "status check failed" in the watcher log |
| 8 | Outputs and log of a failed kernel | `client.py` | 7a: `error_log.txt` has the Python traceback, `status.json` exists |
| 9 | `/kaggle/working` is the output folder | `entrypoint_template.py` | 7a: `env.json` is in the results |
| 10 | `pip install` inside the kernel | `entrypoint_template.py` | first ISLES log: `installing: ...` succeeds, or `all requirements already installed` |
| 11 | Zenodo and the Hugging Face mirror from a kernel, and how fast | `isles24-ctp-reg/src/data.py` | ISLES log: `fetched ... GB in ... min`; `cases.csv`: statuses `ok` |
| 12 | `/kaggle/temp` as cache, and its free space | `isles24-ctp-reg/src/main.py` | ISLES log: `free disk for the cache: ... GB`; no disk errors |
| 13 | Kaggle secret on an API-pushed kernel | `data.py` | only if downloads return 401/403 |
| 14 | Weekly quota and reset day | `state.py` | `kaggle quota` |
| 15 | `concurrency.queue: max` | `dispatch.yml`, `watcher.yml` | Actions tab shows both workflows without "Invalid workflow file". If rejected, delete the `queue` lines |
| 16 | Issue search finds an existing issue | `notify/github.py` | two successful runs share one `[tracking]` issue |
| 17 | Fetching a submitted commit by its sha | `gitutil.py` | after a run from a squash-merged `run-l1` PR: the tracking issue shows a computed verdict, not "No verdict: ..." |
| 18 | Claude action inputs, `id-token`, `claude_args` | `claude.yml` | comment `@claude which rules apply to this repository?` on an issue |
| 19 | DeepSeek model name and thinking switch | `deepseek.py` | 7a: `diagnosis.md` is a summary, not `request failed` |

Remove an `# UNVERIFIED:` comment only after you have seen it work.

## 8. First real L1 run (ISLES'24)

1. Read `directions/isles24-ctp-reg/prereg.md`, especially the open questions about the gates.
2. Decide whether pseudonymous subject IDs and per-subject metrics may be committed to `results/` under the
   dataset licence (CC BY-NC-SA 4.0) and your institution's rules. If not, do not run it from this lab yet.
   (Phenotype columns in `results/` are refused automatically; subject IDs are not.)
3. In a branch: adjust the gates if needed and set `approval: APPROVED`. Then
   `python tools/validate.py --run-l0 isles24-ctp-reg` and
   `python tools/lock_prereg.py isles24-ctp-reg`; merge.
   From here until the verdict, the locked files and `labcore.__version__` cannot change.
4. *Actions → dispatch → Run workflow*: `slug` = `isles24-ctp-reg`, `exp` = `exp_001_pilot`.
   (A merge that adds or changes a config of a locked direction dispatches it by itself.)
5. When the tracking issue reports success: `git pull`, then on a branch
   `python tools/verdict.py isles24-ctp-reg`, write the interpretation below the marker, open a pull
   request.

## Operating notes

- The watcher cannot cancel a kernel. After a `TIMEOUT`, stop it on Kaggle yourself: a queued or running
  version may still use GPU time, also after the automatic retry was submitted.
- A run refused for the weekly cap is queued; start it later with *Run workflow* and `drain_queue`.
- Actions minutes: see the comment at the top of `watcher.yml`.
- Scheduled workflows run only from the default branch and can start late.
