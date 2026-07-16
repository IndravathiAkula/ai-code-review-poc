# AI Code Reviewer — Test Guide

A short, send-to-teammates guide to try out the AI reviewer on a real PR in
under 10 minutes.

---

## What You're Testing

A GitHub Actions workflow that automatically reviews every Pull Request with
an AI model and posts inline review comments. You'll set it up on a test repo
(or branch), open a PR with some intentional issues, and see what the AI
catches.

---

## Setup (3 Steps)

### 1. Add the workflow file

Copy `.github/workflows/ai-review.yml` from this repo into the repo you want
to test on (under `.github/workflows/`). Commit and push to your default
branch.

You don't need to copy anything else — the workflow pulls the reviewer code
itself at runtime.

### 2. Enable workflow permissions

In the test repo on GitHub:

**Settings → Actions → General → Workflow permissions** → select
**"Read and write permissions"** → Save.

### 3. (Optional) Use a paid provider

The default uses GitHub's free AI (`openai/gpt-4o-mini`). No API key needed.

To use Claude or Groq instead, add the API key as a repo secret
(*Settings → Secrets and variables → Actions*) and edit the workflow file.
See the "Switching Provider" section below.

---

## Run Your First Test

1. Create a new branch and add a file with an obvious bug. Example
   (`test.py`):

   ```python
   def get_user(user_id):
       query = "SELECT * FROM users WHERE id = " + user_id
       return db.execute(query)

   API_KEY = "sk-1234567890abcdef"
   ```

   That's a SQL injection plus a hardcoded secret — the reviewer should
   catch both.

2. Open a PR with that change.

3. Wait ~30 seconds. Refresh the PR.

You should see:
- Inline comments on the buggy lines, prefixed with `[AI-REVIEW]`.
- A summary comment at the top with token count, estimated cost, and timing.

If you don't see comments, check the **Actions** tab in your repo for errors.

---

## Things to Try

Once the basic flow works, try these to see how the reviewer behaves:

| Test | What to do | What to expect |
|---|---|---|
| **Clean PR** | Open a PR with just a rename or a comment change. | No findings posted. |
| **Multiple bugs** | Add 3-4 different bugs in one PR. | All should appear as separate comments. |
| **Re-run on push** | Push a new commit without fixing the bugs. | Existing AI comments stay in place (not duplicated). Stale ones get removed. |
| **Skip label** | Add label `skip-ai-review` to a PR and push a commit. | Workflow runs but exits immediately — no comments, no cost. |
| **Sensitive file** | Edit a `.env` file in a PR. | The file is skipped; no comments on it. |

---

## The YML File, in Plain English

The workflow file has three important parts:

**When it runs:**
```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]
```
Runs on every PR open, push, or reopen.

**What it's allowed to do:**
```yaml
permissions:
  contents: read
  pull-requests: write
  models: read
```
Read code, write PR comments, call AI. Don't remove any of these.

**The knobs (the `env:` block):**

| Variable | What it does | Default |
|---|---|---|
| `REVIEWER_PROVIDER` | Which AI service. | `github-models` |
| `MODEL` | Which model. | `openai/gpt-4o-mini` |
| `MIN_CONFIDENCE` | Drop findings below this confidence. | `0.6` |
| `MIN_SEVERITY` | Drop findings below this severity. | `low` |
| `REVIEWER_CONCURRENCY` | Parallel chunks. | `4` |
| `REVIEWER_MAX_FILES_PER_PR` | Cap on chunks reviewed. `0` = unlimited. | `0` |
| `REVIEWER_MAX_TOKENS_PER_PR` | Cap on total tokens. `0` = unlimited. | `0` |
| `REVIEWER_SKIP_LABELS` | PR labels that skip the review. | `skip-ai-review` |

---

## Switching Provider

To swap from GitHub Models (free) to Claude or Groq:

**Claude (best quality):**
```yaml
env:
  REVIEWER_PROVIDER: anthropic
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  MODEL: claude-sonnet-4-6
```

**Groq (fastest):**
```yaml
env:
  REVIEWER_PROVIDER: groq
  GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
  MODEL: llama-3.3-70b-versatile
```

Same pattern works for `openai`, `openrouter`, `nvidia`, `together`,
`anyscale`, `cerebras`, `ollama`.

---

## Tuning the Output

**Too many comments?** Raise the thresholds:
```yaml
MIN_CONFIDENCE: "0.75"
MIN_SEVERITY: medium
```

**Not enough findings?** Lower the confidence:
```yaml
MIN_CONFIDENCE: "0.5"
```

**Want to cap cost on huge PRs?**
```yaml
REVIEWER_MAX_FILES_PER_PR: "50"
REVIEWER_MAX_TOKENS_PER_PR: "200000"
```

---

## Common Issues

| Problem | Fix |
|---|---|
| Workflow didn't run | Confirm "Read and write permissions" is enabled. |
| `HTTP 401` in logs | API key missing or invalid. Re-add the secret. |
| `HTTP 429` in logs | Rate-limited. Lower `REVIEWER_CONCURRENCY` or switch provider. |
| No comments, but log says "0 findings" | AI had nothing to flag. That's fine. |
| Want to stop reviews entirely | Disable the workflow under *Actions → AI Code Review → Disable*, or delete the YML. |

---

## What to Report Back

After testing, share with the team:

- Did the reviewer catch the obvious bugs you planted?
- Were there false positives? (findings that aren't real bugs)
- How long did the review take?
- What was the estimated cost (visible in the summary comment)?
- Any provider/model that gave noticeably better results?

That feedback is what determines whether we roll this out more widely.

---

## Quick Reference

| To do this | Change this |
|---|---|
| Use a different AI model | `MODEL` in the YML |
| Use a different AI provider | `REVIEWER_PROVIDER` + secret |
| Make it stricter | Raise `MIN_CONFIDENCE` / `MIN_SEVERITY` |
| Cap cost | Set `REVIEWER_MAX_TOKENS_PER_PR` |
| Skip a single PR | Add `skip-ai-review` label |
| Disable for the whole repo | Delete the workflow file or disable it in Actions |
