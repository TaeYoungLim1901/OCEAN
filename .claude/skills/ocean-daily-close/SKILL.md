---
name: ocean-daily-close
description: Record the three OCEAN Public scores and create a private GitHub snapshot after 2026-08-20. Use only when P1, P2, and P3 each have the actual submission CSV, matching prediction NPY, and returned Public score; use the existing score skill for local evaluation only.
---

# OCEAN Daily Close

Use one coordinator command and one delegated Luna call. Do not run this
workflow before 2026-08-20, before all three Public scores are known, or while
another P session, long training job, or Codex call is active.

## Coordinator

1. Collect the date, common note, and for each of P1/P2/P3: actual submission
   CSV, matching prediction NPY, and Public score. Stop and request anything
   missing; never guess a Public score.
2. Confirm the date is at least `2026-08-20` and no later than Seoul local
   today. Write one new `P3/specs/OPS-YYYY-MM-DD-daily-close.md`; never
   overwrite an existing spec.
3. Put the following delegated procedure, prohibitions, inputs, and numeric
   report fields in that spec. Run:
   Base the spec on `P3/specs/_TEMPLATE.md`; include `python`, and stop if
   `pip install` is needed, plus background/research-prohibited facts, numeric
   baseline, acceptance criteria, output-path existence checks, and the exact
   rule `결론 문장 금지, 숫자와 표만`.

   ```powershell
   python scripts/preflight.py spec P3/specs/OPS-YYYY-MM-DD-daily-close.md
   scripts\cx.cmd -f P3/specs/OPS-YYYY-MM-DD-daily-close.md
   ```

   Call Luna once. Keep score recording and GitHub push in that one call.

## Delegated Luna procedure

1. Do not recursively call `cx.cmd`. Validate every path is an existing file
   under its matching `P{n}/subs` or `P{n}/preds`; validate finite scores,
   `0..1` for P1 F1, and `>=0` for P2/P3 RMSE.
2. Run all three first: `python P{n}/src/submission_tools.py audit <csv>
   --pred <npy>`. Any nonzero exit means score and Git changes are `0`.
3. Read each problem `experiments/public_scores.csv`. Same SHA and same score:
   skip that problem. Same SHA and different score: stop without overwrite.
4. For remaining problems run `python P{n}/src/submission_tools.py public add
   <csv> <score> --note <note>`, then run `public list` for P1/P2/P3. On a
   partial recording failure, do not touch Git; report failed/already-recorded
   problem numbers. A retry follows step 3.
5. Before Git, run `gh auth status` and
   `gh repo view TaeYoungLim1901/OCEAN --json nameWithOwner,isPrivate`.
   Require the exact private repository. If `.git` is absent, run `git init -b
   main` and add the exact HTTPS origin. Reject another fetch/push URL or a
   non-`main` branch.
6. Never use `git add .`. Stage only `.gitignore`, `.claude`, `docs`,
   `scripts`, root operational Markdown, and each problem's `AGENTS.md`,
   `graph.json`, `src`, `specs`, `experiments/log.md`, and
   `experiments/public_scores.csv`, using explicit `git add -A --` paths.
7. Before commit, print staged names and bytes. Reject data, cache, models,
   preds, subs, experiment logs, `.env`/`.pem`/`.key`, binary/model/pred
   extensions, secret candidates, deleted files, or any file over 50 MB. Run
   `git diff --cached --check`; on failure restore only the index, never the
   worktree.
8. With staged changes, commit `scores: YYYY-MM-DD public results`; otherwise
   commit `0`. Push with `git push -u origin main`, then compare local HEAD
   and `origin/main`. After a previous push failure, skip score recording and
   push the existing commit.
9. Report a table only: each audit exit/SHA/Public/delta-best, record/skip,
   staged file count/bytes, commit SHA, local/remote SHA, and init/commit/push
   counts. Do not update graph/log decisions automatically.

## Prohibitions

Do not push before 2026-08-20; record only after all three scores; upload no
submission, prediction, model, data, or guessed score. Do not add scheduler,
CI, LFS, lock, queue; switch/merge/rebase branches; reset/checkout worktrees;
or use `git add .`.
