# Final release decisions — 23 September 2026

Guide: [PARTICIPANT_GUIDE.md](../task/PARTICIPANT_GUIDE.md), sections 7–10.

## Submit this configuration

Use the existing statistical allocation policy with `AGENT_LLM_MODE=off`.
The release adds an economic acceptance check for optional LLM vetoes and an
external submission checker. The seed-42 CSV and default net ARPU are unchanged.

- Seed 42: net **4,814,532**, gross 4,914,516, cost 99,984, contacts 14,960.
- Final campaigns: **9**; pilots: **18**; no discarded or capped campaigns.
- Mock seeds 0–9: **10/10 positive**; mean 4,464,661; median 4,418,069;
  minimum 3,670,926.
- Full suite: **381 passed**.
- Preflight: current source bundle, valid pilots/resources, two seed-42 replays
  exactly match `submission.csv` without rewriting it.

## Changes included

1. **Numerical check after LLM review.** The existing RiskReviewer can propose a
   veto. Python now evaluates the actual repacked candidate, accounts for pilot
   overlap, and applies it only when predicted net does not decrease and all
   resource/validity checks pass. Rejection or errors preserve the original plan.
   The check applies to cached proposals as well. This constrains predicted
   economics; it cannot establish the unknown true effect.
2. **Submission preflight.** `python tools/preflight.py --check` detects a stale
   bundle, invalid raw campaigns, missing pilots, excessive resource demand,
   runtime violations, nondeterministic replay, and a mismatched CSV. It also
   rejects scorer truncation, a stricter diagnostic than the organizer requires.
   Organizer evaluation and generation scripts remain unchanged.

## Optimizer experiments not included

We screened nine simulated effect worlds with seeds 0–2, then used fresh
pilot-noise seeds 100–104 for the promising alternatives. These are the same
simulated worlds with new noise, not the unknown judging distribution.

| Candidate | Evidence | Decision |
|---|---|---|
| Stronger confidence threshold in coarse packing | Lower mean in all nine screening worlds; introduced a negative permuted-target run | Reject |
| Exact piecewise posterior calculation in knowledge gradient | Matches the channel-transfer mathematics, but changed exploration and materially regressed the prior-misspecified scenario | Defer until the exploration policy can be reassessed |
| Exact calculation plus staged spending on untested arms | Mock mean +1.63%, minimum +14.03%; fresh stress aggregate mean +0.17%, but profitable runs fell from 43/45 to 42/45 | Reject for this deadline |
| Exact calculation plus smaller verification pilots when 200 contacts are uneconomic | Fresh prior-misspecified mean −12.42%; mock seeds 0–9 mean −1.50% | Reject |

The staged-spending candidate's attractive mock result is not a reliable claim
of higher judging ARPU. The current baseline also had two losing runs in the
fresh sign-flipped world, so stress profitability should always name its seeds.
Detailed paired observations: [release_experiments.json](release_experiments.json).

## LLM judge recommendation

Use the existing HypothesisAnalyst and RiskReviewer for the demonstration;
`advise` produces explanations while retaining deterministic decisions. `decide`
is an opt-in experiment: even with the new veto check, hypothesis weights and
live responses can alter exploration and replay. No live LLM advantage was
established in this review, so default `off` is the submission configuration.

## Final commands

```bash
source .venv/bin/activate
python tools/build_agent.py
python local_eval.py
python local_eval.py --runs 10
python make_submission.py
python tools/preflight.py --check
python -m pytest -q
```

Required artifacts: `agent.py`, `submission.csv`, `requirements.txt`.
