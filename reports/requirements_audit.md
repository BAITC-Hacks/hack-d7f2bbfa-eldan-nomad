# Participant-guide compliance audit

Source of truth: [PARTICIPANT_GUIDE.pdf](../task/PARTICIPANT_GUIDE.pdf), especially page 3, sections 7-10.
All amounts below are synthetic task units. Scores include pilot effects/costs and use the organizer's scoring formula.

## What already existed

The starting implementation already provided `Agent.act`, Bayesian uncertainty, adaptive knowledge-gradient pilots,
deployment-verification pilots, global prior calibration, channel-cost allocation, exact scoring simulation,
resource guardrails, and an optional LLM analyst/reviewer. These features were preserved, not reimplemented.

At the start of this audit, 354 tests passed. The current starting snapshot produced the same 9-campaign seed-42
CSV in three runs, matching the existing submission. The older conversation's 10-versus-6 mismatch was not
reproduced with this snapshot; its cause cannot be established from those earlier runs.

## Edits and their guide basis

| Edit | Guide basis and effect |
| --- | --- |
| Clear the planner's posterior cache before each decision | §§7.3, 8: calibration updates arms without incrementing their observation counts. Previously those arms could retain stale estimates and affect pilot ranking. |
| Evaluate the complete deterministic `top_arms` shortlist | §7.5: an elapsed-time cutoff could change candidates under CPU load. The overall run deadline remains enforced between decisions. |
| Default LLM mode is explicitly off | §§7.5, 9: adding the judge's API key must not silently change campaign selection. Explicit `advise` and `decide` modes remain supported. |
| Validate raw output in local evaluation | §§7.1-4: detect invalid counts/tariffs/channels before sanitization masks them; check pilot use and combined resources. Scoring formula is unchanged. |
| Separate final campaign counts and report remaining resources after all campaigns | §§7.2, 7.4, 9: avoid confusing pilot rows with final campaigns or reporting pilot-only remaining budget as final remaining budget. |
| Submission validation, repeat check and stale-bundle check | §§7.5, 10: generate the CSV from current code; reject drift and preserve an existing file on validation failure. |
| Stress evaluator checks validity and returns nonzero for execution failures | §§8, 9: a printed score must not conceal an invalid strategy or missing pilots. |
| Regression and integration tests | §§7-9: exercise stale calibration, slow clocks beyond 12 candidate arms, larger-pilot precision, counterfactual pilot feedback, raw validation and cross-process replay. |
| README and this audit | §10: document the method, submission commands and evidence. |

No changes were made to `scoring_core.py`, `environment.py`, the provided datasets or the supplied `predicted_arpu` baseline.
No new constants were fitted to hidden effects or to improve one particular mock scenario.

## Must-have results: seed 42

| Requirement | Evidence |
| --- | --- |
| Runnable `agent.py` with `Agent.act(env)` | `python local_eval.py` completes; execution checks PASS. |
| 1-10 valid final campaigns | 9 final campaigns, none discarded. The scoring report's 27 total rows include 18 pilots. |
| Pilots used and results affect decisions | 18 pilots; all observations enter the model. A counterfactual test uses identical priors/profile and opposite pilot effects: positive evidence leads to deployment, negative evidence to zero-contact output. |
| Limits satisfied | Cost 99,984 / 100,000; contacts 14,960 / 15,000; 18 / 20 pilots; largest final campaign 4,726 / 5,000 customers. |
| Generated, reproducible submission | `python make_submission.py` checks two runs. Independent processes with different hash seeds and API-key presence produce identical default-mode CSVs. |

Seed-42 gross lift: **4,914,516**. Net lift: **4,814,532**, or **+3.196%** of the supplied baseline.
Baseline total is 150,641,084; total after campaigns is 155,455,616. These are sums, not per-customer ARPU.

## Optional features and measurements

All five optional features have implementations. Tests verify that a 150-customer pilot has lower posterior
uncertainty than a 30-customer pilot, and that observed effects influence final deployment.
Across mock seeds 0-9, the agent uses 13-20 pilots and returns 6-10 final campaigns; all ten net results are positive.
The seed-42 final plan uses `push`, `sms`, and `digital_ads`; pilots also use `call`.

Paired comparison with the code snapshot taken immediately before these edits (LLM off, identical seeds/data):

| Metric, mock seeds 0-9 | Before | After |
| --- | ---: | ---: |
| Mean net gain | 4,435,238 | 4,464,661 |
| Median net gain | 4,418,069 | 4,418,069 |
| Minimum net gain | 3,670,926 | 3,670,926 |
| Maximum net gain | 5,122,772 | 5,122,772 |
| Positive runs | 10/10 | 10/10 |

Mean improvement is **0.66%**; seed 2 improved from 4,597,217 to 4,891,455. The other nine mock seeds were unchanged.
This is a small measured gain, not evidence of better baseline forecasting or guaranteed hidden-world performance.

The final [stress report](stress_eval.json) covers nine worlds with five seeds each: **45/45 profitable, no execution
or requirement errors**. The weakest scenario is permuted target rankings (minimum net +353,425).
Sign-flipped effects remain positive in all five runs. Some scenario scores decrease versus the starting snapshot;
the cache correction is not a universal ARPU improvement. Oracle comparisons were not run.

Live LLM validation used a separate temporary cache and a 60-second LLM budget. The hypothesis analyst timed out
and safely used neutral weights. The risk reviewer returned a validated veto, reducing the plan to 8 campaigns;
net gain was 4,814,432. This verifies a live decision path and graceful fallback, but shows no advantage over
the default statistical plan on this seed. Offline LLM tests also cover structured outputs, caching, veto limits,
invalid responses, errors and timeouts.

## Verification record

- Full suite: **363 passed**.
- Generated `agent.py` exactly matches the source bundle.
- Current submission contains 9 campaigns and matches seed-42 generation.
- Agent SHA-256: `fda6f09b80dc33496079edba38ef9dd4079baa18849660053872baad229e9015`.
- Submission SHA-256: `fda68638e90a2ad8cb42950227f8b00ff240b920f258bed95600c4b9fff0259a`.
- Starting agent SHA-256: `ecc3b0b3874a930c3b41f3f1a6c9932b8e56362e344f7817e5440ff0c9562949`.

Per §9, the real judging effects differ from the mock. These checks establish mechanics and observed robustness,
not the future judging score. Re-run the guide's checks after changing the agent or submission configuration.
