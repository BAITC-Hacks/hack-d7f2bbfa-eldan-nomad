# hack-d7f2bbfa-eldan-nomad

Eldan Nomad's tariff campaign agent, following [PARTICIPANT_GUIDE.pdf](task/PARTICIPANT_GUIDE.pdf), sections 5 and 7-10.
The objective is net campaign ARPU lift after contact costs, including pilots and customer deduplication.
`customer_profile.csv` already supplies `predicted_arpu`; the agent learns campaign effects through `env.run_pilot`.

## Run the guide's checks

```bash
uv sync --group dev
source .venv/bin/activate
python tools/build_agent.py
python make_submission.py
python local_eval.py
python local_eval.py --runs 10
python make_submission.py --check
```

`agent.py` is generated from `agent_src/`; edit the source modules, then rebuild.
`make_submission.py` validates campaigns and compares two seed-42 runs before writing `submission.csv`.
`--check` verifies an existing submission without replacing it. Generation also rejects a stale `agent.py` bundle.

`local_eval.py` uses the organizer's unchanged `scoring_core.py`. Additional checks report malformed output,
missing pilots, resource violations, and (for the single seed-42 run) a missing or mismatched submission.
The campaign total in the scoring report includes pilots; the separate final-campaign count must be 1-10.
The ten-run command returns a failing exit status for invalid execution or any non-positive result.

## How the agent meets the guide

| Guide requirement | Implementation |
| --- | --- |
| §7: `Agent.act(env)` returns 1-10 valid campaigns | Orchestrator and final guardrails validate tariffs, channels and segment filters. |
| §§7-8: learn from pilots and account for uncertainty | Historical priors are updated with pilot observations; observation variance is `noise_sd² / n`. Calibration can adjust all arms. |
| §8: adaptive exploration | Expected information value, contact costs and remaining resources determine pilot size, channel and stopping; reserved pilots verify proposed deployments. |
| §8: meaningful channel choice | Allocation compares posterior lift times segment ARPU against each channel's contact cost and conversion multiplier. |
| §§7, 9: simultaneous limits | Pilot and final spending/reach share the same budget; exact simulation handles campaign caps and best-campaign deduplication. |
| §7: reproducible CSV | Fixed shortlist evaluation, deterministic tie-breaking, credential-independent default decisions, and submission replay checks. |
| §§8-9: optional LLM with fallback | Validated hypothesis weights and risk-review vetoes; errors/timeouts fall back to the statistical agent. |

## LLM mode

The default is `off`, even if `OPENAI_API_KEY` exists: the judge's credentials must not silently change a submission.
Set `AGENT_LLM_MODE=advise` for reports only, or `AGENT_LLM_MODE=decide` to apply hypothesis weights and valid vetoes.
The key comes from the environment (or a local, ignored `.env`); `OPENAI_MODEL` optionally selects the model.
Do not put credentials in source files.

For a single LLM experiment without comparing against the default-mode submission:

```bash
AGENT_LLM_MODE=decide python -c 'from agent import Agent; from local_eval import evaluate_agent; evaluate_agent(Agent(), seed=42)'
```

Live responses may vary; temperature zero alone does not guarantee reproducibility. The current release CSV uses
the default statistical mode. After changing code, data, configuration or LLM mode, regenerate and verify the CSV
in the same configuration that will be used for replay.

## Additional validation

```bash
python -m pytest -q
python tools/stress_eval.py --worlds all --runs 5 --no-oracle --json reports/stress_eval.json
```

Stress scenarios perturb effects, conversions, target ranking and prior accuracy. They are diagnostics for §8
robustness and §9's different hidden effects, not estimates of the judging score. `--no-oracle` skips the optional
oracle benchmark; the JSON records those unavailable metrics as `null`.

See [the requirement audit](reports/requirements_audit.md) for measured results and the purpose of each edit.
Submit `agent.py`, generated `submission.csv`, and `requirements.txt` as specified in §10.
