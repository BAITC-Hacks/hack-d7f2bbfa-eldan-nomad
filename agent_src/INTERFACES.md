# agent_src — interface contract

Authoritative design: `plans/beeline_agent_plan.md`. This file fixes the **module boundaries**.
Follow signatures exactly; extensions = new *optional* params only (document them in your summary).

## Conventions

- **Bundle rule.** Every `agent_src/m*.py` starts with
  ```python
  from __future__ import annotations  # bundle:strip
  from agent_src.contract import *  # noqa: F401,F403  bundle:strip
  ```
  Imports of other agent_src modules are ONLY via lines whose trailing comment ends with `bundle:strip`, e.g.
  `from agent_src.m40_simulator import ScoreSimulator  # bundle:strip`.
  Stdlib / third‑party imports are normal lines at the module top (`import numpy as np`, `import asyncio`, …).
  `from agent_src.contract import *` re-exports `np`, `pd`, `math`, `json`, `os`, `time`, `hashlib`, `dataclasses`,
  `dataclass`, `field`, `Any`, `Callable`, `Iterable`, `Optional` — but import what you use explicitly anyway (harmless in the bundle).
- **Build.** `.venv/bin/python tools/build_agent.py` → `agent.py` = header docstring + one `from __future__ import annotations`
  + `contract.py` + `m*.py` sorted by filename. Stripped lines: any `from __future__` line and any line whose `#` comment ends
  with `bundle:strip`. Missing modules → warning, skipped. The result is `py_compile`d.
- **Single namespace.** In `agent.py` all modules share one global namespace: module‑level names must be unique
  across modules. Prefix private helpers with the module tag (`_dv_…`, `_pr_…`, `_sim_…`, `_alloc_…`, `_pack_…`, `_kg_…`, `_llm_…`, `_gr_…`, `_orc_…`).
  Never rely on `__name__`, module attributes, or `agent_src.X.Y` qualified access at runtime.
- **Forbidden tokens** (build fails, case-insensitive for words): `__closure__`, `import gc`/`gc.`, `inspect`, `internals` —
  also in comments/docstrings/strings. The agent only touches `env.customer_profile`, `env.tariffs`, `env.channels`,
  `env.remaining_budget`, `env.remaining_contacts`, `env.pilots_left`, `env.run_pilot(...)`, `env.pilot_history`.
  Reading `data/change_tariff.csv` and `tariff_dictionary.csv` (prior knowledge) is allowed; no other organizer files.
- **Determinism.** No unseeded randomness; RNG = `np.random.default_rng(cfg.seed)`. Iterate dicts/sets in sorted order
  whenever the order affects output. Ties broken by key tuples.
- **No prints.** Log through `RunLog.log(kind, **fields)` passed in. Secrets only from `os.environ["OPENAI_API_KEY"]`; never log them.
- **Tests.** `tests/test_<module>.py` (e.g. `tests/test_m40_simulator.py`), importing from `agent_src.*`; use fakes for other
  modules. < 20 s per file. Shared fixtures in `tests/conftest.py`:
  `repo_root`, `mock_env_factory(seed) -> env` (session; chdir to repo), `synth_channels`, `synth_tariffs` (4 tariffs),
  `synth_profile` (120 rows, tariffs 1..3, shuffled IDs, NaN in rows 0..2).
  Run: `cd <repo> && .venv/bin/python -m pytest -q tests/test_<module>.py`.

## Module ownership

| File | Owns |
|---|---|
| `contract.py` | types, Config, helpers (below) |
| `m10_dataview.py` | `DataView`, `load_history`, `load_tariff_descriptions` |
| `m20_prior.py` | `PriorBuilder` |
| `m30_arm_model.py` | `ArmModel` |
| `m40_simulator.py` | `ScoreSimulator` |
| `m50_allocator.py` | `Allocator` |
| `m60_packer.py` | `CampaignPacker` |
| `m70_planner.py` | `PilotPlanner` |
| `m80_llm.py` | `LLMLayer` (+ pydantic output models, built lazily) |
| `m85_guardrails.py` | `Guardrails`, `Reporter` |
| `m90_orchestrator.py` | `Orchestrator`, `run_coro`, `Agent` |

## contract.py

```python
CellKey = tuple[str, str]            # (current_tariff, arpu_segment)
SubKey  = tuple[str, str, str, str]  # (current_tariff, arpu_segment, data_segment, call_segment)
ArmKey  = tuple[str, str, str]       # (current_tariff, arpu_segment, target_tariff)
CHANNELS_ORDER = ("push", "sms", "digital_ads", "call")
ARPU_SEGMENTS = ("LOW", "MID", "HIGH"); DATA_SEGMENTS = ("NON_USER", "LITE", "HEAVY"); CALL_SEGMENTS = ("LOW", "MEDIUM", "HIGH")
CAMPAIGN_KEYS = ("campaign_name", "filter_arpu_segment", "filter_data_segment", "filter_call_segment",
                 "filter_current_tariff", "target_tariff", "channel")
LLM_MODES = ("off", "advise", "decide")
```

- `@dataclass(frozen=True) Config` — fields/defaults: `time_budget_s=420.0, llm_budget_s=90.0, llm_call_timeout_s=30.0,
  noise_sd=0.804, z_risk=1.0, p_min=0.8, p_min_push=0.6, tau=0.2, kappa_up=0.9, conv_prior_default=0.3, shrink_k=20.0,
  explore_money_frac=0.25, explore_reach_frac=0.30, max_pilots=20, pilot_sizes=(30,60,100,150,200), min_pilot=10, max_pilot=200,
  max_campaigns=10, max_per_campaign=5000, top_arms=40, top_targets_per_cell=3, max_pilots_per_arm=3, gh_nodes=5, seed=42,
  schema_version="v1", llm_model="gpt-4.1-mini", report_path="agent_report.md", cache_file="llm_cache.json"`.
  - `Config.from_env(**overrides)`: `OPENAI_MODEL` (non-empty) → `llm_model`; then explicit kwargs. LLM mode is NOT a field.
  - `cfg.replace(**changes)` → copy (for calibration / tests).
- `resolve_llm_mode() -> str`: `AGENT_LLM_MODE` (case-insensitive). `off` → off. `advise`/`decide` → that mode **if**
  `OPENAI_API_KEY` non-empty, else `off`. Unset/invalid → `decide` if key set else `off`.
- `SubCell(key, n, sum_p, mean_p, ids: np.ndarray[int64] sorted, p: np.ndarray[float] aligned to ids)`.
- `Cell(key, n, sum_p, mean_p, subs: list[SubKey])` — `subs` sorted; `n/sum_p` over rows with non-NaN tariff+arpu
  (includes rows whose data/call segment is NaN, which belong to no sub).
- `Prior(mu, sd, share, n_hist, source)` — `mu/sd` = base lift ratio at channel multiplier 1.0 (≈ change·share);
  `share` = conversion estimate in (0,1]; `source ∈ {'arm','cell','global','price','none'}`.
- `Observation(arm, channel, y, n, cost, sub, pilot_index)` — `y = observed_lift_ratio`, `n = result["n_customers"]`
  (actual), `pilot_index` = 0-based index into `env.pilot_history`.
- `Posterior(mean, sd)` — lift ratio (fraction of predicted_arpu) for a given arm AND channel (multiplier included).
- `Option(sub, target, channel, n, cost, net_mean, net_sd, net_lcb, p_pos)` — `n` = contacts (= sub.n unless reduced),
  `cost = cost_ch·n`.
- `Plan(options, lambda_money, lambda_reach, total_net_lcb, total_cost, total_contacts)` — at most one option per sub.
- `PilotSpec(arm, channel, n, sub, filters, score, reason)` — `filters` = `run_pilot` filter kwargs
  (`filter_arpu_segment`, `filter_data_segment`, `filter_call_segment`, `filter_current_tariff`; None = no filter).
  `spec.run_kwargs()` → full kwargs for `env.run_pilot(**kw)` (target from `arm[2]`).
- `ExploreState(remaining_budget, remaining_contacts, pilots_left, explore_money_spent, explore_reach_spent,
  pilots_per_arm: dict[ArmKey,int] = {}, used_subs: dict[SubKey,int] = {}, deadline: float = inf)` — `deadline` is
  `time.monotonic()`-based; `state.time_left()`. `used_subs[sub]` = contacts consumed by pilots in that sub.
- `SimResult(gross, cost, contacts, net, per_campaign: list[dict], within_limits)`; `net = gross − cost`.
  `per_campaign` item keys: `name, n_segment, n_contacted, cost, gross, capped_campaign, capped_reach, capped_money`.
- `ReviewOutcome(veto: list[str] campaign names, rationale, summary, applied, source ∈ {'llm','cache','fallback','off'})`.
- `RunLog(events, pilots, llm)`; `log.log(kind, **fields)` appends and returns `{"kind", "t" (s since start), **fields}`.
  `pilots` / `llm` are appended to directly by the orchestrator / LLM layer (free-form dicts, JSON-able).
- Helpers: `norm_cdf(x)` (erf; NaN → 0.5), `campaign_dict(**kw)` (all CAMPAIGN_KEYS in order, missing = None, unknown key →
  KeyError), `canonical_json(obj)` (sorted keys, compact separators, floats rounded to 4 and −0→0, NaN/inf → null,
  tuples/ndarrays → lists, tuple dict keys → `"a|b|c"`, dataclasses → dicts, sets sorted), `sha256_text(s)` (hex),
  `arm_id(arm) -> "from|seg|to"`, `parse_arm_id(s) -> ArmKey` (ValueError if not 3 parts).

## m10_dataview.py

```python
class DataView:
    def __init__(self, profile: pd.DataFrame, tariffs: pd.DataFrame, channels: dict): ...
    cells: dict[CellKey, Cell]; subs: dict[SubKey, SubCell]; n_nan: int
    tariff_codes: list[str]; tariff_price: dict[str, float]; channels: list[str]
    def tariff_info(self, code: str) -> dict            # all dict_tariff columns of that row
    def cost(self, ch: str) -> float; def mult(self, ch: str) -> float
    def cell_of(self, sub: SubKey) -> CellKey            # sub[:2]
    def subs_of(self, cell: CellKey) -> list[SubCell]    # sorted by key
    def targets_for(self, cell: CellKey) -> list[str]    # tariff_codes minus current tariff
def load_history(search_dirs: list[str]) -> pd.DataFrame | None        # <dir>/data/change_tariff.csv or <dir>/change_tariff.csv; never raises
def load_tariff_descriptions(search_dirs: list[str]) -> dict[str, str] # tariff_dictionary.csv; {} on failure
```
- `n_nan` = rows with NaN in any of current_tariff/arpu_segment/data_segment/call_segment (excluded from subs).
- `channels` = `CHANNELS_ORDER` filtered to `env.channels`, then any extra env channels sorted.
- `tariff_codes` sorted naturally (`tariff_2` < `tariff_10`).

## m20_prior.py
`PriorBuilder(cfg).build(history: pd.DataFrame | None, dv: DataView) -> dict[ArmKey, Prior]` — one Prior for every
cell in `dv.cells` × every target in `dv.targets_for(cell)`. Never raises (no history → price-sign weak prior, sd≈0.25).

## m30_arm_model.py
```python
class ArmModel:
    def __init__(self, cfg: Config, dv: DataView, priors: dict[ArmKey, Prior]): ...
    def add(self, obs: Observation) -> None
    def posterior(self, arm: ArmKey, channel: str) -> Posterior
    def posterior_after(self, arm, pilot_channel, y, n, eval_channel) -> Posterior   # hypothetical, no mutation
    def option(self, sub: SubCell, target: str, channel: str) -> Option
    def n_obs(self, arm: ArmKey) -> int
    observations: list[Observation]   # property / attribute, in add() order
```
- Prior on channel c: `mean = mu·k(c)`, where `k(c) = min(mult_c·ĉ, 1)/ĉ` (ĉ = prior share) scales the base ratio.
- Observation on channel p transfers to channel c with `k(c,p) = min(mult_c·ĉ,1)/min(mult_p·ĉ,1)`; if `k>1` it is
  multiplied by `kappa_up` (upscale discount); transferred noise sd = `noise_sd/√n · k`.
- `option`: `net = post.mean·sum_p − cost·n`, `net_sd = post.sd·sum_p`, `net_lcb = net − z_risk·net_sd`,
  `p_pos = norm_cdf(net/net_sd)` (net_sd=0 → 1.0 if net>0 else 0.0). Unknown arm → prior-less Posterior(0, 0.25).

## m40_simulator.py
```python
class ScoreSimulator:
    def __init__(self, dv: DataView, profile: pd.DataFrame): ...
    def simulate(self, campaigns: list[dict], ratio_fn: Callable[[str, str, str, str], float],
                 budget: float, contacts: int) -> SimResult
    def segment(self, campaign: dict) -> pd.DataFrame
```
- `ratio_fn(current_tariff, arpu_segment, target, channel)` → per-customer lift ratio (multiplier included);
  customer lift = ratio · predicted_arpu. Exact replica of `scoring_core.score_campaigns`: filters (`pd.notna`, `;` lists,
  stripped), sort by ID_NUMBER, cap 5000 → reach cap (remaining contacts) → money cap (push is not money-capped),
  cost/contacts accrue for every contacted customer, gross = Σ over customers of max lift across campaigns
  (dedup by max), `net = gross − cost`. `within_limits` = no campaign was reach/money-capped and ≤ 10 campaigns.
  `budget/contacts` = starting remaining limits (pass `env.remaining_*`).
- `segment(campaign)` → rows after filters, sorted by ID_NUMBER, before caps.

## m50_allocator.py
```python
class Allocator:
    def __init__(self, cfg: Config, dv: DataView, model: ArmModel): ...
    def options_for(self, sub: SubCell) -> list[Option]
    def allocate(self, budget: float, contacts: int, exclude_subs: set[SubKey] = frozenset(),
                 extra_cost: dict[SubKey, float] | None = None) -> Plan
    def shadow_prices(self, budget: float, contacts: int) -> tuple[float, float]   # (lambda_money, lambda_reach)
```
- `options_for`: top `top_targets_per_cell` targets (by posterior mean at mult 1) × `dv.channels`; keep options with
  `net_lcb > 0` and `p_pos ≥ p_min` (push: `p_pos ≥ p_min_push`). Sorted by `net_lcb` desc.
- `allocate`: Lagrangian over (money, reach), nested bisection; each sub picks `argmax(net_lcb − lm·cost − lr·n −
  extra_cost[sub])` or nothing; then feasibility repair (drop worst ratio until within limits). Deterministic.

## m60_packer.py
```python
class CampaignPacker:
    def __init__(self, cfg: Config, dv: DataView, sim: ScoreSimulator): ...
    def pack(self, plan: Plan, model: ArmModel, budget: float, contacts: int) -> list[dict]
    def empty_campaign(self) -> dict
```
- ≤ 10 disjoint campaigns (dicts via `campaign_dict`), always `filter_current_tariff` + `filter_arpu_segment` set,
  `;`-joined tariff lists (target ∉ from-list), coarsening (drop call split → data split → drop min-net campaign) when > 10,
  split > 5000, order by net per contact desc, deterministic names (`c01_<target>_<channel>_<arpu>…`), re-simulated.
- `empty_campaign()` → valid tariff/channel (`push`) whose filters match 0 customers.

## m70_planner.py
```python
class PilotPlanner:
    def __init__(self, cfg: Config, dv: DataView, model: ArmModel, allocator: Allocator): ...
    def set_weights(self, w: dict[ArmKey, float]) -> None        # LLM plausibility weights, default 1.0
    def next_pilot(self, state: ExploreState) -> PilotSpec | None # None = stop exploring
    def register_result(self, spec: PilotSpec, result: dict) -> None
```
- KG with Gauss–Hermite (`gh_nodes`), immediate value, shadow prices, stop rule, reserve limits
  (`explore_money_frac`, `explore_reach_frac`), winner's-curse confirm bonus, `max_pilots_per_arm`.
- `register_result`: bookkeeping only (used_subs, pilots_per_arm, failures). `result` is the `run_pilot` dict, or
  `{"error": str}` on failure (arm/sub then marked unavailable). The orchestrator is the one that calls `model.add`.

## m80_llm.py
```python
class LLMLayer:
    def __init__(self, cfg: Config, mode: str, cache_path: str, log: RunLog): ...
    async def assess_hypotheses(self, batches: dict[str, list[dict]], tariffs: dict[str, dict]) -> dict[ArmKey, float]
    async def review_plan(self, campaigns: list[dict], evidence: list[dict],
                          what_if: Callable[[list[str]], dict]) -> ReviewOutcome
```
- `batches` keyed by arpu segment; each item has at least `arm_id` (see `arm_id()`), prior mu/sd, cell n/sum_p.
  Weights ∈ {0.8, 1.0, 1.2}; neutral 1.0 for every arm on off/error/timeout. In `advise` mode results are logged but
  neutral weights are returned; `review_plan` returns `applied=False`.
- Pydantic AI Agents built lazily (import inside methods; ImportError → behaves as `off`),
  `OpenAIResponsesModel` + `OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL"))`,
  temperature 0 + timeout, `UsageLimits`, `output_validator` + `ModelRetry`, read-only tools, JSON cache keyed by
  `sha256_text(schema_version + agent + model + canonical_json(input))`, `asyncio.wait_for` per call and total `llm_budget_s`.
- `what_if(veto_names) -> {"net": float, "gross": float, "cost": float, "contacts": int}` (simulator on posterior means).

## m85_guardrails.py
```python
class Guardrails:
    def __init__(self, cfg: Config, dv: DataView, sim: ScoreSimulator): ...
    def validate(self, campaigns: list[dict], budget: float, contacts: int) -> list[dict]
class Reporter:
    def __init__(self, cfg: Config, path: str): ...
    def write(self, log: RunLog, campaigns: list[dict], extra: dict) -> None   # markdown; swallows OSError
```
- `validate`: sanitize (only CAMPAIGN_KEYS, strings stripped, `NaN`→None), drop invalid (unknown target/channel/segment,
  target in from-list), dedupe names, ≤ 10, ≥ 1 (empty-campaign fallback), limits check via simulator. Never raises.

## m90_orchestrator.py
```python
class Orchestrator:
    def __init__(self, cfg: Config): ...
    async def run(self, env) -> list[dict]      # research / hypotheses / explore / allocate / review / finalize
def run_coro(coro)                              # asyncio.run, or a dedicated thread with its own loop if a loop is running
class Agent:
    def act(self, env) -> list[dict]            # never raises; guarded fallback
```
- Stage deadlines via `asyncio.timeout`; `env.run_pilot` strictly sequential from the event loop; CPU stages via
  `asyncio.to_thread`. On timeout the pipeline proceeds to allocate with what it knows.

## Integration extensions (optional, backward compatible)

- `CampaignPacker.pack_coarse(model, budget, contacts, z=None) -> list[dict]` — cell-level greedy packing:
  campaigns `(target, channel, arpu, from-tariff list)` with no data/call filter, ratios `mean − z·sd`
  (default `z = cfg.z_risk`), shadow-price grid sweep, then simulated-margin pruning. Sets `last_sim` / `last_report`.
- `ArmModel.calibrate(min_arms=3, beta_prior_sd=0.5, max_tau=0.25) -> dict` — empirical-Bayes check of the history
  prior against pilots (first observation per arm): prior mean ×β (β ~ N(1, 0.5²), clipped [0, 2]), prior variance
  +τ² (moments, capped). `prior_on`/`posterior` use it; `model.calibration` holds the last fit.
- Orchestrator: `allocate` also tries variant `"coarse"` (`pack_coarse` with `z = 0.5·z_risk`; plan rebuilt from the
  campaigns for evidence; veto = plain drop). Exploration = KG planner phase with `_ORC_VERIFY_PILOTS = 8` pilots
  reserved, then verification pilots (`_verify_spec`): the deployed arm of the current coarse plan with the largest
  expected downside `sum_p·E[max(0, −r)]`, piloted if that exceeds pilot money + contact opportunity cost.
  `model.calibrate()` runs after every pilot result.
