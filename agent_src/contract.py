"""Shared contract: types, config and helpers used by every agent module.

Общий контракт: типы, конфигурация и вспомогательные функции агента.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Keys and constants
# ---------------------------------------------------------------------------

CellKey = tuple[str, str]  # (current_tariff, arpu_segment)
SubKey = tuple[str, str, str, str]  # (current_tariff, arpu_segment, data_segment, call_segment)
ArmKey = tuple[str, str, str]  # (current_tariff, arpu_segment, target_tariff)

CHANNELS_ORDER: tuple[str, ...] = ("push", "sms", "digital_ads", "call")
ARPU_SEGMENTS: tuple[str, ...] = ("LOW", "MID", "HIGH")
DATA_SEGMENTS: tuple[str, ...] = ("NON_USER", "LITE", "HEAVY")
CALL_SEGMENTS: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
CAMPAIGN_KEYS: tuple[str, ...] = (
    "campaign_name",
    "filter_arpu_segment",
    "filter_data_segment",
    "filter_call_segment",
    "filter_current_tariff",
    "target_tariff",
    "channel",
)
LLM_MODES: tuple[str, ...] = ("off", "advise", "decide")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Immutable run configuration (limits, risk knobs, LLM settings)."""

    time_budget_s: float = 420.0
    llm_budget_s: float = 90.0
    llm_call_timeout_s: float = 30.0
    noise_sd: float = 0.804
    z_risk: float = 1.0
    p_min: float = 0.8
    p_min_push: float = 0.6
    tau: float = 0.2
    kappa_up: float = 0.9
    conv_prior_default: float = 0.3
    shrink_k: float = 20.0
    explore_money_frac: float = 0.25
    explore_reach_frac: float = 0.30
    max_pilots: int = 20
    pilot_sizes: tuple[int, ...] = (30, 60, 100, 150, 200)
    min_pilot: int = 10
    max_pilot: int = 200
    max_campaigns: int = 10
    max_per_campaign: int = 5000
    top_arms: int = 40
    top_targets_per_cell: int = 3
    max_pilots_per_arm: int = 3
    gh_nodes: int = 5
    seed: int = 42
    schema_version: str = "v1"
    llm_model: str = "gpt-4.1-mini"
    report_path: str = "agent_report.md"
    cache_file: str = "llm_cache.json"

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        """Build config applying env overrides (OPENAI_MODEL) and explicit kwargs.

        AGENT_LLM_MODE is not stored here; it is resolved by resolve_llm_mode().
        """
        kw: dict[str, Any] = {}
        model = os.getenv("OPENAI_MODEL")
        if model and model.strip():
            kw["llm_model"] = model.strip()
        kw.update(overrides)
        return cls(**kw)

    def replace(self, **changes: Any) -> "Config":
        """Return a copy with some fields changed."""
        return dataclasses.replace(self, **changes)


def resolve_llm_mode() -> str:
    """Return 'off'|'advise'|'decide' from AGENT_LLM_MODE.

    Unset/invalid value -> 'off', independent of the machine's credentials.
    An explicit 'advise'/'decide' without a key degrades to 'off'.
    """
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    raw = os.getenv("AGENT_LLM_MODE", "").strip().lower()
    if raw == "off":
        return "off"
    if raw in ("advise", "decide"):
        return raw if has_key else "off"
    return "off"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SubCell:
    """Targeting unit: cell x data_segment x call_segment."""

    key: SubKey
    n: int
    sum_p: float
    mean_p: float
    ids: np.ndarray  # sorted ID_NUMBER
    p: np.ndarray  # predicted_arpu aligned to ids


@dataclass
class Cell:
    """Effect unit: (current_tariff, arpu_segment)."""

    key: CellKey
    n: int
    sum_p: float
    mean_p: float
    subs: list[SubKey]


@dataclass
class Prior:
    """Prior on base lift ratio at channel multiplier 1.0 (= change * share)."""

    mu: float
    sd: float
    share: float  # conversion estimate
    n_hist: int
    source: str  # e.g. 'arm' | 'cell' | 'global' | 'price' | 'none'


@dataclass
class Observation:
    """One pilot result attributed to an arm."""

    arm: ArmKey
    channel: str
    y: float  # observed_lift_ratio
    n: int  # n_customers actually contacted
    cost: float
    sub: Optional[SubKey]
    pilot_index: int


@dataclass
class Posterior:
    """Posterior of lift ratio (fraction of predicted_arpu) for an arm AND channel."""

    mean: float
    sd: float


@dataclass
class Option:
    """Candidate action for one sub-cell."""

    sub: SubKey
    target: str
    channel: str
    n: int
    cost: float
    net_mean: float
    net_sd: float
    net_lcb: float
    p_pos: float


@dataclass
class Plan:
    """Allocator output."""

    options: list[Option]
    lambda_money: float
    lambda_reach: float
    total_net_lcb: float
    total_cost: float
    total_contacts: int


@dataclass
class PilotSpec:
    """Pilot to run. filters = env.run_pilot filter kwargs (None = no filter)."""

    arm: ArmKey
    channel: str
    n: int
    sub: Optional[SubKey]
    filters: dict
    score: float
    reason: str

    def run_kwargs(self) -> dict:
        """Full kwargs for env.run_pilot."""
        kw = {
            "target_tariff": self.arm[2],
            "channel": self.channel,
            "n_customers": int(self.n),
        }
        for k in ("filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff"):
            kw[k] = self.filters.get(k)
        return kw


@dataclass
class ExploreState:
    """Snapshot of exploration resources."""

    remaining_budget: float
    remaining_contacts: int
    pilots_left: int
    explore_money_spent: float
    explore_reach_spent: int
    pilots_per_arm: dict = field(default_factory=dict)  # dict[ArmKey, int]
    used_subs: dict = field(default_factory=dict)  # dict[SubKey, int] contacts used by pilots
    deadline: float = math.inf  # time.monotonic() based

    def time_left(self) -> float:
        """Seconds until deadline (may be negative)."""
        return self.deadline - time.monotonic()


@dataclass
class SimResult:
    """Output of ScoreSimulator.simulate."""

    gross: float
    cost: float
    contacts: int
    net: float
    per_campaign: list[dict]
    within_limits: bool


@dataclass
class ReviewOutcome:
    """Result of the LLM risk review."""

    veto: list[str]
    rationale: str
    summary: str
    applied: bool
    source: str  # 'llm' | 'cache' | 'fallback' | 'off'


@dataclass
class RunLog:
    """Structured in-memory run log (no prints)."""

    events: list[dict] = field(default_factory=list)
    pilots: list[dict] = field(default_factory=list)
    llm: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.monotonic)

    def log(self, kind: str, **fields: Any) -> dict:
        """Append an event {'kind': kind, 't': elapsed_s, **fields}; returns it."""
        ev = {"kind": kind, "t": round(time.monotonic() - self.t0, 3), **fields}
        self.events.append(ev)
        return ev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (handles +-inf)."""
    if x != x:  # NaN
        return 0.5
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def campaign_dict(**kw: Any) -> dict:
    """Campaign dict with all CAMPAIGN_KEYS (missing = None); unknown keys are rejected."""
    unknown = set(kw) - set(CAMPAIGN_KEYS)
    if unknown:
        raise KeyError(f"unknown campaign keys: {sorted(unknown)}")
    return {k: kw.get(k) for k in CAMPAIGN_KEYS}


def _canon(obj: Any) -> Any:
    """Convert to JSON-safe canonical structure (floats rounded to 4 decimals)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            return None
        r = round(f, 4)
        return 0.0 if r == 0 else r
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _canon(v)
                for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_canon(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_canon(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canon({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact, floats rounded to 4, tuples->lists, tuple keys joined by '|'."""
    return json.dumps(_canon(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(s: str) -> str:
    """Hex sha256 of a UTF-8 string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def arm_id(arm: ArmKey) -> str:
    """Stable string id for an arm: 'from|seg|to'."""
    return "|".join(arm)


def parse_arm_id(s: str) -> ArmKey:
    """Inverse of arm_id; raises ValueError on bad input."""
    parts = s.split("|")
    if len(parts) != 3:
        raise ValueError(f"bad arm id {s!r}")
    return (parts[0], parts[1], parts[2])
