#!/usr/bin/env python3
"""Evaluate an agent against deterministic alternative impact-model worlds."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import make_environment
from tools.eval_checks import campaign_errors
from mock_environment import (
    CHANNELS,
    MAX_TOTAL_CONTACTS,
    TOTAL_BUDGET,
    _mock_fallback,
    _mock_impact_model,
)
from scoring_core import MAX_CAMPAIGNS, sanitize_campaigns, score_campaigns
from tools.oracle_eval import oracle_net

Fallback = Callable[[str, str, str, pd.DataFrame, float], tuple[float, float]]
FILTER_COLUMNS = [
    "filter_arpu_segment", "filter_data_segment", "filter_call_segment",
    "filter_current_tariff", "explicit_ids",
]
WORLD_SEED = 20260923


@dataclass(frozen=True)
class World:
    name: str
    impact: pd.DataFrame
    fallback: Fallback
    description: str


def _keep_self_zero(fallback: Fallback) -> Fallback:
    """Keep a "switch" to the current tariff a no-op (mock gives it 0 change)."""
    def guarded(current: str, target: str, segment: str, tariffs: pd.DataFrame,
                fallback_conversion: float) -> tuple[float, float]:
        if str(current) == str(target):
            return _mock_fallback(current, target, segment, tariffs, fallback_conversion)
        return fallback(current, target, segment, tariffs, fallback_conversion)
    return guarded


def _wrap_fallback(
    transform: Callable[[float, float, str, str, str], tuple[float, float]],
) -> Fallback:
    """Apply ``transform`` to the mock fallback, exactly once, off the diagonal.

    Note: the scorer passes the *world* impact median as ``fallback_conversion``,
    so a world that rescales ``conversion_rate`` in its impact table already
    rescales the fallback conversion; the transform must not rescale it again.
    """
    def wrapped(current: str, target: str, segment: str, tariffs: pd.DataFrame,
                fallback_conversion: float) -> tuple[float, float]:
        change, conversion = _mock_fallback(current, target, segment, tariffs, fallback_conversion)
        return transform(float(change), float(conversion), current, target, segment)
    return _keep_self_zero(wrapped)


def _stable_uniform(*parts: str) -> float:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    integer = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return integer / float(2**64)


def build_worlds(base: pd.DataFrame, tariffs: pd.DataFrame) -> dict[str, World]:
    """Create all named worlds from a mock impact model."""
    rng = np.random.default_rng(WORLD_SEED)
    worlds: dict[str, World] = {}

    def add(name: str, impact: pd.DataFrame, fallback: Fallback, description: str) -> None:
        ordered = impact[list(base.columns)].reset_index(drop=True).copy()
        worlds[name] = World(name, ordered, fallback, description)

    add("mock", base.copy(), _mock_fallback, "historical mock")

    for factor, name in [(2.0, "scale_x2"), (0.5, "scale_x0.5")]:
        impact = base.copy()
        impact["arpu_change_pct"] *= factor
        fallback = _wrap_fallback(lambda c, v, _f, _t, _s, k=factor: (c * k, v))
        add(name, impact, fallback, f"ARPU effects x{factor:g}")

    impact = base.copy()
    impact.loc[impact["arpu_segment"].astype(str) == "HIGH", "arpu_change_pct"] += 0.15
    fallback = _wrap_fallback(lambda c, v, _f, _t, s: (c + (0.15 if s == "HIGH" else 0.0), v))
    add("high_plus_0.15", impact, fallback, "+0.15 change on HIGH arms")

    # One stable permutation per (source, segment) over all non-self targets:
    # world(f, T, s) = mock(f, perm[T], s), for materialized rows and fallback
    # alike, so the known/missing-arm boundary is permuted consistently too.
    codes = sorted(
        set(tariffs["tariff_plan_code"].dropna().astype(str))
        | set(base["tariff_plan_code_from"].astype(str))
        | set(base["tariff_plan_code_to"].astype(str))
    )
    target_permutations: dict[tuple[str, str], dict[str, str]] = {}
    for current in codes:
        others = [code for code in codes if code != current]
        for segment in ("LOW", "MID", "HIGH"):
            local_rng = np.random.default_rng(
                int(_stable_uniform("permute", current, segment) * (2**32 - 1))
            )
            shuffled = [others[i] for i in local_rng.permutation(len(others))]
            mapping = dict(zip(others, shuffled))
            mapping[current] = current
            target_permutations[(current, segment)] = mapping
    inverse = {
        key: {source: target for target, source in mapping.items()}
        for key, mapping in target_permutations.items()
    }
    impact = base.copy()
    impact["tariff_plan_code_to"] = [
        inverse.get((str(f), str(s)), {}).get(str(t), str(t))
        for f, t, s in zip(base["tariff_plan_code_from"], base["tariff_plan_code_to"],
                           base["arpu_segment"])
    ]

    def permuted_fallback(current: str, target: str, segment: str,
                          dict_tariff: pd.DataFrame, fallback_conversion: float) -> tuple[float, float]:
        permuted_target = target_permutations.get((str(current), str(segment)), {}).get(target, target)
        return _mock_fallback(current, permuted_target, segment, dict_tariff, fallback_conversion)

    add(
        "permute_targets", impact, permuted_fallback,
        "target ranking permuted within each source cell",
    )

    impact = base.copy()
    impact["conversion_rate"] *= 3.0
    # The fallback conversion is the world impact median, already tripled above.
    add("saturation", impact, _mock_fallback, "conversion x3; scoring clips at probability 1")

    impact = base.copy()
    flip = rng.random(len(impact)) < 0.30
    impact.loc[flip, "arpu_change_pct"] *= -1.0
    fallback = _wrap_fallback(
        lambda c, v, f, t, s: (-c if _stable_uniform("flip", f, t, s) < 0.30 else c, v)
    )
    add("flip_sign_30", impact, fallback, "sign flipped on a deterministic random 30% of arms")

    impact = base.copy()
    changes = base["arpu_change_pct"].to_numpy(copy=True)
    conversions = base["conversion_rate"].to_numpy(copy=True)
    impact["arpu_change_pct"] = changes[rng.permutation(len(changes))]
    impact["conversion_rate"] = conversions[rng.permutation(len(conversions))]

    def random_marginal_fallback(current: str, target: str, segment: str,
                                 _tariffs: pd.DataFrame, _fallback_conversion: float) -> tuple[float, float]:
        u1 = _stable_uniform("prior-change", current, target, segment)
        u2 = _stable_uniform("prior-conv", current, target, segment)
        return (
            float(np.quantile(changes, u1)),
            float(np.quantile(conversions, u2)),
        )

    add(
        "prior_misspecified", impact, _keep_self_zero(random_marginal_fallback),
        "effects randomized independently of history with matching marginals",
    )

    impact = base.copy()
    price = tariffs.set_index("tariff_plan_code")["price_tariff"]
    deltas = np.array([
        float(price.get(str(t), np.nan) - price.get(str(f), np.nan))
        for f, t in zip(impact["tariff_plan_code_from"], impact["tariff_plan_code_to"])
    ])
    effects = impact["arpu_change_pct"].to_numpy(dtype=float)
    valid = np.isfinite(deltas) & np.isfinite(effects)
    if valid.sum() >= 2 and np.std(deltas[valid]) > 0:
        slope = float(np.cov(deltas[valid], effects[valid], ddof=0)[0, 1] / np.var(deltas[valid]))
        residual = effects.copy()
        residual[valid] = effects[valid] - slope * (deltas[valid] - deltas[valid].mean())
        if np.std(residual[valid]) > 0:
            residual[valid] = ((residual[valid] - residual[valid].mean())
                               * (np.std(effects[valid]) / np.std(residual[valid]))
                               + effects[valid].mean())
        impact["arpu_change_pct"] = residual
    segment_medians = base.groupby("arpu_segment", observed=True)["arpu_change_pct"].median().to_dict()
    global_median = float(base["arpu_change_pct"].median())

    def decorrelated_fallback(current: str, target: str, segment: str,
                              _tariffs: pd.DataFrame, fallback_conversion: float) -> tuple[float, float]:
        # Stable source/segment jitter, deliberately independent of target price.
        center = float(segment_medians.get(segment, global_median))
        jitter = (_stable_uniform("decorrelated", current, segment) - 0.5) * float(np.std(effects))
        return center + jitter, fallback_conversion

    add("price_decorrelated", impact, _keep_self_zero(decorrelated_fallback),
        "linear target-price signal removed")
    return worlds


ALIASES = {
    "x2": "scale_x2", "2x": "scale_x2", "scale2": "scale_x2",
    "scale-x2": "scale_x2",
    "x0.5": "scale_x0.5", "0.5x": "scale_x0.5", "half": "scale_x0.5",
    "scale-x0.5": "scale_x0.5",
    "high": "high_plus_0.15", "high_plus": "high_plus_0.15",
    "high_plus_015": "high_plus_0.15", "high+0.15": "high_plus_0.15",
    "permute": "permute_targets", "permuted": "permute_targets",
    "permute_target_ranking": "permute_targets",
    "flip": "flip_sign_30", "flip_sign": "flip_sign_30",
    "flip_sign_30%": "flip_sign_30", "flip-sign-30": "flip_sign_30",
    "prior": "prior_misspecified", "misspecified": "prior_misspecified",
    "prior-misspecified": "prior_misspecified",
    "price": "price_decorrelated", "decorrelated": "price_decorrelated",
    "price-decorrelated": "price_decorrelated",
}


def _select_worlds(spec: str, worlds: dict[str, World]) -> list[World]:
    if spec.strip().lower() == "all":
        return list(worlds.values())
    selected: list[World] = []
    unknown: list[str] = []
    for raw in spec.split(","):
        requested = raw.strip()
        if not requested:
            continue
        canonical = ALIASES.get(requested.lower(), requested.lower())
        if canonical not in worlds:
            unknown.append(requested)
        elif all(world.name != canonical for world in selected):
            selected.append(worlds[canonical])
    if unknown:
        raise ValueError(f"unknown worlds: {', '.join(unknown)}; choices: {', '.join(worlds)}")
    if not selected:
        raise ValueError("--worlds selected no worlds")
    return selected


def _load_agent(module_name: str):
    importlib.invalidate_caches()
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"cannot import agent module {module_name!r}: {type(exc).__name__}: {exc}") from exc
    agent_class = getattr(module, "Agent", None)
    if agent_class is None or not callable(agent_class):
        raise RuntimeError(f"agent module {module_name!r} has no callable Agent class")
    return agent_class


def _evaluate_once(agent_class, world: World, seed: int, profile: pd.DataFrame,
                   tariffs: pd.DataFrame) -> dict:
    env, internals = make_environment(
        customer_profile=profile,
        impact_model=world.impact,
        dict_tariff=tariffs,
        channels=CHANNELS,
        total_budget=TOTAL_BUDGET,
        max_total_contacts=MAX_TOTAL_CONTACTS,
        fallback_predict=world.fallback,
        seed=seed,
    )
    error: str | None = None
    try:
        agent = agent_class()
        final_campaigns = agent.act(env)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"[!] world={world.name} seed={seed}: agent failed: {error}", file=sys.stderr)
        final_campaigns = []

    try:
        invalid = campaign_errors(final_campaigns, tariffs)
        if invalid:
            raise ValueError("; ".join(invalid))
        finals = sanitize_campaigns(final_campaigns, tariffs)[:MAX_CAMPAIGNS]
    except Exception as exc:
        validation_error = f"invalid agent result ({type(exc).__name__}: {exc})"
        error = f"{error}; {validation_error}" if error else validation_error
        print(f"[!] world={world.name} seed={seed}: {validation_error}", file=sys.stderr)
        finals = []
    pilots = internals.executed_pilot_campaigns()
    if not 1 <= len(pilots) <= 20:
        issue = f"expected 1..20 pilots, got {len(pilots)}"
        error = f"{error}; {issue}" if error else issue
    if env.remaining_budget < 0 or env.remaining_contacts < 0 or env.pilots_left < 0:
        issue = "pilot resource limits exceeded"
        error = f"{error}; {issue}" if error else issue
    campaigns = pd.DataFrame(pilots + finals)
    required = ["target_tariff", "channel", *FILTER_COLUMNS]
    for column in required:
        if column not in campaigns.columns:
            campaigns[column] = None
    result = score_campaigns(
        campaigns,
        profile,
        world.impact,
        tariffs,
        float(profile["predicted_arpu"].sum()),
        world.fallback,
        team_id=f"stress:{world.name}:{seed}",
    )
    return {
        "seed": seed,
        "net": float(result["net_arpu_gain"]),
        "gross": float(result["gross_arpu_lift"]),
        "cost": float(result["total_cost"]),
        "contacts": int(result["total_contacts"]),
        "campaigns": int(result["n_campaigns"]),
        "final_campaigns": len(finals),
        "pilots": len(pilots),
        "error": error,
    }


def _summary(name: str, runs: list[dict], oracle: float) -> dict:
    values = pd.Series([r["net"] for r in runs], dtype=float)
    q25, q75 = map(float, values.quantile([0.25, 0.75]))
    median = float(values.median())
    return {
        "world": name,
        "runs": len(runs),
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "min": float(values.min()),
        "max": float(values.max()),
        "share_positive": float((values > 0).mean()),
        "oracle_net": float(oracle),
        "regret": float(oracle - median),
    }


def _json_safe(value):
    """Recursively replace NaN/inf floats with ``None`` for strict JSON."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _print_table(summaries: list[dict]) -> None:
    header = (
        f"{'world':<22} {'median':>13} {'q25':>13} {'q75':>13} {'IQR':>13} "
        f"{'min':>13} {'max':>13} {'positive':>9} {'oracle':>13} {'regret':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in summaries:
        print(
            f"{row['world']:<22} {row['median']:>13,.0f} {row['q25']:>13,.0f} "
            f"{row['q75']:>13,.0f} {row['iqr']:>13,.0f} {row['min']:>13,.0f} "
            f"{row['max']:>13,.0f} {row['share_positive']:>8.1%} "
            f"{row['oracle_net']:>13,.0f} {row['regret']:>13,.0f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", default="all", help="comma-separated world names, aliases, or 'all'")
    parser.add_argument("--runs", type=int, default=5, help="number of pilot-noise seeds per world")
    parser.add_argument("--agent-module", default="agent", help="importable module containing Agent")
    parser.add_argument("--json", type=Path, dest="json_path", help="write full machine-readable results")
    parser.add_argument("--llm", action="store_true", help="preserve AGENT_LLM_MODE instead of forcing off")
    parser.add_argument("--no-oracle", action="store_true", help="skip the (~10s/world) oracle benchmark")
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if not args.llm:
        os.environ["AGENT_LLM_MODE"] = "off"

    try:
        agent_class = _load_agent(args.agent_module)
    except RuntimeError as exc:
        print(f"[!] {exc}", file=sys.stderr)
        return 2

    try:
        profile = pd.read_csv(ROOT / "customer_profile.csv")
        tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
        base = _mock_impact_model(pd.read_csv(ROOT / "data" / "change_tariff.csv"))
        worlds = build_worlds(base, tariffs)
        selected = _select_worlds(args.worlds, worlds)
    except (OSError, ValueError) as exc:
        print(f"[!] setup failed: {exc}", file=sys.stderr)
        return 2

    details: dict[str, list[dict]] = {}
    summaries: list[dict] = []
    for world in selected:
        runs = [
            _evaluate_once(agent_class, world, seed, profile, tariffs)
            for seed in range(args.runs)
        ]
        details[world.name] = runs
        oracle = (float("nan") if args.no_oracle
                  else oracle_net(world.impact, world.fallback, profile, tariffs))
        summaries.append(_summary(world.name, runs, oracle))

    _print_table(summaries)
    if args.json_path:
        payload = {
            "agent_module": args.agent_module,
            "llm_forced_off": not args.llm,
            "world_seed": WORLD_SEED,
            "agent_sha256": hashlib.sha256(Path(importlib.import_module(args.agent_module).__file__).read_bytes()).hexdigest(),
            "summaries": summaries,
            "runs": details,
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False)
        args.json_path.write_text(text + "\n", encoding="utf-8")
        print(f"JSON: {args.json_path}")
    failures = sum(run["error"] is not None for runs in details.values() for run in runs)
    if failures:
        print(f"[!] {failures} runs failed execution/requirement checks", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
