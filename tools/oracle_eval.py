#!/usr/bin/env python3
"""Full-information benchmark for the tariff-campaign environment.

The oracle is deliberately stronger than a normal agent: it sees the impact
model, chooses the best target/channel for every targetable cell/subcell, and
then packs the resulting campaign groups under the official limits.  Its final
number is always recomputed by :func:`scoring_core.score_campaigns`.

It is a strong *heuristic* benchmark (a fixed shadow-price sweep plus greedy
packing), not a proven optimum, so an agent's regret against it can be
slightly negative.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mock_environment import _mock_fallback, _mock_impact_model
from scoring_core import (
    CHANNELS,
    MAX_CAMPAIGNS,
    MAX_CUSTOMERS_PER_CAMPAIGN,
    MAX_TOTAL_CONTACTS,
    TOTAL_BUDGET,
    apply_filters,
    score_campaign,
    score_campaigns,
)

Fallback = Callable[[str, str, str, pd.DataFrame, float], tuple[float, float]]


FINE_GROUP = ("target_tariff", "channel", "arpu_segment", "data_segment", "call_segment")
COARSE_GROUP = ("target_tariff", "channel", "arpu_segment")
SUBCELL_KEYS = ("current_tariff", "arpu_segment", "data_segment", "call_segment")
CELL_KEYS = ("current_tariff", "arpu_segment")
MONEY_LAMBDAS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@dataclass(frozen=True)
class _Candidate:
    campaign: dict
    net: float
    cost: float
    contacts: int


def _arm_lookup(impact: pd.DataFrame) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Return (from, segment, to) -> (change, conversion)."""
    return {
        (str(r.tariff_plan_code_from), str(r.arpu_segment), str(r.tariff_plan_code_to)):
        (float(r.arpu_change_pct), float(r.conversion_rate))
        for r in impact.itertuples(index=False)
    }


def _best_subcell_actions(
    impact: pd.DataFrame,
    fallback: Fallback,
    profile: pd.DataFrame,
    dict_tariff: pd.DataFrame,
    money_lambda: float = 0.0,
    key_cols: tuple[str, ...] = SUBCELL_KEYS,
) -> pd.DataFrame:
    """Choose the best arm per ``key_cols`` group by ``net - money_lambda * cost``.

    ``key_cols`` is either :data:`SUBCELL_KEYS` (exact data/call filters) or
    :data:`CELL_KEYS` (whole tariff x ARPU cell, including rows whose
    data/call segment is missing, exactly what a coarse campaign contacts).
    """
    arm = _arm_lookup(impact)
    fallback_conversion = float(impact["conversion_rate"].median())
    targets = sorted(dict_tariff["tariff_plan_code"].dropna().astype(str).unique())
    rows: list[dict] = []
    key_cols = list(key_cols)

    # Rows without predicted_arpu are still contacted (cost) but add no lift.
    clean = profile.dropna(subset=key_cols)
    for key, segment in clean.groupby(key_cols, observed=True, sort=True):
        values = dict(zip(key_cols, map(str, key)))
        current, arpu = values["current_tariff"], values["arpu_segment"]
        n = int(len(segment))
        sum_arpu = float(segment["predicted_arpu"].sum(skipna=True))
        best: dict | None = None
        for target in targets:
            if target == current:
                continue  # a "switch" to the current tariff is not an action
            known = arm.get((current, arpu, target))
            # Mirror scoring_core.score_campaign: a NaN change counts as missing
            # (fallback); a NaN conversion yields a NaN lift that becomes 0.
            if known is None or not np.isfinite(known[0]):
                change, conversion = fallback(
                    current, target, arpu, dict_tariff, fallback_conversion
                )
            else:
                change, conversion = known
            if not np.isfinite(change) or not np.isfinite(conversion):
                continue
            for channel, channel_info in CHANNELS.items():
                effective_conversion = min(float(conversion) * channel_info["conversion_multiplier"], 1.0)
                gross = float(change) * effective_conversion * sum_arpu
                cost = n * float(channel_info["cost_per_contact"])
                net = gross - cost
                if net <= 0:
                    continue
                reduced = net - money_lambda * cost
                if best is None or reduced > best["reduced"]:
                    best = {
                        **values,
                        "target_tariff": target,
                        "channel": channel,
                        "net": net,
                        "reduced": reduced,
                    }
        if best is not None and best["net"] > 0:
            rows.append(best)
    return pd.DataFrame(rows)


def _campaign_candidates(
    actions: pd.DataFrame,
    impact: pd.DataFrame,
    fallback: Fallback,
    profile: pd.DataFrame,
    dict_tariff: pd.DataFrame,
    group_cols: tuple[str, ...] = COARSE_GROUP,
    split: bool = True,
) -> list[_Candidate]:
    """Consolidate chosen subcells into campaigns with ';'-joined tariff lists.

    Effects depend only on (tariff, ARPU), so the coarse grouping
    (target, channel, ARPU) broadens each tariff/ARPU cell and covers more
    subcells per campaign slot; the fine grouping keeps exact data/call
    filters.  The official scorer values each campaign exactly either way.
    With ``split`` a group larger than the per-campaign cap is split by tariff
    into several campaigns; without it the scorer truncates it by ``ID_NUMBER``.
    """
    if actions.empty:
        return []
    fallback_conversion = float(impact["conversion_rate"].median())
    candidates: list[_Candidate] = []
    for key, group in actions.groupby(list(group_cols), observed=True, sort=True):
        values = dict(zip(group_cols, map(str, key)))
        target, channel, arpu = values["target_tariff"], values["channel"], values["arpu_segment"]
        data, call = values.get("data_segment"), values.get("call_segment")
        tariffs = sorted(group["current_tariff"].astype(str).unique())
        name_parts = [target, channel, arpu] + [v for v in (data, call) if v is not None]
        base = {
            "filter_arpu_segment": arpu,
            "filter_data_segment": data,
            "filter_call_segment": call,
            "target_tariff": target,
            "channel": channel,
        }
        chunks = _split_tariffs(profile, base, tariffs) if split else [tariffs]
        for index, chunk in enumerate(chunks):
            suffix = f"_part{index + 1}" if len(chunks) > 1 else ""
            campaign = {
                "campaign_name": "oracle_" + "_".join(name_parts) + suffix,
                **base,
                "filter_current_tariff": ";".join(chunk),
            }
            selected = apply_filters(profile, pd.Series(campaign)).sort_values("ID_NUMBER")
            selected = selected.iloc[:MAX_CUSTOMERS_PER_CAMPAIGN]
            scored = score_campaign(
                selected,
                target,
                impact,
                dict_tariff,
                fallback_conversion,
                channel,
                fallback,
            )
            contacts = int(len(selected))
            cost = contacts * float(CHANNELS[channel]["cost_per_contact"])
            net = float(scored["expected_lift_per_customer"].sum()) - cost
            if contacts and net > 0:
                candidates.append(_Candidate(campaign, net, cost, contacts))
    return candidates


def _split_tariffs(profile: pd.DataFrame, base: dict, tariffs: list[str]) -> list[list[str]]:
    """Split a tariff list into consecutive chunks of at most 5000 contacts each.

    A single tariff that alone exceeds the cap stays in its own chunk (the
    scorer then truncates it by ``ID_NUMBER``, which is unavoidable).
    """
    segment = apply_filters(profile, pd.Series({**base, "filter_current_tariff": ";".join(tariffs)}))
    counts = segment["current_tariff"].astype(str).value_counts()
    chunks: list[list[str]] = []
    current: list[str] = []
    size = 0
    for tariff in tariffs:
        n = int(counts.get(tariff, 0))
        if current and size + n > MAX_CUSTOMERS_PER_CAMPAIGN:
            chunks.append(current)
            current, size = [], 0
        current.append(tariff)
        size += n
    if current:
        chunks.append(current)
    return chunks


def _pack(candidates: list[_Candidate]) -> list[dict]:
    """Greedy/Lagrangian two-resource packing with a ten-campaign cap."""
    if not candidates:
        return []

    positive = [c for c in candidates if c.net > 0]
    money_density = np.array([c.net / c.cost for c in positive if c.cost > 0], dtype=float)
    reach_density = np.array([c.net / c.contacts for c in positive], dtype=float)

    def grid(values: np.ndarray) -> list[float]:
        if not len(values):
            return [0.0]
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, 9))
        return sorted(set([0.0, *map(float, quantiles)]))

    money_lambdas = grid(money_density)
    reach_lambdas = grid(reach_density)
    selections: list[list[_Candidate]] = []

    # Include familiar one-dimensional greedy orderings as well as the
    # Lagrangian sweep.  The fill pass makes every selection locally maximal.
    orderings = [
        sorted(positive, key=lambda c: c.net, reverse=True),
        sorted(positive, key=lambda c: c.net / c.contacts, reverse=True),
        sorted(positive, key=lambda c: (float("inf") if c.cost == 0 else c.net / c.cost), reverse=True),
    ]
    for lm in money_lambdas:
        for lr in reach_lambdas:
            ordered = sorted(
                positive,
                key=lambda c: (c.net - lm * c.cost - lr * c.contacts, c.net),
                reverse=True,
            )
            ordered = [c for c in ordered if c.net - lm * c.cost - lr * c.contacts > 0]
            orderings.append(ordered)

    for ordered in orderings:
        chosen: list[_Candidate] = []
        used_money = 0.0
        used_reach = 0
        for candidate in ordered:
            if len(chosen) >= MAX_CAMPAIGNS:
                break
            if used_money + candidate.cost > TOTAL_BUDGET:
                continue
            if used_reach + candidate.contacts > MAX_TOTAL_CONTACTS:
                continue
            chosen.append(candidate)
            used_money += candidate.cost
            used_reach += candidate.contacts
        # A reduced-cost ordering can leave room. Fill it using actual value.
        for candidate in sorted(positive, key=lambda c: c.net, reverse=True):
            if candidate in chosen or len(chosen) >= MAX_CAMPAIGNS:
                continue
            if used_money + candidate.cost <= TOTAL_BUDGET and used_reach + candidate.contacts <= MAX_TOTAL_CONTACTS:
                chosen.append(candidate)
                used_money += candidate.cost
                used_reach += candidate.contacts
        selections.append(chosen)

    best = max(selections, key=lambda choice: sum(c.net for c in choice), default=[])
    # Highest-value campaigns first makes truncation behavior deterministic.
    return [c.campaign for c in sorted(best, key=lambda c: c.net, reverse=True)]


def oracle_plan(
    world_impact: pd.DataFrame,
    fallback: Fallback,
    profile: pd.DataFrame,
    dict_tariff: pd.DataFrame,
) -> tuple[list[dict], dict]:
    """Build and score a reproducible full-information campaign plan.

    Sweeps a money shadow price (channel choice per cell/subcell) and four
    grouping/splitting variants; every plan is scored exactly by ``score_campaigns`` and the
    best one is returned.
    """
    columns = [
        "campaign_name", "filter_arpu_segment", "filter_data_segment",
        "filter_call_segment", "filter_current_tariff", "target_tariff", "channel",
    ]
    baseline = float(profile["predicted_arpu"].sum())
    # Doing nothing is always feasible, so the benchmark never scores below it.
    empty = score_campaigns(
        pd.DataFrame([], columns=columns), profile, world_impact, dict_tariff,
        baseline, fallback, team_id="oracle",
    )
    best: tuple[list[dict], dict] = ([], empty)
    # Fine campaigns come from subcell actions.  Coarse campaigns are built two
    # ways: from whole-cell actions (one action per tariff x ARPU cell, so the
    # candidates are disjoint and the packer's additive value is exact) and
    # from subcell actions (may overlap; dedup is handled by exact scoring).
    # Oversized groups are tried both split into <=5000 parts and truncated.
    groupings = (
        (FINE_GROUP, SUBCELL_KEYS, True),
        (COARSE_GROUP, CELL_KEYS, True),
        (COARSE_GROUP, SUBCELL_KEYS, True),
        (COARSE_GROUP, SUBCELL_KEYS, False),
    )
    for money_lambda in MONEY_LAMBDAS:
        actions_by_keys: dict[tuple[str, ...], pd.DataFrame] = {}
        for group_cols, key_cols, split in groupings:
            if key_cols not in actions_by_keys:
                actions_by_keys[key_cols] = _best_subcell_actions(
                    world_impact, fallback, profile, dict_tariff, money_lambda, key_cols
                )
            candidates = _campaign_candidates(
                actions_by_keys[key_cols], world_impact, fallback, profile, dict_tariff,
                group_cols, split,
            )
            campaigns = _pack(candidates)
            strategy = pd.DataFrame(campaigns, columns=columns)
            result = score_campaigns(
                strategy, profile, world_impact, dict_tariff, baseline, fallback,
                team_id="oracle",
            )
            if result["net_arpu_gain"] > best[1]["net_arpu_gain"]:
                best = (campaigns, result)
    return best


def oracle_net(
    world_impact: pd.DataFrame,
    fallback: Fallback,
    profile: pd.DataFrame,
    dict_tariff: pd.DataFrame,
) -> float:
    """Return the score of the strong full-information benchmark plan."""
    _, result = oracle_plan(world_impact, fallback, profile, dict_tariff)
    return float(result["net_arpu_gain"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "customer_profile.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--show-plan", action="store_true", help="print the selected campaigns")
    args = parser.parse_args(argv)

    profile = pd.read_csv(args.profile)
    tariffs = pd.read_csv(args.data_dir / "dict_tariff.csv")
    impact = _mock_impact_model(pd.read_csv(args.data_dir / "change_tariff.csv"))
    campaigns, result = oracle_plan(impact, _mock_fallback, profile, tariffs)
    print(f"Oracle mock net: {result['net_arpu_gain']:,.2f}")
    print(
        f"Campaigns: {len(campaigns)} | contacts: {result['total_contacts']:,} | "
        f"cost: {result['total_cost']:,.2f} | gross: {result['gross_arpu_lift']:,.2f}"
    )
    if args.show_plan:
        for campaign in campaigns:
            print(campaign)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
