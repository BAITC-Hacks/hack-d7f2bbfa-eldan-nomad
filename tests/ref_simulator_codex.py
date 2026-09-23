"""Small, independent oracle for the campaign-scoring mechanics.

The implementation is deliberately row-oriented.  It does not use the
DataFrame filtering, merging, or grouping code from :mod:`scoring_core`, so it
can catch regressions in those operations during differential tests.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

import scoring_core


def _notna(value: Any) -> bool:
    """Scalar equivalent of ``pd.notna`` for campaign/profile values."""
    if value is None:
        return False
    try:
        # Covers float/Decimal NaN, NaT, and ordinary scalar values.  pandas.NA
        # reaches the exception because its truth value is intentionally vague.
        return not bool(value != value)
    except (TypeError, ValueError):
        return False


def _equal(left: Any, right: Any) -> bool:
    """Equality with the false-on-missing behavior of a pandas filter mask."""
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _missing_kind(value: Any) -> str | None:
    """Kind of missing sentinel (``Series.isin`` only matches like with like)."""
    if value is None:
        return "none"
    if _notna(value):
        return None
    if value is pd.NA:
        return "na"
    if value is pd.NaT:
        return "nat"
    return "nan"


def _isin(value: Any, candidates: Sequence[Any]) -> bool:
    """The subset of Series.isin semantics needed for explicit customer IDs."""
    value_missing = _missing_kind(value)
    for candidate in candidates:
        candidate_missing = _missing_kind(candidate)
        if value_missing is not None or candidate_missing is not None:
            # pandas matches None/None, NaN/NaN, NA/NA, NaT/NaT but no cross pairs.
            if value_missing is not None and value_missing == candidate_missing:
                return True
            continue
        if _equal(value, candidate):
            return True
    return False


def ref_simulate(
    profile,
    campaigns,
    ratio_fn: Callable[[str, str, str, str], float],
    budget,
    contacts,
    max_per_campaign: int = 5000,
    channels: Mapping[str, Mapping[str, float]] = scoring_core.CHANNELS,
) -> dict:
    """Evaluate campaigns with the same limiting and deduplication rules.

    ``ratio_fn`` returns the channel-adjusted expected lift ratio.  The oracle
    multiplies it by each customer's ``predicted_arpu`` and returns only the
    four aggregate values used by planning code plus a ``per_campaign`` list
    (segment size, contacts, cost, pre-dedup gross and cap flags).  Missing profile keys are
    passed to ``ratio_fn`` unchanged (NaN, as scoring_core's merge does); the
    agent ScoreSimulator passes ``None`` instead, so ratio functions used in
    differential tests must treat both alike.  Row order for ties in
    ``ID_NUMBER`` is stable, whereas pandas' default quicksort is not, so
    duplicate IDs straddling a cap boundary are outside the oracle's contract.
    """
    column_names = (
        "ID_NUMBER",
        "current_tariff",
        "arpu_segment",
        "data_segment",
        "call_segment",
        "predicted_arpu",
    )
    columns = {name: profile[name].to_numpy(copy=False) for name in column_names}
    row_count = len(profile)

    remaining_money = budget
    remaining_contacts = contacts
    total_cost = 0.0
    total_contacts = 0
    best_by_customer: dict[Any, float] = {}
    per_campaign: list[dict] = []

    for campaign in campaigns or []:
        explicit = campaign.get("explicit_ids")
        if isinstance(explicit, (list, tuple, set, np.ndarray)) and len(explicit) > 0:
            explicit_values = list(explicit)
            selected = [
                row for row in range(row_count)
                if _isin(columns["ID_NUMBER"][row], explicit_values)
            ]
        else:
            selected = list(range(row_count))
            filter_pairs = (
                ("arpu_segment", "filter_arpu_segment"),
                ("data_segment", "filter_data_segment"),
                ("call_segment", "filter_call_segment"),
            )
            for profile_column, campaign_key in filter_pairs:
                wanted = campaign.get(campaign_key)
                if _notna(wanted):
                    selected = [
                        row for row in selected
                        if _equal(columns[profile_column][row], wanted)
                    ]

            tariff_filter = campaign.get("filter_current_tariff")
            if _notna(tariff_filter):
                wanted_tariffs = [
                    item.strip()
                    for item in str(tariff_filter).split(";")
                    if item.strip()
                ]
                selected = [
                    row for row in selected
                    if _isin(columns["current_tariff"][row], wanted_tariffs)
                ]

        ids = columns["ID_NUMBER"]
        selected.sort(key=lambda row: (not _notna(ids[row]), ids[row] if _notna(ids[row]) else 0))

        n_segment = len(selected)
        per_limit = max(int(max_per_campaign), 0)
        capped_campaign = len(selected) > per_limit
        if capped_campaign:
            selected = selected[:per_limit]
        capped_reach = len(selected) > remaining_contacts
        if capped_reach:
            selected = selected[:max(int(remaining_contacts), 0)]

        channel = campaign["channel"]
        unit_cost = channels[channel]["cost_per_contact"]
        capped_money = False
        if unit_cost > 0:
            affordable = int(remaining_money // unit_cost)
            if len(selected) > affordable:
                capped_money = True
                selected = selected[:max(affordable, 0)]

        contacted = len(selected)
        campaign_cost = contacted * unit_cost
        remaining_contacts -= contacted
        remaining_money -= campaign_cost
        total_contacts += contacted
        total_cost += campaign_cost

        target = campaign["target_tariff"]
        campaign_lifts: list[float] = []
        for row in selected:
            ratio = ratio_fn(
                columns["current_tariff"][row],
                columns["arpu_segment"][row],
                target,
                channel,
            )
            lift = ratio * columns["predicted_arpu"][row]
            if not _notna(lift):
                lift = 0.0
            campaign_lifts.append(float(lift))

            customer_id = ids[row]
            # pandas groupby drops missing keys, although their contacts and
            # communication costs still count.
            if not _notna(customer_id):
                continue
            lift = float(lift)
            previous = best_by_customer.get(customer_id)
            if previous is None or lift > previous:
                best_by_customer[customer_id] = lift

        per_campaign.append(
            {
                "n_segment": n_segment,
                "n_contacted": contacted,
                "cost": float(campaign_cost),
                "gross": float(math.fsum(campaign_lifts)),
                "capped_campaign": capped_campaign,
                "capped_reach": capped_reach,
                "capped_money": capped_money,
            }
        )

    # fsum keeps the oracle's own rounding error negligible versus pandas sums.
    gross = float(math.fsum(best_by_customer.values()))
    cost = float(total_cost)
    return {
        "gross": gross,
        "cost": cost,
        "contacts": total_contacts,
        "net": gross - cost,
        "per_campaign": per_campaign,
    }
