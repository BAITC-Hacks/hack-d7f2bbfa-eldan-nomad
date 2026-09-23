#!/usr/bin/env python3
"""Print compact, README-ready facts about profile and tariff history data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mock_environment import _mock_impact_model
from scoring_core import CHANNELS


def _fmt_quantiles(series: pd.Series) -> str:
    q = series.dropna().quantile([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return "  ".join(f"p{int(p * 100):02d}={value:+.4f}" for p, value in q.items())


def _int_stat(series: pd.Series, how: str) -> str:
    """Integer min/max of a possibly empty series, ``n/a`` when empty."""
    return "n/a" if series.empty else f"{int(getattr(series, how)()):,}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=ROOT / "customer_profile.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args(argv)

    profile = pd.read_csv(args.profile)
    history = pd.read_csv(args.data_dir / "change_tariff.csv")
    tariffs = pd.read_csv(args.data_dir / "dict_tariff.csv")
    impact = _mock_impact_model(history)

    cell_cols = ["current_tariff", "arpu_segment"]
    subcell_cols = cell_cols + ["data_segment", "call_segment"]
    cells = profile.dropna(subset=cell_cols).groupby(cell_cols, observed=True).size()
    subcells = profile.dropna(subset=subcell_cols).groupby(subcell_cols, observed=True).size()
    possible_subcells = len(cells) * 3 * 3

    print("=== Audience cells ===")
    print(f"customers: {len(profile):,}")
    print(f"non-empty tariff x ARPU cells: {len(cells):,} (largest: {_int_stat(cells, 'max')})")
    print(
        f"non-empty tariff x ARPU x data x call subcells: {len(subcells):,} "
        f"of {possible_subcells:,}; empty filter combinations: {possible_subcells - len(subcells):,}"
    )
    print(f"subcells with n >= 50: {int((subcells >= 50).sum()):,}")

    key_nan = profile[["current_tariff", "arpu_segment", "data_segment", "call_segment"]].isna()
    print("\n=== Missing values ===")
    print(f"profile rows with any NaN: {int(profile.isna().any(axis=1).sum()):,}")
    print(f"profile rows with NaN in targeting keys: {int(key_nan.any(axis=1).sum()):,}")
    for column in key_nan.columns:
        print(f"  {column}: {int(key_nan[column].sum()):,}")
    print(f"history rows with any NaN: {int(history.isna().any(axis=1).sum()):,}")

    valid_history = history[history["AVG_ARPU_PREV_3M"] >= 100].copy()
    valid_history["arpu_segment"] = pd.cut(
        valid_history["AVG_ARPU_PREV_3M"],
        bins=[-np.inf, 1000, 5000, np.inf],
        labels=["LOW", "MID", "HIGH"],
    )
    all_arm_counts = valid_history.groupby(
        ["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment"], observed=True
    ).size()
    # Eligible arms: non-empty profile (tariff, ARPU) cells x known targets,
    # excluding the self "switch" to the current tariff.
    targets = set(tariffs["tariff_plan_code"].dropna().astype(str))
    profile_cells = {(str(f), str(s)) for f, s in cells.index}
    possible_arms = sum(len(targets - {f}) for f, _ in profile_cells)
    eligible = [
        (str(f), str(s)) in profile_cells and str(t) in targets and str(t) != str(f)
        for f, t, s in all_arm_counts.index
    ]
    arm_counts = all_arm_counts[np.asarray(eligible, dtype=bool)]
    print("\n=== History arm coverage ===")
    print(
        f"observed eligible arms: {len(arm_counts):,} of {possible_arms:,} "
        f"(profile cell x non-self target); all history arms: {len(all_arm_counts):,}"
    )
    print(f"eligible arms with n >= 20: {int((arm_counts >= 20).sum()):,}")
    if arm_counts.empty:
        print("arm observations: n/a")
    else:
        print(
            f"arm observations: min={_int_stat(arm_counts, 'min')}, median={arm_counts.median():.0f}, "
            f"p90={arm_counts.quantile(0.9):.0f}, max={_int_stat(arm_counts, 'max')}"
        )
    overlap = set(profile["ID_NUMBER"]) & set(history["ID_NUMBER"])
    print(f"profile/history customer-ID overlap: {len(overlap):,}")

    changes = ((valid_history["AVG_ARPU_NEXT_3M"] - valid_history["AVG_ARPU_PREV_3M"])
               / valid_history["AVG_ARPU_PREV_3M"]).clip(-1, 3)
    base_effect = impact["arpu_change_pct"] * impact["conversion_rate"]
    print("\n=== Effect distribution ===")
    print(f"per-customer clipped relative change: mean={changes.mean():+.4f}, std={changes.std():.4f}")
    print("relative change quantiles: " + _fmt_quantiles(changes))
    print("mock arm base-effect quantiles (change x conversion): " + _fmt_quantiles(base_effect))
    print(f"negative mock arms: {(impact['arpu_change_pct'] < 0).mean():.1%}")

    price = tariffs.set_index("tariff_plan_code")["price_tariff"]
    # Same population as the ARPU-downsell line: valid history rows only.
    source_price = valid_history["tariff_plan_code_from"].map(price)
    target_price = valid_history["tariff_plan_code_to"].map(price)
    known = source_price.notna() & target_price.notna()
    downsell = target_price[known] < source_price[known]
    print("\n=== Transition direction ===")
    arpu_downsell = valid_history["AVG_ARPU_NEXT_3M"] < valid_history["AVG_ARPU_PREV_3M"]
    print(
        f"observed ARPU downsell among valid history rows: {arpu_downsell.mean():.1%} "
        f"({int(arpu_downsell.sum()):,}/{len(arpu_downsell):,})"
    )
    print(
        f"lower-price target among valid history rows: {downsell.mean():.1%} "
        f"({int(downsell.sum()):,}/{int(known.sum()):,})"
    )

    print("\n=== Channel thresholds ===")
    print("Break-even post-channel lift ratio = contact cost / predicted ARPU.")
    segment_arpu = profile.groupby("arpu_segment", observed=True)["predicted_arpu"].median()
    for segment in ["LOW", "MID", "HIGH"]:
        median_arpu = float(segment_arpu.get(segment, np.nan))
        pieces = []
        for channel, info in CHANNELS.items():
            post = info["cost_per_contact"] / median_arpu if median_arpu > 0 else np.nan
            base = post / info["conversion_multiplier"] if info["conversion_multiplier"] else np.inf
            pieces.append(f"{channel}={base:.4f} base ({post:.4f} post-channel)")
        print(f"{segment} median predicted ARPU {median_arpu:,.1f}: " + "; ".join(pieces))
    ordered = list(CHANNELS)
    print("Pairwise switch thresholds for r x predicted_ARPU (without saturation):")
    for lower, higher in zip(ordered, ordered[1:]):
        a, b = CHANNELS[lower], CHANNELS[higher]
        threshold = ((b["cost_per_contact"] - a["cost_per_contact"])
                     / (b["conversion_multiplier"] - a["conversion_multiplier"]))
        print(f"  {higher} beats {lower} above {threshold:,.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
