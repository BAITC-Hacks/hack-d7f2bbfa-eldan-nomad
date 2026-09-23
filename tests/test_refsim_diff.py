"""Differential tests for the independent campaign-scoring oracle."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scoring_core
from agent_src.m10_dataview import DataView
from agent_src.m40_simulator import ScoreSimulator
from tests.ref_simulator_codex import ref_simulate


TARIFFS = ("tariff_1", "tariff_2", "tariff_3", "tariff_4")
ARPU_SEGMENTS = ("LOW", "MID", "HIGH")
DATA_SEGMENTS = ("NON_USER", "LITE", "HEAVY")
CALL_SEGMENTS = ("LOW", "MEDIUM", "HIGH")


def _inputs():
    rng = np.random.default_rng(7319)
    n = 5_207  # deliberately exceeds the per-campaign limit
    ids = np.arange(10_000, 10_000 + n, dtype=np.int64)
    profile = pd.DataFrame(
        {
            "ID_NUMBER": ids[rng.permutation(n)],
            "current_tariff": rng.choice(TARIFFS[:3], n).astype(object),
            "arpu_segment": rng.choice(ARPU_SEGMENTS, n).astype(object),
            "data_segment": rng.choice(DATA_SEGMENTS, n).astype(object),
            "call_segment": rng.choice(CALL_SEGMENTS, n).astype(object),
            "predicted_arpu": rng.uniform(250.0, 15_000.0, n),
        }
    )
    profile.loc[3, "current_tariff"] = np.nan
    profile.loc[7, "arpu_segment"] = np.nan
    profile.loc[11, "data_segment"] = np.nan
    profile.loc[19, "call_segment"] = np.nan
    profile.loc[23, "predicted_arpu"] = np.nan

    tariffs = pd.DataFrame(
        {
            "tariff_plan_code": TARIFFS,
            "price_tariff": [900.0, 2_400.0, 5_700.0, 10_500.0],
        }
    )
    impact_rows = []
    for source_i, source in enumerate(TARIFFS[:3]):
        for target_i, target in enumerate(TARIFFS):
            for segment_i, segment in enumerate(ARPU_SEGMENTS):
                # Leave a deterministic set of holes to exercise fallback.
                if (source_i + 2 * target_i + segment_i) % 5 == 0:
                    continue
                impact_rows.append(
                    {
                        "tariff_plan_code_from": source,
                        "tariff_plan_code_to": target,
                        "arpu_segment": segment,
                        "arpu_change_pct": -0.35 + 0.17 * target_i + 0.06 * segment_i,
                        "conversion_rate": 0.22 + 0.23 * ((source_i + target_i + segment_i) % 4),
                    }
                )
    impact_model = pd.DataFrame(impact_rows)
    return profile, tariffs, impact_model


def _fallback(current_tariff, target_tariff, arpu_segment, dict_tariff, fallback_conversion):
    del dict_tariff
    source_i = TARIFFS.index(current_tariff) if current_tariff in TARIFFS else -1
    target_i = TARIFFS.index(target_tariff)
    segment_i = ARPU_SEGMENTS.index(arpu_segment) if arpu_segment in ARPU_SEGMENTS else 1
    change = -0.18 + 0.09 * (target_i - source_i) + 0.025 * segment_i
    return change, fallback_conversion


def _make_ratio_fn(impact_model, tariffs, channels=scoring_core.CHANNELS):
    fallback_conversion = float(impact_model["conversion_rate"].median())
    lookup = {
        (row.tariff_plan_code_from, row.arpu_segment, row.tariff_plan_code_to):
            (row.arpu_change_pct, row.conversion_rate)
        for row in impact_model.itertuples(index=False)
    }

    def ratio_fn(current_tariff, arpu_segment, target, channel):
        values = lookup.get((current_tariff, arpu_segment, target))
        # scoring_core falls back whenever the merged arpu_change_pct is NaN.
        if values is None or pd.isna(values[0]):
            values = _fallback(current_tariff, target, arpu_segment, tariffs, fallback_conversion)
        change, conversion = values
        effective_conversion = min(
            conversion * channels[channel]["conversion_multiplier"],
            1.0,
        )
        return float(change * effective_conversion)

    return ratio_fn


def _random_campaigns(rng, profile, case_number):
    campaigns = []
    count = int(rng.integers(0, 14))
    all_ids = profile["ID_NUMBER"].dropna().to_numpy()
    for campaign_i in range(count):
        campaign = {
            "campaign_name": f"case_{case_number}_campaign_{campaign_i}",
            "target_tariff": str(rng.choice(TARIFFS)),
            "channel": str(rng.choice(tuple(scoring_core.CHANNELS))),
        }

        # Empty/missing filters, valid filters, and impossible values all have
        # useful semantics in scoring_core, so draw from all three categories.
        arpu = rng.choice((None, np.nan, *ARPU_SEGMENTS, "UNKNOWN"))
        data = rng.choice((None, np.nan, *DATA_SEGMENTS, "UNKNOWN"))
        call = rng.choice((None, np.nan, *CALL_SEGMENTS, "UNKNOWN"))
        tariff = rng.choice(
            (
                None,
                np.nan,
                "tariff_1",
                "tariff_2; tariff_3",
                " ; tariff_1 ; tariff_4; ",
                "missing_tariff",
            )
        )
        if rng.random() < 0.85:
            campaign["filter_arpu_segment"] = arpu
        if rng.random() < 0.85:
            campaign["filter_data_segment"] = data
        if rng.random() < 0.85:
            campaign["filter_call_segment"] = call
        if rng.random() < 0.85:
            campaign["filter_current_tariff"] = tariff

        explicit_roll = rng.random()
        if explicit_roll < 0.12:
            picked = rng.choice(all_ids, size=int(rng.integers(1, 18)), replace=False)
            explicit_type = int(rng.integers(4))
            if explicit_type == 0:
                campaign["explicit_ids"] = picked.tolist()
            elif explicit_type == 1:
                campaign["explicit_ids"] = tuple(picked.tolist())
            elif explicit_type == 2:
                campaign["explicit_ids"] = set(picked.tolist())
            else:
                campaign["explicit_ids"] = picked
        elif explicit_roll < 0.16:
            campaign["explicit_ids"] = []
        campaigns.append(campaign)

    # Periodic unfiltered campaigns guarantee coverage of the 5,000, reach,
    # and money caps rather than leaving those branches to chance.
    if case_number % 10 == 0:
        campaigns[:0] = [
            {"campaign_name": f"broad_{case_number}_push", "target_tariff": "tariff_4", "channel": "push"},
            {"campaign_name": f"broad_{case_number}_sms", "target_tariff": "tariff_3", "channel": "sms"},
            {"campaign_name": f"broad_{case_number}_call", "target_tariff": "tariff_2", "channel": "call"},
        ]
    return campaigns


def _core_result(profile, tariffs, impact_model, campaigns):
    strategy = pd.DataFrame(campaigns)
    baseline = float(profile["predicted_arpu"].sum())
    return scoring_core.score_campaigns(
        strategy,
        profile,
        impact_model,
        tariffs,
        baseline,
        _fallback,
    )


def _assert_ref_matches_core(reference, core, case_number):
    assert reference["contacts"] == core["total_contacts"], f"case {case_number}"
    np.testing.assert_allclose(reference["cost"], core["total_cost"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(reference["gross"], core["gross_arpu_lift"], rtol=1e-13, atol=1e-8)
    np.testing.assert_allclose(reference["net"], core["net_arpu_gain"], rtol=1e-13, atol=1e-8)
    detail = core["campaigns_detail"]
    assert len(reference["per_campaign"]) == len(detail), f"case {case_number}"
    for ref_c, core_c in zip(reference["per_campaign"], detail):
        assert ref_c["n_contacted"] == core_c["n_contacts"], f"case {case_number}"
        assert ref_c["cost"] == core_c["cost"], f"case {case_number}"
        assert ref_c["capped_campaign"] == core_c["capped_at_campaign_limit"], f"case {case_number}"
        assert ref_c["capped_reach"] == core_c["capped_at_reach_budget"], f"case {case_number}"
        assert ref_c["capped_money"] == core_c["capped_at_money_budget"], f"case {case_number}"
        np.testing.assert_allclose(ref_c["gross"], core_c["gross_lift"], rtol=1e-12, atol=1e-8)


def _assert_agent_matches_ref(agent, reference, campaigns, case_number):
    assert agent.contacts == reference["contacts"], f"case {case_number}"
    np.testing.assert_allclose(agent.cost, reference["cost"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(agent.gross, reference["gross"], rtol=1e-13, atol=1e-8)
    np.testing.assert_allclose(agent.net, reference["net"], rtol=1e-13, atol=1e-8)
    kept = [c for c in agent.per_campaign if not c.get("dropped")]
    assert len(kept) == len(reference["per_campaign"]), f"case {case_number}"
    for agent_c, ref_c in zip(kept, reference["per_campaign"]):
        for key in ("n_segment", "n_contacted", "capped_campaign", "capped_reach", "capped_money"):
            assert agent_c[key] == ref_c[key], f"case {case_number}: {key}"
        np.testing.assert_allclose(agent_c["cost"], ref_c["cost"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(agent_c["gross"], ref_c["gross"], rtol=1e-12, atol=1e-8)
    if len(campaigns) > scoring_core.MAX_CAMPAIGNS:
        assert agent.within_limits is False, f"case {case_number}"


def test_ref_simulate_matches_scoring_core_on_200_seeded_campaign_lists():
    profile, tariffs, impact_model = _inputs()
    ratio_fn = _make_ratio_fn(impact_model, tariffs)
    rng = np.random.default_rng(20_260_923)

    for case_number in range(200):
        campaigns = _random_campaigns(rng, profile, case_number)
        reference = ref_simulate(
            profile,
            campaigns,
            ratio_fn,
            scoring_core.TOTAL_BUDGET,
            scoring_core.MAX_TOTAL_CONTACTS,
        )
        core = _core_result(profile, tariffs, impact_model, campaigns)
        _assert_ref_matches_core(reference, core, case_number)


@pytest.mark.parametrize(
    "budget, contacts",
    [
        (scoring_core.TOTAL_BUDGET, scoring_core.MAX_TOTAL_CONTACTS),
        (7_321.0, 2_345),  # typical remaining limits after pilots
        (0, 4_000),
        (100_000, 0),
    ],
)
def test_three_way_differential_with_agent_simulator(budget, contacts):
    profile, tariffs, impact_model = _inputs()
    ratio_fn = _make_ratio_fn(impact_model, tariffs)
    simulator = ScoreSimulator(DataView(profile, tariffs, scoring_core.CHANNELS), profile)
    rng = np.random.default_rng(40_404)
    full_limits = (budget, contacts) == (scoring_core.TOTAL_BUDGET, scoring_core.MAX_TOTAL_CONTACTS)

    for case_number in range(40 if full_limits else 15):
        campaigns = _random_campaigns(rng, profile, case_number)
        reference = ref_simulate(profile, campaigns, ratio_fn, budget, contacts)
        agent = simulator.simulate(campaigns, ratio_fn, budget, contacts)
        if full_limits:
            core = _core_result(profile, tariffs, impact_model, campaigns)
            _assert_ref_matches_core(reference, core, case_number)
        _assert_agent_matches_ref(agent, reference, campaigns, case_number)


def _edge_profiles(profile):
    nan_ids = profile.copy()
    nan_ids["ID_NUMBER"] = nan_ids["ID_NUMBER"].astype(float)
    nan_ids.loc[[5, 9, 40], "ID_NUMBER"] = np.nan
    duplicate_ids = pd.concat([profile.iloc[:300], profile.iloc[:100]], ignore_index=True)
    string_ids = profile.copy()
    string_ids["ID_NUMBER"] = string_ids["ID_NUMBER"].astype(str)
    return {"nan_ids": nan_ids, "duplicate_ids": duplicate_ids, "string_ids": string_ids}


def test_ref_simulate_matches_scoring_core_on_edge_profiles():
    profile, tariffs, impact_model = _inputs()
    ratio_fn = _make_ratio_fn(impact_model, tariffs)
    first_id = profile["ID_NUMBER"].iloc[0]
    campaigns = [
        {"target_tariff": "tariff_4", "channel": "push"},
        {"target_tariff": "tariff_3", "channel": "sms", "filter_arpu_segment": "HIGH"},
        {"target_tariff": "tariff_2", "channel": "call", "filter_current_tariff": "tariff_1;tariff_1"},
        {"target_tariff": "tariff_4", "channel": "digital_ads", "filter_data_segment": pd.NA},
        {"target_tariff": "tariff_4", "channel": "sms", "explicit_ids": [1, 2, first_id]},
    ]
    for label, edge in _edge_profiles(profile).items():
        reference = ref_simulate(
            edge, campaigns, ratio_fn, scoring_core.TOTAL_BUDGET, scoring_core.MAX_TOTAL_CONTACTS
        )
        core = _core_result(edge, tariffs, impact_model, campaigns)
        _assert_ref_matches_core(reference, core, label)


@pytest.mark.parametrize("budget, contacts", [(0, 15_000), (-5, 15_000), (100_000, 0), (100_000, -3), (3, 7)])
def test_ref_simulate_respects_exhausted_limits(budget, contacts):
    profile, tariffs, impact_model = _inputs()
    ratio_fn = _make_ratio_fn(impact_model, tariffs)
    campaigns = [
        {"target_tariff": "tariff_4", "channel": "sms"},
        {"target_tariff": "tariff_3", "channel": "push"},
        {"target_tariff": "tariff_2", "channel": "call"},
    ]
    result = ref_simulate(profile, campaigns, ratio_fn, budget, contacts)
    assert result["contacts"] <= max(contacts, 0)
    assert result["cost"] <= max(budget, 0)
    if contacts <= 0:
        assert (result["gross"], result["cost"], result["contacts"], result["net"]) == (0.0, 0.0, 0, 0.0)
    if budget <= 0 < contacts:
        assert result["cost"] == 0.0
        assert result["contacts"] == min(contacts, scoring_core.MAX_CUSTOMERS_PER_CAMPAIGN)


def test_three_way_differential_on_edge_profiles():
    profile, tariffs, impact_model = _inputs()
    ratio_fn = _make_ratio_fn(impact_model, tariffs)
    campaigns = [
        {"target_tariff": "tariff_4", "channel": "sms"},
        {"target_tariff": "tariff_2", "channel": "call", "filter_arpu_segment": "LOW"},
        {"target_tariff": "tariff_3", "channel": "push", "explicit_ids": [profile["ID_NUMBER"].iloc[0]]},
    ]
    for label, edge in _edge_profiles(profile).items():
        simulator = ScoreSimulator(DataView(edge, tariffs, scoring_core.CHANNELS), edge)
        reference = ref_simulate(
            edge, campaigns, ratio_fn, scoring_core.TOTAL_BUDGET, scoring_core.MAX_TOTAL_CONTACTS
        )
        core = _core_result(edge, tariffs, impact_model, campaigns)
        _assert_ref_matches_core(reference, core, label)
        agent = simulator.simulate(
            campaigns, ratio_fn, scoring_core.TOTAL_BUDGET, scoring_core.MAX_TOTAL_CONTACTS
        )
        _assert_agent_matches_ref(agent, reference, campaigns, label)


@pytest.mark.parametrize(
    "value, candidates, expected",
    [
        (None, [None], True),
        (np.nan, [np.nan], True),
        (pd.NA, [pd.NA], True),
        (np.nan, [None], False),
        (None, [np.nan], False),
        (pd.NA, [np.nan], False),
    ],
)
def test_explicit_missing_sentinels_follow_series_isin(value, candidates, expected):
    series = pd.Series([value, 1.0], dtype=object)
    assert bool(series.isin(candidates).iloc[0]) is expected
    profile = pd.DataFrame(
        {
            "ID_NUMBER": pd.Series([value, 1.0], dtype=object),
            "current_tariff": ["tariff_1", "tariff_1"],
            "arpu_segment": ["LOW", "LOW"],
            "data_segment": ["LITE", "LITE"],
            "call_segment": ["LOW", "LOW"],
            "predicted_arpu": [1_000.0, 1_000.0],
        }
    )
    campaigns = [{"target_tariff": "tariff_2", "channel": "sms", "explicit_ids": list(candidates)}]
    result = ref_simulate(profile, campaigns, lambda *_: 0.1, 100_000, 15_000)
    assert result["contacts"] == int(expected)


@pytest.mark.parametrize("limit, expected", [(0, 0), (-1, 0), (3, 3)])
def test_ref_simulate_clamps_per_campaign_limit(limit, expected):
    profile, tariffs, impact_model = _inputs()
    ratio_fn = _make_ratio_fn(impact_model, tariffs)
    campaigns = [{"target_tariff": "tariff_4", "channel": "push"}]
    result = ref_simulate(profile, campaigns, ratio_fn, 100_000, 15_000, max_per_campaign=limit)
    assert result["contacts"] == expected


def test_ratio_fn_falls_back_on_nan_impact_row_like_scoring_core():
    profile, tariffs, impact_model = _inputs()
    impact_model = impact_model.copy()
    impact_model.loc[impact_model.index[:6], "arpu_change_pct"] = np.nan
    ratio_fn = _make_ratio_fn(impact_model, tariffs)
    campaigns = [
        {"target_tariff": target, "channel": "sms", "filter_current_tariff": "tariff_1"}
        for target in TARIFFS
    ]
    reference = ref_simulate(profile, campaigns, ratio_fn, 100_000, 15_000)
    core = _core_result(profile, tariffs, impact_model, campaigns)
    _assert_ref_matches_core(reference, core, "nan-impact")
