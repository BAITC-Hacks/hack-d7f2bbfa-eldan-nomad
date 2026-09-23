"""Differential tests: ScoreSimulator vs organizer scoring_core.score_campaigns."""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scoring_core as sc
from agent_src.m40_simulator import ScoreSimulator

SEGS = {
    "arpu_segment": ["LOW", "MID", "HIGH"],
    "data_segment": ["NON_USER", "LITE", "HEAVY"],
    "call_segment": ["LOW", "MEDIUM", "HIGH"],
}


def _fake_dv(tariffs: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        tariff_codes=list(tariffs["tariff_plan_code"]),
        channels=list(sc.CHANNELS),
        cost=lambda ch: float(sc.CHANNELS[ch]["cost_per_contact"]),
        mult=lambda ch: float(sc.CHANNELS[ch]["conversion_multiplier"]),
    )


def _fallback(current, target, seg, dict_tariff, fallback_conversion):
    """Price-based fallback (same shape as the organizer mock)."""
    price = dict_tariff.set_index("tariff_plan_code")["price_tariff"]
    if current not in price.index or target not in price.index:
        return 0.0, fallback_conversion
    scale = max(price.median(), 1.0)
    ratio = (price[target] - price[current]) / scale
    return float(np.clip(ratio * 0.4, -1.0, 3.0)), fallback_conversion


def _ratio_fn_from(impact: pd.DataFrame, tariffs: pd.DataFrame):
    """ratio_fn equivalent to scoring_core.score_campaign for the given impact model."""
    fc = impact["conversion_rate"].median()
    table = {
        (r.tariff_plan_code_from, r.arpu_segment, r.tariff_plan_code_to): (r.arpu_change_pct, r.conversion_rate)
        for r in impact.itertuples()
    }

    def ratio_fn(cur, seg, target, channel):
        hit = table.get((cur, seg, target))
        if hit is None:
            hit = _fallback(cur, target, seg, tariffs, fc)
        pct, conv = hit
        mult = sc.CHANNELS[channel]["conversion_multiplier"]
        return pct * min(conv * mult, 1.0)

    return ratio_fn


def _synth_world(seed: int, n: int = 12000, n_tariffs: int = 6):
    rng = np.random.default_rng(seed)
    codes = [f"tariff_{i}" for i in range(1, n_tariffs + 1)]
    tariffs = pd.DataFrame({"tariff_plan_code": codes,
                            "price_tariff": np.round(rng.uniform(0, 10000, n_tariffs), 0)})
    prof = pd.DataFrame({
        "ID_NUMBER": rng.permutation(np.arange(10, 10 + n)),
        "current_tariff": np.array(codes)[rng.integers(0, n_tariffs, n)],
        "arpu_segment": np.array(SEGS["arpu_segment"])[rng.integers(0, 3, n)],
        "data_segment": np.array(SEGS["data_segment"])[rng.integers(0, 3, n)],
        "call_segment": np.array(SEGS["call_segment"])[rng.integers(0, 3, n)],
        "predicted_arpu": rng.uniform(200, 15000, n),
    })
    prof = prof.astype({c: "str" for c in ["current_tariff", "arpu_segment", "data_segment", "call_segment"]})
    for col in ["current_tariff", "arpu_segment", "data_segment"]:
        prof.loc[rng.choice(n, 25, replace=False), col] = np.nan
    rows = []
    for f in codes:
        for s in SEGS["arpu_segment"]:
            for t in codes:
                if f != t and rng.random() < 0.6:
                    rows.append((f, t, s, rng.normal(0.05, 0.3), rng.uniform(0.05, 0.9)))
    impact = pd.DataFrame(rows, columns=["tariff_plan_code_from", "tariff_plan_code_to", "arpu_segment",
                                         "arpu_change_pct", "conversion_rate"])
    return prof, tariffs, impact


def _random_campaign(rng, codes, ids, i):
    def pick(vals):
        r = rng.random()
        if r < 0.45:
            return None
        if r < 0.55:
            return np.nan
        if r < 0.58:
            return "BOGUS"
        return vals[rng.integers(0, len(vals))]

    c = {"campaign_name": f"c{i}", "target_tariff": codes[rng.integers(0, len(codes))],
         "channel": list(sc.CHANNELS)[rng.integers(0, 4)]}
    for key, col in [("filter_arpu_segment", "arpu_segment"), ("filter_data_segment", "data_segment"),
                     ("filter_call_segment", "call_segment")]:
        c[key] = pick(SEGS[col])
    r = rng.random()
    if r < 0.4:
        c["filter_current_tariff"] = None
    elif r < 0.45:
        c["filter_current_tariff"] = ""
    else:
        k = int(rng.integers(1, 4))
        chosen = list(rng.choice(codes, k, replace=False))
        if rng.random() < 0.2:
            chosen.append("tariff_unknown")
        c["filter_current_tariff"] = " ; ".join(chosen) + (";" if rng.random() < 0.3 else "")
    if rng.random() < 0.08:
        c["explicit_ids"] = [int(x) for x in rng.choice(ids, int(rng.integers(1, 300)), replace=False)]
    return c


def _reference(campaigns, prof, impact, tariffs):
    strategy = pd.DataFrame(campaigns)
    for col in ["filter_arpu_segment", "filter_data_segment", "filter_call_segment",
                "filter_current_tariff", "explicit_ids"]:
        if col not in strategy.columns:
            strategy[col] = None
    return sc.score_campaigns(strategy, prof, impact, tariffs, float(prof["predicted_arpu"].sum()), _fallback)


def _assert_match(res, ref, campaigns):
    assert res.cost == pytest.approx(ref["total_cost"], rel=1e-9, abs=1e-9)
    assert res.contacts == ref["total_contacts"]
    assert res.gross == pytest.approx(ref["gross_arpu_lift"], rel=1e-6, abs=1e-6)
    assert res.net == pytest.approx(ref["net_arpu_gain"], rel=1e-6, abs=1e-6)
    assert len(res.per_campaign) == len(ref["campaigns_detail"])
    for mine, theirs in zip(res.per_campaign, ref["campaigns_detail"]):
        assert mine["n_contacted"] == theirs["n_contacts"]
        assert mine["cost"] == pytest.approx(theirs["cost"])
        assert mine["gross"] == pytest.approx(theirs["gross_lift"], rel=1e-6, abs=1e-6)
        assert mine["capped_campaign"] == theirs["capped_at_campaign_limit"]
        assert mine["capped_reach"] == theirs["capped_at_reach_budget"]
        assert mine["capped_money"] == theirs["capped_at_money_budget"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_differential_random(seed):
    prof, tariffs, impact = _synth_world(seed)
    codes = list(tariffs["tariff_plan_code"])
    sim = ScoreSimulator(_fake_dv(tariffs), prof)
    ratio_fn = _ratio_fn_from(impact, tariffs)
    rng = np.random.default_rng(100 + seed)
    ids = prof["ID_NUMBER"].to_numpy()
    for trial in range(12):
        k = int(rng.integers(1, 11))
        campaigns = [_random_campaign(rng, codes, ids, i) for i in range(k)]
        if trial % 4 == 0:  # force big unfiltered campaigns -> 5000 cap + reach cap
            campaigns[0] = {"campaign_name": "big", "target_tariff": codes[1], "channel": "push"}
            campaigns.append({"campaign_name": "big2", "target_tariff": codes[2], "channel": "push",
                              "filter_arpu_segment": None})
        if trial % 4 == 1:  # force money cap
            campaigns.insert(0, {"campaign_name": "calls", "target_tariff": codes[0], "channel": "call"})
        res = sim.simulate(campaigns, ratio_fn, sc.TOTAL_BUDGET, sc.MAX_TOTAL_CONTACTS)
        ref = _reference(campaigns, prof, impact, tariffs)
        _assert_match(res, ref, campaigns)


def test_caps_flags_and_within_limits():
    prof, tariffs, impact = _synth_world(5)
    codes = list(tariffs["tariff_plan_code"])
    sim = ScoreSimulator(_fake_dv(tariffs), prof)
    ratio_fn = _ratio_fn_from(impact, tariffs)
    camps = [{"campaign_name": "a", "target_tariff": codes[0], "channel": "push"},
             {"campaign_name": "b", "target_tariff": codes[1], "channel": "call"}]
    res = sim.simulate(camps, ratio_fn, sc.TOTAL_BUDGET, sc.MAX_TOTAL_CONTACTS)
    assert res.per_campaign[0]["capped_campaign"] and res.per_campaign[0]["n_contacted"] == 5000
    assert res.per_campaign[1]["capped_money"] and res.per_campaign[1]["n_contacted"] == 100_000 // 160
    assert not res.within_limits
    small = [{"campaign_name": "s", "target_tariff": codes[1], "channel": "sms",
              "filter_current_tariff": codes[0], "filter_arpu_segment": "HIGH", "filter_data_segment": "LITE"}]
    res2 = sim.simulate(small, ratio_fn, sc.TOTAL_BUDGET, sc.MAX_TOTAL_CONTACTS)
    assert res2.within_limits and res2.contacts == len(sim.segment(small[0]))
    assert res2.net == pytest.approx(res2.gross - res2.cost)
    eleven = small * 11
    assert not sim.simulate(eleven, ratio_fn, 1e9, 10**9).within_limits


def test_remaining_limits_and_dedup():
    prof, tariffs, impact = _synth_world(6)
    codes = list(tariffs["tariff_plan_code"])
    sim = ScoreSimulator(_fake_dv(tariffs), prof)
    camp = {"campaign_name": "x", "target_tariff": codes[2], "channel": "sms",
            "filter_current_tariff": f"{codes[0]};{codes[1]}"}
    res = sim.simulate([camp], lambda *a: 0.1, budget=40.0, contacts=100)
    assert res.contacts == 10 and res.cost == 40.0 and not res.within_limits
    res = sim.simulate([camp], lambda *a: 0.1, budget=1e9, contacts=7)
    assert res.contacts == 7 and res.per_campaign[0]["capped_reach"]
    # duplicate campaign: cost doubles, gross counted once (max)
    one = sim.simulate([camp], lambda *a: 0.1, 1e9, 10**6)
    two = sim.simulate([camp, camp], lambda *a: 0.1, 1e9, 10**6)
    assert two.cost == pytest.approx(2 * one.cost) and two.gross == pytest.approx(one.gross)
    # unknown target/channel skipped like sanitize_campaigns
    bad = sim.simulate([{"target_tariff": "nope", "channel": "sms"}, {"target_tariff": codes[0], "channel": "fax"}],
                       lambda *a: 0.1, 1e9, 10**6)
    assert bad.contacts == 0 and bad.gross == 0.0 and all(p.get("dropped") for p in bad.per_campaign)


def test_segment_sorted_and_raw_filters():
    prof, tariffs, _ = _synth_world(7, n=2000)
    sim = ScoreSimulator(_fake_dv(tariffs), prof)
    camp = {"filter_current_tariff": " tariff_1 ;tariff_2;", "filter_arpu_segment": "MID", "filter_data_segment": None}
    seg = sim.segment(camp)
    ref = sc.apply_filters(prof, pd.Series(camp)).sort_values("ID_NUMBER")
    assert seg["ID_NUMBER"].tolist() == ref["ID_NUMBER"].tolist()
    assert seg["ID_NUMBER"].is_monotonic_increasing


def test_real_profile_mock_model_and_speed(repo_root):
    from mock_environment import _mock_fallback, _mock_impact_model

    prof = pd.read_csv(repo_root / "customer_profile.csv")
    tariffs = pd.read_csv(repo_root / "data" / "dict_tariff.csv")
    impact = _mock_impact_model(pd.read_csv(repo_root / "data" / "change_tariff.csv"))
    impact = impact.astype({"arpu_segment": "str"})
    sim = ScoreSimulator(_fake_dv(tariffs), prof)
    fc = impact["conversion_rate"].median()
    table = {(r.tariff_plan_code_from, r.arpu_segment, r.tariff_plan_code_to): (r.arpu_change_pct, r.conversion_rate)
             for r in impact.itertuples()}

    def ratio_fn(cur, seg, target, channel):
        pct, conv = table.get((cur, seg, target)) or _mock_fallback(cur, target, seg, tariffs, fc)
        return pct * min(conv * sc.CHANNELS[channel]["conversion_multiplier"], 1.0)

    codes = list(tariffs["tariff_plan_code"])
    rng = np.random.default_rng(3)
    ids = prof["ID_NUMBER"].to_numpy()
    for _ in range(3):
        campaigns = [_random_campaign(rng, codes, ids, i) for i in range(10)]
        t0 = time.perf_counter()
        res = sim.simulate(campaigns, ratio_fn, sc.TOTAL_BUDGET, sc.MAX_TOTAL_CONTACTS)
        assert time.perf_counter() - t0 < 1.0  # generous: guards O(n^2) regressions only
        strategy = pd.DataFrame(campaigns)
        for col in ["filter_arpu_segment", "filter_data_segment", "filter_call_segment",
                    "filter_current_tariff", "explicit_ids"]:
            if col not in strategy.columns:
                strategy[col] = None
        ref = sc.score_campaigns(strategy, prof, _mock_impact_model(pd.read_csv(repo_root / "data" / "change_tariff.csv")),
                                 tariffs, float(prof["predicted_arpu"].sum()), _mock_fallback)
        _assert_match(res, ref, campaigns)
    # whole base, NaN rows included
    full = [{"campaign_name": "all", "target_tariff": codes[-1], "channel": "push"}]
    res = sim.simulate(full, ratio_fn, sc.TOTAL_BUDGET, sc.MAX_TOTAL_CONTACTS)
    strategy = pd.DataFrame(full).assign(filter_arpu_segment=None, explicit_ids=None)
    ref = sc.score_campaigns(strategy, prof, _mock_impact_model(pd.read_csv(repo_root / "data" / "change_tariff.csv")),
                             tariffs, 1.0, _mock_fallback)
    _assert_match(res, ref, full)


@pytest.mark.parametrize("budget,contacts", [(0.5, 15000), (3.0, 15000), (160.0, 3), (5000.0, 0), (37_123.0, 9_000)])
def test_differential_remaining_limits(monkeypatch, budget, contacts):
    """Post-pilot mode: starting limits below the organizer totals (reference patched to match)."""
    prof, tariffs, impact = _synth_world(11)
    codes = list(tariffs["tariff_plan_code"])
    sim = ScoreSimulator(_fake_dv(tariffs), prof)
    ratio_fn = _ratio_fn_from(impact, tariffs)
    monkeypatch.setattr(sc, "TOTAL_BUDGET", budget)
    monkeypatch.setattr(sc, "MAX_TOTAL_CONTACTS", contacts)
    rng = np.random.default_rng(42)
    ids = prof["ID_NUMBER"].to_numpy()
    for trial in range(4):
        campaigns = [_random_campaign(rng, codes, ids, i) for i in range(6)]
        campaigns.insert(trial % 3, {"campaign_name": "free", "target_tariff": codes[3], "channel": "push",
                                     "filter_arpu_segment": "MID"})
        campaigns.append({"campaign_name": "paid", "target_tariff": codes[4], "channel": "call",
                          "filter_data_segment": "HEAVY"})
        res = sim.simulate(campaigns, ratio_fn, budget, contacts)
        _assert_match(res, _reference(campaigns, prof, impact, tariffs), campaigns)
        assert res.cost <= budget + 1e-9 and res.contacts <= contacts


def test_duplicate_null_ids_and_nan_arpu_match_reference():
    """Dedup is per ID_NUMBER (groupby), null IDs never count in gross, NaN ARPU -> 0 lift."""
    prof, tariffs, impact = _synth_world(12, n=3000)
    prof["ID_NUMBER"] = prof["ID_NUMBER"].astype(float)
    prof.loc[prof.index[:40], "ID_NUMBER"] = prof["ID_NUMBER"].iloc[40:80].to_numpy()  # duplicates
    prof.loc[prof.index[100:110], "ID_NUMBER"] = np.nan
    prof.loc[prof.index[200:230], "predicted_arpu"] = np.nan
    codes = list(tariffs["tariff_plan_code"])
    sim = ScoreSimulator(_fake_dv(tariffs), prof)
    ratio_fn = _ratio_fn_from(impact, tariffs)
    dup_ids = [float(x) for x in prof["ID_NUMBER"].iloc[40:60]]
    campaigns = [
        {"campaign_name": "a", "target_tariff": codes[1], "channel": "sms", "filter_arpu_segment": "HIGH"},
        {"campaign_name": "b", "target_tariff": codes[2], "channel": "push"},
        {"campaign_name": "p", "target_tariff": codes[3], "channel": "sms", "explicit_ids": dup_ids},
        {"campaign_name": "c", "target_tariff": codes[4], "channel": "digital_ads", "filter_call_segment": "LOW"},
    ]
    res = sim.simulate(campaigns, ratio_fn, sc.TOTAL_BUDGET, sc.MAX_TOTAL_CONTACTS)
    _assert_match(res, _reference(campaigns, prof, impact, tariffs), campaigns)
    assert res.per_campaign[2]["n_contacted"] == 40  # each duplicated ID matches two rows


def test_robustness_non_finite_limits_and_bad_inputs():
    prof, tariffs, _ = _synth_world(13, n=500)
    codes = list(tariffs["tariff_plan_code"])
    sim = ScoreSimulator(_fake_dv(tariffs), prof)
    camp = {"campaign_name": "x", "target_tariff": codes[1], "channel": "sms"}
    res = sim.simulate([camp], lambda *a: 0.1, float("inf"), float("inf"))
    assert res.contacts == 500 and res.within_limits
    for b, c in [(float("nan"), 100), (float("-inf"), 100), (1e9, float("nan")), (-5.0, 100), (1e9, -3)]:
        res = sim.simulate([camp], lambda *a: 0.1, b, c)
        assert res.contacts == 0 and res.cost == 0.0 and res.gross == 0.0
    # unhashable target/channel -> dropped, not TypeError; unhashable explicit ids -> empty
    bad = sim.simulate([{"target_tariff": [codes[1]], "channel": "sms"},
                        {"target_tariff": codes[1], "channel": {"sms"}},
                        {"target_tariff": codes[1], "channel": "sms", "explicit_ids": [[1, 2]]}],
                       lambda *a: 0.1, 1e9, 10**6)
    assert [p.get("dropped", False) for p in bad.per_campaign] == [True, True, False]
    assert bad.contacts == 0
    # non-finite ratios count as 0 consistently in per-campaign and total gross
    res = sim.simulate([camp], lambda cur, seg, t, ch: float("inf") if seg == "HIGH" else 0.1, 1e9, 10**6)
    assert math.isfinite(res.gross) and res.gross > 0
    assert res.per_campaign[0]["gross"] == pytest.approx(res.gross)
    # ratio_fn errors -> ratio 0 and counted
    def boom(cur, seg, t, ch):
        if seg == "LOW":
            raise KeyError(seg)
        return 0.1
    res = sim.simulate([camp], boom, 1e9, 10**6)
    assert res.per_campaign[0]["ratio_errors"] >= 1 and res.gross > 0
    # name default only when the key is absent (organizer uses .get with default)
    res = sim.simulate([{"target_tariff": codes[1], "channel": "push"},
                        {"campaign_name": None, "target_tariff": codes[1], "channel": "push"}],
                       lambda *a: 0.1, 1e9, 10**6)
    assert [p["name"] for p in res.per_campaign] == ["campaign_0", None]


def test_empty_profile_and_no_campaigns():
    prof, tariffs, _ = _synth_world(14, n=50)
    sim = ScoreSimulator(_fake_dv(tariffs), prof.iloc[:0])
    res = sim.simulate([{"target_tariff": "tariff_1", "channel": "sms"}], lambda *a: 0.1, 1e9, 100)
    assert res.contacts == 0 and res.gross == 0.0 and res.within_limits
    res = ScoreSimulator(_fake_dv(tariffs), prof).simulate([], lambda *a: 0.1, 1e9, 100)
    assert res.per_campaign == [] and res.net == 0.0 and res.within_limits


def test_deterministic_repeat():
    prof, tariffs, impact = _synth_world(15, n=4000)
    codes = list(tariffs["tariff_plan_code"])
    sim = ScoreSimulator(_fake_dv(tariffs), prof)
    ratio_fn = _ratio_fn_from(impact, tariffs)
    rng = np.random.default_rng(7)
    camps = [_random_campaign(rng, codes, prof["ID_NUMBER"].to_numpy(), i) for i in range(8)]
    a = sim.simulate(camps, ratio_fn, 50_000.0, 9_000)
    b = sim.simulate(camps, ratio_fn, 50_000.0, 9_000)
    assert a == b
