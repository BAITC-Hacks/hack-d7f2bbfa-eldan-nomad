"""Tests for agent_src.m60_packer.CampaignPacker (fakes for DataView / simulator / model)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from agent_src.contract import (
    CALL_SEGMENTS,
    CAMPAIGN_KEYS,
    DATA_SEGMENTS,
    Config,
    Plan,
    RunLog,
)
from agent_src.m60_packer import CampaignPacker
from scoring_core import validate_strategy
from tests.fakes_packer import FakeDataView, FakeModel, FakeSim, make_option

CODES = [f"tariff_{i}" for i in range(1, 11)]


def grid_profile(tariffs, arpus, datas, calls, per: int, seed: int = 0) -> pd.DataFrame:
    """Profile with `per` rows in every combo; shuffled IDs."""
    rows = list(itertools.product(tariffs, arpus, datas, calls))
    n = len(rows) * per
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([r for r in rows for _ in range(per)], columns=[
        "current_tariff", "arpu_segment", "data_segment", "call_segment"])
    df["ID_NUMBER"] = rng.permutation(np.arange(1, n + 1))
    df["predicted_arpu"] = np.round(rng.uniform(1000, 9000, n), 2)
    return df


def build(profile, channels, codes=CODES, **cfg_kw):
    dv = FakeDataView(profile, codes, channels)
    sim = FakeSim(profile, channels)
    return dv, sim, CampaignPacker(Config(**cfg_kw), dv, sim, log=RunLog())


def dict_tariff(codes=CODES) -> pd.DataFrame:
    return pd.DataFrame({"tariff_plan_code": codes})


def check_valid(campaigns, sim, codes=CODES, max_c=10, max_per=5000):
    assert 1 <= len(campaigns) <= max_c
    validate_strategy(pd.DataFrame(campaigns), dict_tariff(codes))
    seen: set = set()
    names = set()
    for c in campaigns:
        assert list(c) == list(CAMPAIGN_KEYS)
        assert c["filter_current_tariff"] and c["filter_arpu_segment"]
        froms = c["filter_current_tariff"].split(";")
        assert c["target_tariff"] not in froms
        ids = set(sim.segment(c)["ID_NUMBER"].tolist())
        indivisible = len(froms) == 1 and c["filter_data_segment"] and c["filter_call_segment"]
        assert 0 < len(ids) and (len(ids) <= max_per or indivisible)  # scorer caps indivisible subs
        assert not (ids & seen), f"overlap in {c['campaign_name']}"
        seen |= ids
        names.add(c["campaign_name"])
    assert len(names) == len(campaigns)
    return seen


def test_groups_tariffs_into_one_campaign(synth_channels):
    prof = grid_profile(CODES[:3], ["HIGH"], ["HEAVY"], ["LOW", "HIGH"], per=5)
    dv, sim, packer = build(prof, synth_channels)
    ratios = {(t, "HIGH", "tariff_4"): 0.2 for t in CODES[:3]}
    opts = [make_option(dv, (t, "HIGH", "HEAVY", "LOW"), "tariff_4", "sms") for t in CODES[:3]]
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), FakeModel(ratios, synth_channels), 100_000, 15_000)
    assert len(out) == 1
    c = out[0]
    assert c["filter_current_tariff"] == "tariff_1;tariff_2;tariff_3"
    assert (c["filter_arpu_segment"], c["filter_data_segment"], c["filter_call_segment"]) == ("HIGH", "HEAVY", "LOW")
    assert c["campaign_name"] == "c01_t4_sms_HIGH_HEAVY_LOW_t1+t2+t3"
    check_valid(out, sim)


def test_target_never_in_from_list(synth_channels):
    prof = grid_profile(CODES[:3], ["MID"], ["LITE"], ["LOW"], per=5)
    dv, sim, packer = build(prof, synth_channels)
    opts = [make_option(dv, (t, "MID", "LITE", "LOW"), "tariff_2", "push") for t in CODES[:3]]
    model = FakeModel({(t, "MID", "tariff_2"): 0.1 for t in CODES[:3]}, synth_channels)
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), model, 100_000, 15_000)
    assert len(out) == 1 and out[0]["filter_current_tariff"] == "tariff_1;tariff_3"
    check_valid(out, sim)


def test_coarsens_identical_call_splits(synth_channels):
    # 12 (arpu,data,call) groups, but every sub of (tariff, arpu, data) goes to the same arm+channel
    prof = grid_profile(CODES[:2], ["HIGH", "MID"], ["HEAVY", "LITE"], CALL_SEGMENTS, per=4)
    dv, sim, packer = build(prof, synth_channels)
    opts, ratios = [], {}
    for sk in dv.subs:
        tgt = "tariff_5" if sk[1] == "HIGH" else "tariff_6"
        opts.append(make_option(dv, sk, tgt, "sms"))
        ratios[(sk[0], sk[1], tgt)] = 0.3
    fine_ids = {i for s in dv.subs.values() for i in s.ids.tolist()}
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), FakeModel(ratios, synth_channels), 100_000, 15_000)
    assert len(out) <= 10
    seen = check_valid(out, sim)
    assert seen == fine_ids  # merges are exact: no customer lost or added
    assert any(c["filter_call_segment"] is None for c in out)
    assert all(s["step"] == "merge" for s in packer.last_report["coarsen_steps"])


def test_non_identical_assignment_drops_lowest_net(synth_channels):
    # 18 groups (2 arpu x 9 data/call); call splits differ in channel -> no exact merge possible
    prof = grid_profile(["tariff_1"], ["HIGH", "LOW"], DATA_SEGMENTS, CALL_SEGMENTS, per=6)
    dv, sim, packer = build(prof, synth_channels)
    chans = ["push", "sms", "digital_ads"]
    ratios = {("tariff_1", a, "tariff_2"): 0.5 for a in ("HIGH", "LOW")}
    opts = []
    for i, sk in enumerate(sorted(dv.subs)):
        opts.append(make_option(dv, sk, "tariff_2", chans[CALL_SEGMENTS.index(sk[3])], net=1000.0 + 10 * i))
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), FakeModel(ratios, synth_channels), 100_000, 15_000)
    check_valid(out, sim)
    assert len(out) == 10
    steps = packer.last_report["coarsen_steps"]
    assert [s["step"] for s in steps] == ["drop"] * 8
    kept = {(c["filter_arpu_segment"], c["filter_data_segment"], c["filter_call_segment"]) for c in out}
    lowest = sorted(dv.subs)[:8]  # smallest est_net = first 8 by key order
    assert not any(sk[1:] in kept for sk in lowest)


def test_split_over_5000(synth_channels):
    prof = grid_profile(CODES[:4], ["HIGH"], ["HEAVY"], ["HIGH"], per=3000)
    dv, sim, packer = build(prof, synth_channels)
    opts = [make_option(dv, sk, "tariff_9", "push") for sk in dv.subs]
    model = FakeModel({(t, "HIGH", "tariff_9"): 0.05 for t in CODES[:4]}, synth_channels)
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), model, 100_000, 15_000)
    assert packer.last_report["n_groups"] == 1 and packer.last_report["n_after_split"] == 4
    seen = check_valid(out, sim)
    assert len(out) == 4 and len(seen) == 12_000


def test_prune_negative_marginal_and_order(synth_channels):
    prof = grid_profile(CODES[:3], ["MID"], ["LITE"], ["LOW", "HIGH"], per=20)
    dv, sim, packer = build(prof, synth_channels)
    ratios = {("tariff_1", "MID", "tariff_5"): 0.3, ("tariff_2", "MID", "tariff_5"): 0.3,
              ("tariff_3", "MID", "tariff_6"): -0.2}
    opts = [
        make_option(dv, ("tariff_1", "MID", "LITE", "LOW"), "tariff_5", "sms", net=500.0),
        make_option(dv, ("tariff_2", "MID", "LITE", "HIGH"), "tariff_5", "call", net=5000.0),
        make_option(dv, ("tariff_3", "MID", "LITE", "LOW"), "tariff_6", "sms", net=50.0),  # truly negative
    ]
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), FakeModel(ratios, synth_channels), 100_000, 15_000)
    check_valid(out, sim)
    assert [c["target_tariff"] for c in out] == ["tariff_5", "tariff_5"]
    assert [c["channel"] for c in out] == ["call", "sms"]  # 5000/20 > 500/20 per contact
    assert [c["campaign_name"][:3] for c in out] == ["c01", "c02"]
    assert packer.last_report["pruned"] and packer.last_sim is not None and packer.last_sim.net > 0


def test_empty_plan_returns_empty(synth_channels):
    prof = grid_profile(CODES[:2], ["LOW"], ["LITE"], ["LOW"], per=3)
    _, _, packer = build(prof, synth_channels)
    assert packer.pack(Plan([], 0, 0, 0, 0, 0), FakeModel({}, synth_channels), 100_000, 15_000) == []


def test_empty_campaign_matches_nobody(synth_channels, synth_profile, synth_tariffs):
    codes = list(synth_tariffs["tariff_plan_code"])
    dv, sim, packer = build(synth_profile, synth_channels, codes=codes)
    c = packer.empty_campaign()
    assert len(sim.segment(c)) == 0
    assert c["channel"] == "push" and c["target_tariff"] not in c["filter_current_tariff"].split(";")
    validate_strategy(pd.DataFrame([c]), dict_tariff(codes))


def test_realistic_profile_random_plan(repo_root, synth_channels):
    """Real customer profile, seeded random assignment over all subs: constraints hold, deterministic."""
    prof = pd.read_csv(repo_root / "customer_profile.csv", usecols=[
        "ID_NUMBER", "current_tariff", "arpu_segment", "data_segment", "call_segment", "predicted_arpu"])
    codes = sorted(set(prof["current_tariff"].dropna()), key=lambda s: int(s.split("_")[1]))
    dv, sim, packer = build(prof, synth_channels, codes=codes)
    rng = np.random.default_rng(3)
    chans = ["push", "sms"]
    opts, ratios = [], {}
    for sk in sorted(dv.subs):
        if rng.random() < 0.3:
            continue
        tgt = codes[int(rng.integers(0, 3))]
        if tgt == sk[0]:
            tgt = codes[3]
        opts.append(make_option(dv, sk, tgt, chans[int(rng.integers(0, 2))], net=float(rng.normal(500, 300))))
        ratios[(sk[0], sk[1], tgt)] = 0.05
    plan = Plan(opts, 0, 0, 0, 0, 0)
    model = FakeModel(ratios, synth_channels)
    out = packer.pack(plan, model, 100_000, 15_000)
    check_valid(out, sim, codes=codes)
    assert out == packer.pack(plan, model, 100_000, 15_000)
    empty = packer.empty_campaign()
    assert len(sim.segment(empty)) == 0
    validate_strategy(pd.DataFrame([empty]), dict_tariff(codes))


def test_last_sim_reset_on_empty_plan(synth_channels):
    prof = grid_profile(CODES[:2], ["LOW"], ["LITE"], ["LOW"], per=5)
    dv, _, packer = build(prof, synth_channels)
    model = FakeModel({("tariff_1", "LOW", "tariff_3"): 0.5}, synth_channels)
    out = packer.pack(Plan([make_option(dv, ("tariff_1", "LOW", "LITE", "LOW"), "tariff_3", "sms")], 0, 0, 0, 0, 0),
                      model, 100_000, 15_000)
    assert out and packer.last_sim is not None
    assert packer.pack(Plan([], 0, 0, 0, 0, 0), model, 100_000, 15_000) == []
    assert packer.last_sim is None and packer.last_report["sim_net"] is None


def test_nan_net_and_zero_budget(synth_channels):
    prof = grid_profile(CODES[:2], ["MID"], ["LITE"], ["LOW", "HIGH"], per=5)
    dv, sim, packer = build(prof, synth_channels)
    opts = [make_option(dv, ("tariff_1", "MID", "LITE", "LOW"), "tariff_3", "sms", net=float("nan")),
            make_option(dv, ("tariff_2", "MID", "LITE", "HIGH"), "tariff_3", "push", net=200.0)]
    model = FakeModel({(t, "MID", "tariff_3"): 0.4 for t in CODES[:2]}, synth_channels)
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), model, 100_000, 15_000)
    check_valid(out, sim)
    assert out == packer.pack(Plan(opts, 0, 0, 0, 0, 0), model, 100_000, 15_000)
    assert packer.pack(Plan(opts, 0, 0, 0, 0, 0), model, 0.0, 0) == []


def test_single_sub_over_5000_kept_whole(synth_channels):
    prof = grid_profile(["tariff_1"], ["HIGH"], ["HEAVY"], ["HIGH"], per=6000)
    dv, sim, packer = build(prof, synth_channels)
    opts = [make_option(dv, ("tariff_1", "HIGH", "HEAVY", "HIGH"), "tariff_2", "push")]
    model = FakeModel({("tariff_1", "HIGH", "tariff_2"): 0.05}, synth_channels)
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), model, 100_000, 15_000)
    assert len(out) == 1 and len(check_valid(out, sim)) == 6000


def test_config_limits_clamped_to_organizer_caps(synth_channels):
    prof = grid_profile(CODES[:2], ["HIGH"], ["HEAVY"], ["HIGH"], per=3000)
    dv, sim, packer = build(prof, synth_channels, max_per_campaign=9000, max_campaigns=50)
    opts = [make_option(dv, sk, "tariff_9", "push") for sk in dv.subs]
    model = FakeModel({(t, "HIGH", "tariff_9"): 0.05 for t in CODES[:2]}, synth_channels)
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), model, 100_000, 15_000)
    assert len(out) == 2 and len(check_valid(out, sim)) == 6000


def test_no_merge_when_nan_segment_rows_would_leak(synth_channels):
    # tariff_1/HIGH/HEAVY fully assigned, plus rows with NaN call: dropping the call filter would add them
    prof = grid_profile(["tariff_1"], ["HIGH"], ["HEAVY"], CALL_SEGMENTS, per=3)
    extra = prof.iloc[:4].copy()
    extra["call_segment"] = None
    extra["ID_NUMBER"] = np.arange(1000, 1004)
    prof = pd.concat([prof, extra], ignore_index=True)
    dv, sim, packer = build(prof, synth_channels, max_campaigns=1)
    opts = [make_option(dv, sk, "tariff_2", "sms", net=100.0 + i) for i, sk in enumerate(sorted(dv.subs))]
    model = FakeModel({("tariff_1", "HIGH", "tariff_2"): 0.5}, synth_channels)
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), model, 100_000, 15_000)
    seen = check_valid(out, sim, max_c=1)
    assert not seen & {1000, 1001, 1002, 1003}
    assert all(st["step"] == "drop" for st in packer.last_report["coarsen_steps"])


def test_real_simulator_matches_organizer_filters(repo_root, synth_channels):
    """Production DataView + ScoreSimulator: organizer filters select the same, disjoint IDs."""
    from agent_src.m10_dataview import DataView
    from agent_src.m40_simulator import ScoreSimulator
    from scoring_core import apply_filters

    prof = pd.read_csv(repo_root / "customer_profile.csv")
    tariffs = pd.read_csv(repo_root / "tariff_dictionary.csv")
    dv = DataView(prof, tariffs, synth_channels)
    sim = ScoreSimulator(dv, prof)
    packer = CampaignPacker(Config(), dv, sim, log=RunLog())
    codes = dv.tariff_codes
    opts, ratios = [], {}
    for sk in sorted(dv.subs):
        if sk[1] != "HIGH":
            continue
        tgt = codes[0] if sk[0] != codes[0] else codes[1]
        opts.append(make_option(dv, sk, tgt, "sms" if sk[2] == "HEAVY" else "push"))
        ratios[(sk[0], sk[1], tgt)] = 0.03
    out = packer.pack(Plan(opts, 0, 0, 0, 0, 0), FakeModel(ratios, synth_channels), 200_000, 15_000)
    assert 1 <= len(out) <= 10
    validate_strategy(pd.DataFrame(out), tariffs)
    names = [pc["name"] for pc in packer.last_sim.per_campaign]
    assert names == [c["campaign_name"] for c in out]
    ids = [set(apply_filters(prof, pd.Series(c))["ID_NUMBER"]) for c in out]
    assert ids == [set(sim.segment(c)["ID_NUMBER"]) for c in out]  # simulator selects what the organizer selects
    assigned = {int(i) for o in opts for i in dv.subs[tuple(o.sub)].ids.tolist()}
    for i in range(len(ids)):
        assert ids[i] <= assigned, "campaign reaches customers outside the plan (NaN-segment leak)"
        for j in range(i + 1, len(ids)):
            assert not ids[i] & ids[j]
    contacts = sum(pc["n_contacted"] for pc in packer.last_sim.per_campaign)
    assert contacts == packer.last_sim.contacts <= 15_000
