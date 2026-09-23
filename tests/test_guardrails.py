"""Tests for agent_src.m85_guardrails (Guardrails, Reporter)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_src.contract import CAMPAIGN_KEYS, Config, RunLog
from agent_src.m85_guardrails import Guardrails, Reporter
from tests.fakes_guardrails import FakeDV, FakeSim

import scoring_core


@pytest.fixture()
def world(synth_profile, synth_tariffs, synth_channels):
    dv = FakeDV(synth_profile, synth_tariffs, synth_channels)
    sim = FakeSim(dv, synth_profile, synth_channels)
    return dv, sim


@pytest.fixture()
def gr(world):
    dv, sim = world
    return Guardrails(Config(), dv, sim, log=RunLog())


def _c(**kw):
    base = dict(campaign_name="c", filter_arpu_segment="MID", filter_data_segment=None, filter_call_segment=None,
                filter_current_tariff="tariff_1", target_tariff="tariff_2", channel="sms")
    base.update(kw)
    return base


def _assert_scoring_ok(out: list[dict], tariffs: pd.DataFrame) -> None:
    assert 1 <= len(out) <= 10
    assert all(list(c.keys()) == list(CAMPAIGN_KEYS) for c in out)
    assert scoring_core.sanitize_campaigns(out, tariffs) == out
    scoring_core.validate_strategy(pd.DataFrame(out), tariffs)
    names = [c["campaign_name"] for c in out]
    assert len(set(names)) == len(names)
    for c in out:
        frm = c["filter_current_tariff"].split(";")
        assert c["target_tariff"] not in frm and all(frm)


def test_valid_passthrough_and_order(gr, synth_tariffs):
    cs = [_c(campaign_name="a"), _c(campaign_name="b", filter_arpu_segment="LOW", channel="push")]
    out = gr.validate(cs, 100_000, 15_000)
    assert [c["campaign_name"] for c in out] == ["a", "b"]
    assert out[0]["filter_current_tariff"] == "tariff_1"
    _assert_scoring_ok(out, synth_tariffs)


def test_coercion_strip_case_nan(gr, synth_tariffs):
    raw = {"campaign_name": "  x ", "filter_arpu_segment": " mid ", "filter_data_segment": float("nan"),
           "filter_call_segment": "", "filter_current_tariff": " tariff_1 ; TARIFF_3 ;; ",
           "target_tariff": " Tariff_2", "channel": " SMS ", "junk_key": 123}
    out = gr.validate([raw], 100_000, 15_000)
    assert out == [{"campaign_name": "x", "filter_arpu_segment": "MID", "filter_data_segment": None,
                    "filter_call_segment": None, "filter_current_tariff": "tariff_1;tariff_3",
                    "target_tariff": "tariff_2", "channel": "sms"}]
    _assert_scoring_ok(out, synth_tariffs)


def test_drops_unknown_values(gr):
    cs = [_c(campaign_name="bad_t", target_tariff="tariff_99"),
          _c(campaign_name="bad_ch", channel="telegram"),
          _c(campaign_name="bad_seg", filter_arpu_segment="ULTRA"),
          _c(campaign_name="bad_call", filter_call_segment="MID"),
          _c(campaign_name="ok")]
    out = gr.validate(cs, 100_000, 15_000)
    assert [c["campaign_name"] for c in out] == ["ok"]
    assert any("dropped_unknown_target" in s for s in gr.issues)
    assert any("dropped_unknown_channel" in s for s in gr.issues)
    assert any("dropped_unknown_segment" in s for s in gr.issues)


def test_from_list_handling(gr):
    out = gr.validate([_c(campaign_name="a", filter_current_tariff="tariff_2;tariff_1;tariff_zzz")], 1e5, 15000)
    assert out[0]["filter_current_tariff"] == "tariff_1"
    # only the target -> empty from-list -> dropped -> fallback
    out = gr.validate([_c(campaign_name="b", filter_current_tariff="tariff_2")], 1e5, 15000)
    assert out[0]["campaign_name"] == "g00_noop_push"
    # list input accepted
    out = gr.validate([_c(campaign_name="c", filter_current_tariff=["tariff_3", "tariff_1"])], 1e5, 15000)
    assert out[0]["filter_current_tariff"] == "tariff_1;tariff_3"
    # missing from-list -> all tariffs except target
    out = gr.validate([_c(campaign_name="d", filter_current_tariff=None)], 1e5, 15000)
    assert out[0]["filter_current_tariff"] == "tariff_1;tariff_3;tariff_4"


def test_dedupe_filters_and_names(gr):
    cs = [_c(campaign_name="a", filter_current_tariff="tariff_1;tariff_3"),
          _c(campaign_name="b", filter_current_tariff="tariff_3;tariff_1", channel="push"),  # same filters
          _c(campaign_name="a", filter_arpu_segment="LOW"),
          _c(campaign_name=None, filter_arpu_segment="HIGH")]
    out = gr.validate(cs, 1e5, 15000)
    names = [c["campaign_name"] for c in out]
    assert names[0] == "a" and names[1] == "a_2" and len(out) == 3
    assert names[2].startswith("g04_")


def test_truncate_to_max(world, synth_tariffs):
    dv, sim = world
    cs = []
    for t in ("tariff_1", "tariff_3", "tariff_4"):
        for a in ("LOW", "MID", "HIGH"):
            for d in ("NON_USER", "LITE"):
                cs.append(_c(campaign_name=f"{t}{a}{d}", filter_current_tariff=t, filter_arpu_segment=a,
                             filter_data_segment=d, channel="push"))
    g = Guardrails(Config(), dv, sim)
    out = g.validate(cs, 1e5, 15000)
    assert len(out) == 10 and [c["campaign_name"] for c in out] == [c["campaign_name"] for c in cs[:10]]
    out = Guardrails(Config(max_campaigns=3), dv, sim).validate(cs, 1e5, 15000)
    assert len(out) == 3
    _assert_scoring_ok(out, synth_tariffs)


def test_limits_trim_capped(world):
    dv, sim = world
    cs = [_c(campaign_name="push_all", filter_arpu_segment=None, filter_current_tariff="tariff_1;tariff_3",
             channel="push"),
          _c(campaign_name="call_mid", filter_arpu_segment="MID", channel="call"),
          _c(campaign_name="push_tail", filter_arpu_segment="LOW", filter_current_tariff="tariff_3",
             target_tariff="tariff_4", channel="push")]
    n_call = len(sim.segment(dict(zip(CAMPAIGN_KEYS, [None] * 7)) | {k: v for k, v in cs[1].items()}))
    assert n_call > 1
    g = Guardrails(Config(), dv, sim)
    # money allows fewer call contacts than the segment -> call campaign is money-capped and dropped
    out = g.validate(cs, budget=160 * (n_call - 1), contacts=15000)
    assert [c["campaign_name"] for c in out] == ["push_all", "push_tail"]
    assert any("trimmed_over_limits" in s for s in g.issues)
    res = sim.simulate(out, lambda *a: 0.0, 160 * (n_call - 1), 15000)
    assert res.within_limits
    # reach: nothing fits -> keep a single campaign
    out = g.validate(cs, budget=1e5, contacts=0)
    assert len(out) == 1


def test_simulator_error_is_tolerated(synth_profile, synth_tariffs, synth_channels):
    dv = FakeDV(synth_profile, synth_tariffs, synth_channels)
    g = Guardrails(Config(), dv, FakeSim(dv, synth_profile, synth_channels, raise_error=True))
    out = g.validate([_c(campaign_name="a")], 1e5, 15000)
    assert [c["campaign_name"] for c in out] == ["a"]
    assert any("simulate_error" in s for s in g.issues)
    assert Guardrails(Config(), dv, None).validate([_c()], 1e5, 15000)[0]["campaign_name"] == "c"


@pytest.mark.parametrize("garbage", [
    None, 42, "campaign", {"target_tariff": "tariff_2", "channel": "sms"}, [None, 1, "x", [], {}],
    [{"target_tariff": float("nan"), "channel": None}], [{"target_tariff": ["tariff_2"], "channel": {"a": 1}}],
    [{"campaign_name": object(), "target_tariff": "tariff_2", "channel": "push", "filter_arpu_segment": 3.5}],
    [{"campaign_name": 5, "target_tariff": "tariff_2", "channel": "push", "filter_current_tariff": 7}],
    pd.DataFrame([{"target_tariff": "tariff_3", "channel": "push", "filter_current_tariff": "tariff_1"}]),
    iter([{"target_tariff": "tariff_4", "channel": "push"}]),
])
def test_garbage_never_raises(gr, synth_tariffs, garbage):
    out = gr.validate(garbage, 1e5, 15000)
    _assert_scoring_ok(out, synth_tariffs)


def test_empty_campaign_matches_nobody(world, synth_profile, synth_tariffs):
    dv, sim = world
    g = Guardrails(Config(), dv, sim)
    out = g.validate([], 1e5, 15000)
    assert len(out) == 1 and out[0]["channel"] == "push"
    assert len(sim.segment(out[0])) == 0
    _assert_scoring_ok(out, synth_tariffs)


def test_no_tariffs_returns_empty():
    class DV:
        tariff_codes: list = []
        channels = ["push"]
        subs: dict = {}
    assert Guardrails(Config(), DV(), None).validate([_c()], 1e5, 15000) == []


def test_mock_env_end_to_end(mock_env_factory):
    env = mock_env_factory(0)
    profile = env.customer_profile
    dv = FakeDV(profile, env.tariffs, env.channels)
    sim = FakeSim(dv, profile, env.channels)
    codes = dv.tariff_codes
    cs = [{"campaign_name": f"k{i}", "filter_arpu_segment": a, "filter_current_tariff": f"{codes[i]};{codes[i + 1]}",
           "target_tariff": codes[i + 2], "channel": ch}
          for i, (a, ch) in enumerate([("HIGH", "call"), ("MID", "digital_ads"), ("LOW", "sms"), ("MID", "push")])]
    cs.append({"target_tariff": "nope", "channel": "sms"})
    g = Guardrails(Config(), dv, sim)
    out = g.validate(cs, env.remaining_budget, env.remaining_contacts)
    _assert_scoring_ok(out, env.tariffs)
    res = sim.simulate(out, lambda *a: 0.0, env.remaining_budget, env.remaining_contacts)
    assert res.within_limits
    assert res.cost <= env.remaining_budget and res.contacts <= env.remaining_contacts


def test_deterministic(gr):
    cs = [_c(campaign_name="a"), _c(campaign_name="a", filter_arpu_segment="LOW"), {"bad": 1}]
    assert gr.validate(cs, 1e5, 15000) == gr.validate(cs, 1e5, 15000)


# ------------------------------------------------------------------ Reporter


def _log() -> RunLog:
    log = RunLog()
    log.log("stage", name="research", dur=0.1)
    log.pilots.append({"pilot_index": 0, "arm_id": "tariff_1|MID|tariff_2", "channel": "sms", "n_customers": 100,
                       "cost": 400.0, "y": 0.05})
    log.pilots.append({"pilot_index": 1, "arm": ("tariff_3", "LOW", "tariff_4"), "channel": "push", "n": 0,
                       "y": float("nan"), "error": "RuntimeError: no|customers"})
    log.llm.append({"agent": "review", "source": "cache", "applied": True, "veto": ["c02"], "summary": "ok\nfine"})
    return log


def test_reporter_writes_markdown(tmp_path: Path):
    p = tmp_path / "rep.md"
    campaigns = [_c(campaign_name="c01"), _c(campaign_name="c02", channel="push")]
    extra = {"per_campaign": [{"name": "c01", "n_contacted": 50, "cost": 200.0, "gross": 900.0}],
             "expected_net": 700.0, "net_sd": 350.0, "timings": {"explore": 12.5, "allocate": 0.4},
             "notes": ["hello"], "llm_mode": "off", "expected_contacts": 50}
    Reporter(Config(), str(p)).write(_log(), campaigns, extra)
    text = p.read_text(encoding="utf-8")
    for s in ("# Agent report", "## Pilots", "## Final plan", "## LLM decisions", "## Timings", "## Notes",
              "tariff_1/MID/tariff_2", "c01", "P(net>0): 0.9772", "explore", "RuntimeError: no/customers"):
        assert s in text, s
    half = 1.96 * 0.804 / math.sqrt(100)
    assert f"[{0.05 - half:.4f}".rstrip("0") in text
    assert "700" in text and "| 700 |" in text  # c01 net = gross - cost


def test_reporter_swallows_oserror(tmp_path: Path):
    Reporter(Config(), str(tmp_path / "missing_dir" / "r.md")).write(_log(), [], {})
    Reporter(Config(), str(tmp_path)).write(_log(), [], {})  # path is a directory
    assert not (tmp_path / "missing_dir").exists()


@pytest.mark.parametrize("campaigns,extra", [(None, None), ("x", 5), ([1, None], {"per_campaign": [3, {"x": 1}]}),
                                             ([], {"expected_net": "nan", "timings": {1: np.float64(2.0)}})])
def test_reporter_garbage(tmp_path: Path, campaigns, extra):
    p = tmp_path / "g.md"
    Reporter(Config(), str(p)).write(RunLog(), campaigns, extra)
    assert p.read_text(encoding="utf-8").startswith("# Agent report")


def test_reporter_empty_log_sections(tmp_path: Path):
    text = Reporter(Config(), str(tmp_path / "x.md")).render(RunLog(), [], {})
    assert "No pilots were run." in text and "No LLM calls" in text


# ------------------------------------------------------------------ cross-check additions


@pytest.mark.parametrize("budget,contacts", [(float("nan"), 15000), (1e5, float("nan")), ("x", None),
                                             (-5.0, -1), (float("inf"), float("inf"))])
def test_bad_limits_never_raise(gr, synth_tariffs, budget, contacts):
    out = gr.validate([_c(campaign_name="a"), _c(campaign_name="b", filter_arpu_segment="LOW")], budget, contacts)
    _assert_scoring_ok(out, synth_tariffs)
    assert not any("validate_error" in s for s in gr.issues)


def test_negative_limits_clamped_to_zero(world):
    dv, sim = world
    g = Guardrails(Config(), dv, sim)
    out = g.validate([_c(campaign_name="a", channel="call"), _c(campaign_name="b", filter_arpu_segment="LOW")],
                     -100.0, -3)
    assert len(out) == 1 and out[0]["campaign_name"] == "a"  # all capped -> keep a single campaign


def test_endless_generator_is_bounded(gr, synth_tariffs):
    def gen():
        while True:
            yield {"target_tariff": "zzz", "channel": "push"}
    out = gr.validate(gen(), 1e5, 15000)
    assert any("scan_capped" in s for s in gr.issues)
    _assert_scoring_ok(out, synth_tariffs)


def test_no_channels_returns_empty(synth_profile, synth_tariffs):
    class DV:
        tariff_codes = ["tariff_1", "tariff_2"]
        channels: list = []
        subs: dict = {}
    assert Guardrails(Config(), DV(), None).validate([_c()], 1e5, 15000) == []


def test_overlap_is_logged_not_dropped(gr):
    cs = [_c(campaign_name="broad", filter_arpu_segment=None, filter_current_tariff="tariff_1", channel="push"),
          _c(campaign_name="narrow", filter_arpu_segment="MID", filter_current_tariff="tariff_1;tariff_3")]
    out = gr.validate(cs, 1e5, 15000)
    assert [c["campaign_name"] for c in out] == ["broad", "narrow"]
    assert any(s.startswith("overlap ") and "broad" in s for s in gr.issues)


def test_real_dataview_and_simulator_within_limits(mock_env_factory):
    from agent_src.m10_dataview import DataView
    from agent_src.m40_simulator import ScoreSimulator

    env = mock_env_factory(0)
    dv = DataView(env.customer_profile, env.tariffs, env.channels)
    sim = ScoreSimulator(dv, env.customer_profile)
    codes = dv.tariff_codes
    cs = [{"campaign_name": f"k{i}", "filter_arpu_segment": a, "filter_current_tariff": ";".join(codes[:i + 1]),
           "target_tariff": codes[-1], "channel": ch}
          for i, (a, ch) in enumerate([("HIGH", "call"), ("MID", "call"), ("LOW", "digital_ads"), ("MID", "push")])]
    budget, contacts = 3000.0, 2000
    g = Guardrails(Config(), dv, sim)
    out = g.validate(cs, budget, contacts)
    _assert_scoring_ok(out, env.tariffs)
    res = sim.simulate(out, lambda *a: 0.0, budget, contacts)
    if len(out) > 1:
        assert res.within_limits
    assert res.cost <= budget and res.contacts <= contacts
    empty = g.empty_campaign()
    assert empty is not None and len(sim.segment(empty)) == 0


def test_reporter_finite_fallbacks_and_details(tmp_path: Path):
    log = RunLog()
    log.pilots.append({"index": 0, "arm": "tariff_1|MID|tariff_2", "channel": "sms", "n_req": 100,
                       "n_customers": None, "n": 100, "y": None, "observed_lift_ratio": 0.1, "cost": 400.0})
    extra = {"final_sim": {"net": 1234.5, "cost": 100.0, "within_limits": True},
             "stages": {"explore": "ok"}, "weird": {"o": object()}}
    text = Reporter(Config(), str(tmp_path / "r.md")).render(log, [_c(campaign_name="c01")], extra)
    half = 1.96 * 0.804 / math.sqrt(100)
    assert f"{0.1 - half:.4f}".rstrip("0") in text
    assert "Expected net: 1 234" in text or "Expected net: 1234" in text
    assert "## Run details" in text and "final_sim" in text and "stages" in text
    assert "n_req" in text
