"""Tests for agent_src.m90_orchestrator (fakes for every other module; real mock env if available)."""
from __future__ import annotations

import asyncio
import time

import pytest

from agent_src.contract import CAMPAIGN_KEYS, Config, Observation
from agent_src.m90_orchestrator import (
    Agent,
    Orchestrator,
    _orc_campaign_matches,
    _orc_emergency_campaigns,
    run_coro,
)
from tests import fakes_orchestrator as F


@pytest.fixture(autouse=True)
def _reset_fakes(monkeypatch, tmp_path):
    F.FakeAllocator.calls = []
    F.FakeReporter.written = []
    F.FakeLLM.veto = []
    F.FakeLLM.applied = True
    F.FakePlanner.max_specs = 3
    F.FakePlanner.delay_s = 0.0
    monkeypatch.delenv("AGENT_LLM_MODE", raising=False)
    yield


def _cfg(tmp_path, **kw) -> Config:
    base = dict(time_budget_s=30.0, report_path=str(tmp_path / "report.md"))
    base.update(kw)
    return Config(**base)


def _valid(out) -> bool:
    """1..10 campaign dicts with exactly the CAMPAIGN_KEYS schema."""
    return (isinstance(out, list) and 1 <= len(out) <= 10
            and all(isinstance(c, dict) and tuple(c.keys()) == CAMPAIGN_KEYS for c in out))


# --------------------------------------------------------------------------- run_coro


async def _plus_one(x):
    await asyncio.sleep(0)
    return x + 1


async def _raise():
    raise KeyError("x")


def test_run_coro_without_loop():
    assert run_coro(_plus_one(1)) == 2


def test_run_coro_inside_running_loop():
    async def main():
        return run_coro(_plus_one(41))

    assert asyncio.run(main()) == 42


def test_run_coro_propagates_errors_from_thread():
    async def main():
        return run_coro(_raise())

    with pytest.raises(KeyError):
        asyncio.run(main())


# --------------------------------------------------------------------------- happy path with fakes


def test_full_pipeline_with_fakes(tmp_path):
    env = F.FakeEnv()
    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="decide")
    out = run_coro(orch.run(env))

    assert _valid(out) and len(out) == 2
    assert all(tuple(c.keys()) == CAMPAIGN_KEYS for c in out)
    # pilots: sequential, observations use actual n/cost/y from result
    obs = orch.model.observations
    assert len(obs) == 3 and len(env.calls) == 3
    assert all(isinstance(o, Observation) for o in obs)
    assert [o.n for o in obs] == [2, 2, 2]  # actual n, not the requested 60
    assert [o.pilot_index for o in obs] == [0, 1, 2]
    assert all(o.y == pytest.approx(0.25) and o.cost == 8.0 for o in obs)
    assert len(orch.planner.registered) == 3
    assert orch.explore_money_spent == pytest.approx(24.0)
    assert orch.explore_reach_spent == 6
    assert sum(orch.used_subs.values()) == 6
    # hypothesis weights forwarded to the planner; batches keyed by arpu segment
    assert orch.planner.weights and all(v == 1.2 for v in orch.planner.weights.values())
    assert set(orch.llm.batches) == {"HIGH", "MID"}
    assert all("arm_id" in it and "cell_sum_p" in it for b in orch.llm.batches.values() for it in b)
    # allocation tried both pilot-overlap variants
    assert any(c["extra"] for c in F.FakeAllocator.calls)
    assert any(c["exclude"] for c in F.FakeAllocator.calls)
    # report written with stage info
    assert F.FakeReporter.written and F.FakeReporter.written[-1]["path"] == str(tmp_path / "report.md")
    assert orch.stages["allocate"] == "ok" and orch.stages["finalize"] == "ok"
    assert len(orch.log.pilots) == 3


def test_agent_act_inside_running_event_loop(tmp_path):
    agent = Agent(cfg=_cfg(tmp_path), components=F.fake_components(), llm_mode="off")

    async def main():
        return agent.act(F.FakeEnv())

    out = asyncio.run(main())
    assert _valid(out) and len(out) == 2


# --------------------------------------------------------------------------- review / veto


def test_veto_applied_in_decide_mode(tmp_path):
    F.FakeLLM.veto = ["c01_tariff_3_sms_HIGH"]
    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="decide")
    out = run_coro(orch.run(F.FakeEnv()))
    assert len(out) == 1
    assert out[0]["filter_arpu_segment"] == "MID"
    # simulated net minus the pilot-overlap over-count on the kept MID sub (2 customers · 8500 · 0.2)
    assert orch.llm.what_if_result["net"] == pytest.approx(99992.0 - 3400.0)
    assert orch.extra["review"]["veto"] == ["c01_tariff_3_sms_HIGH"]


def test_veto_ignored_in_advise_mode(tmp_path):
    F.FakeLLM.veto = ["c01_tariff_3_sms_HIGH"]
    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="advise")
    out = run_coro(orch.run(F.FakeEnv()))
    assert len(out) == 2


def test_veto_of_everything_is_ignored(tmp_path):
    F.FakeLLM.veto = ["c01_tariff_3_sms_HIGH", "c02_tariff_3_sms_MID"]
    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="decide")
    out = run_coro(orch.run(F.FakeEnv()))
    assert len(out) == 2


# --------------------------------------------------------------------------- failures


def test_pilot_errors_are_registered_and_bounded(tmp_path):
    env = F.FakeEnv(fail_pilots=True)
    F.FakePlanner.max_specs = 50
    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="off")
    out = run_coro(orch.run(env))
    assert _valid(out) and out
    assert len(env.calls) == 3  # stops after 3 consecutive failures
    assert all("error" in r for _, r in orch.planner.registered)
    assert orch.model.observations == []


def _raiser(*a, **k):
    raise F.Boom("sabotage")


class _BoomClass:
    def __init__(self, *a, **k):
        raise F.Boom("ctor")


class _BoomPlanner(F.FakePlanner):
    def next_pilot(self, state):
        raise F.Boom("planner")


class _BoomModel(F.FakeArmModel):
    def add(self, obs):
        raise F.Boom("add")


class _BoomLLM(F.FakeLLM):
    async def assess_hypotheses(self, batches, tariffs):
        raise F.Boom("llm")

    async def review_plan(self, campaigns, evidence, what_if):
        raise F.Boom("llm")


class _BoomReporter(F.FakeReporter):
    def write(self, log, campaigns, extra):
        raise OSError("disk")


@pytest.mark.parametrize(
    "name,repl",
    [
        ("DataView", _BoomClass),
        ("load_history", _raiser),
        ("load_tariff_descriptions", _raiser),
        ("PriorBuilder", _BoomClass),
        ("ArmModel", _BoomClass),
        ("ArmModel", _BoomModel),
        ("ScoreSimulator", _BoomClass),
        ("Allocator", _BoomClass),
        ("CampaignPacker", _BoomClass),
        ("PilotPlanner", _BoomClass),
        ("PilotPlanner", _BoomPlanner),
        ("LLMLayer", _BoomClass),
        ("LLMLayer", _BoomLLM),
        ("Guardrails", _BoomClass),
        ("Reporter", _BoomReporter),
    ],
)
def test_any_stage_failure_still_returns_valid_output(tmp_path, name, repl):
    env = F.FakeEnv()
    agent = Agent(cfg=_cfg(tmp_path), components=F.fake_components(**{name: repl}), llm_mode="decide")
    out = agent.act(env)
    assert _valid(out)
    assert out, f"expected a (fallback) campaign when {name} fails"
    for c in out:
        assert c.get("target_tariff") in {"tariff_1", "tariff_2", "tariff_3"}
        assert c.get("channel") in env.channels
    assert any(v.startswith("error") for v in agent.last_orchestrator.stages.values())


def test_missing_components_fall_back_to_emergency(tmp_path, monkeypatch):
    """Nothing resolvable at all (no modules) -> emergency campaign matching nobody."""
    env = F.FakeEnv()
    comps = {k: None for k in F.fake_components()}

    class _Orc(Orchestrator):
        def _c(self, name):
            raise NameError(name)

    monkeypatch.setattr("agent_src.m90_orchestrator.Orchestrator", _Orc)
    import agent_src.m90_orchestrator as m

    out = m.Agent(cfg=_cfg(tmp_path), components=comps, llm_mode="off").act(env)
    assert len(out) == 1
    c = out[0]
    prof = env.customer_profile
    sel = prof[(prof.current_tariff == c["filter_current_tariff"]) & (prof.arpu_segment == c["filter_arpu_segment"])
               & (prof.data_segment == c["filter_data_segment"]) & (prof.call_segment == c["filter_call_segment"])]
    assert len(sel) == 0 and c["target_tariff"] != c["filter_current_tariff"]


def test_run_crash_in_act_uses_guardrails_fallback(tmp_path, monkeypatch):
    async def _bad_run(self, env):
        raise RuntimeError("pipeline crash")

    monkeypatch.setattr(Orchestrator, "run", _bad_run)
    out = Agent(cfg=_cfg(tmp_path), components=F.fake_components(), llm_mode="off").act(F.FakeEnv())
    assert out == [F.FakePacker(None, None, None).empty_campaign()]


def test_global_timeout_proceeds_to_allocate(tmp_path):
    """Slow planner + tiny time budget -> learning stage times out, allocate still produces a plan."""
    F.FakePlanner.max_specs = 1000
    F.FakePlanner.delay_s = 0.2
    cfg = _cfg(tmp_path, time_budget_s=2.0)
    orch = Orchestrator(cfg, components=F.fake_components(), llm_mode="off")
    t0 = time.monotonic()
    out = run_coro(orch.run(F.FakeEnv()))
    assert time.monotonic() - t0 < 10.0
    assert len(out) == 2
    assert orch.stages["allocate"] == "ok"
    assert len(orch.log.pilots) < 20


def test_slow_llm_times_out_to_neutral(tmp_path):
    class _SlowLLM(F.FakeLLM):
        async def assess_hypotheses(self, batches, tariffs):
            await asyncio.sleep(30)
            return {}

    cfg = _cfg(tmp_path, llm_budget_s=0.3)
    orch = Orchestrator(cfg, components=F.fake_components(LLMLayer=_SlowLLM), llm_mode="decide")
    out = run_coro(orch.run(F.FakeEnv()))
    assert len(out) == 2
    assert orch.planner.weights is None  # neutral: never set
    assert orch.stages["hypotheses"].startswith("error")
    assert len(orch.model.observations) == 3


# --------------------------------------------------------------------------- helpers


def test_campaign_matches_semantics():
    sub = ("tariff_1", "HIGH", "LITE", "LOW")
    assert _orc_campaign_matches({"filter_current_tariff": "tariff_2; tariff_1", "filter_arpu_segment": "HIGH"}, sub)
    assert not _orc_campaign_matches({"filter_current_tariff": "tariff_2"}, sub)
    assert not _orc_campaign_matches({"filter_data_segment": "HEAVY"}, sub)
    assert _orc_campaign_matches({"filter_data_segment": float("nan"), "filter_call_segment": None}, sub)


def test_emergency_campaigns_handles_garbage_env():
    class _E:
        tariffs = None
        channels = {}

    assert _orc_emergency_campaigns(_E()) == []


# --------------------------------------------------------------------------- real modules on the mock env


def _real_components_or_skip(planner_fake: bool):
    try:
        from agent_src.m10_dataview import DataView, load_history, load_tariff_descriptions
        from agent_src.m20_prior import PriorBuilder
        from agent_src.m30_arm_model import ArmModel
        from agent_src.m40_simulator import ScoreSimulator
        from agent_src.m50_allocator import Allocator
        from agent_src.m60_packer import CampaignPacker
        from agent_src.m80_llm import LLMLayer
        from agent_src.m85_guardrails import Guardrails, Reporter
    except ImportError as exc:
        pytest.skip(f"real modules missing: {exc}")
    comps = dict(DataView=DataView, load_history=load_history, load_tariff_descriptions=load_tariff_descriptions,
                 PriorBuilder=PriorBuilder, ArmModel=ArmModel, ScoreSimulator=ScoreSimulator, Allocator=Allocator,
                 CampaignPacker=CampaignPacker, LLMLayer=LLMLayer, Guardrails=Guardrails, Reporter=Reporter)
    if planner_fake:
        comps["PilotPlanner"] = _RealArmPlanner
    else:
        try:
            from agent_src.m70_planner import PilotPlanner
        except ImportError as exc:
            pytest.skip(f"planner missing: {exc}")
        comps["PilotPlanner"] = PilotPlanner
    return comps


class _RealArmPlanner:
    """Planner stand-in for real-module runs: 2 pilots on the largest sub-cells."""

    def __init__(self, cfg, dv, model, allocator):
        self.dv = dv
        self.done = 0

    def set_weights(self, w):
        pass

    def next_pilot(self, state):
        from agent_src.contract import PilotSpec

        if self.done >= 2:
            return None
        subs = sorted(self.dv.subs.values(), key=lambda s: (-s.n, s.key))
        sub = subs[self.done].key
        target = self.dv.targets_for(sub[:2])[0]
        self.done += 1
        return PilotSpec(arm=(sub[0], sub[1], target), channel="sms", n=100, sub=sub,
                         filters={"filter_current_tariff": sub[0], "filter_arpu_segment": sub[1],
                                  "filter_data_segment": sub[2], "filter_call_segment": sub[3]},
                         score=0.0, reason="test")

    def register_result(self, spec, result):
        pass


@pytest.mark.parametrize("planner_fake", [True, False])
def test_real_modules_on_mock_env(tmp_path, mock_env_factory, planner_fake):
    comps = _real_components_or_skip(planner_fake)
    env = mock_env_factory(0)
    agent = Agent(cfg=_cfg(tmp_path, time_budget_s=120.0), components=comps, llm_mode="off")
    out = agent.act(env)
    assert _valid(out) and 1 <= len(out) <= 10
    from scoring_core import sanitize_campaigns

    assert len(sanitize_campaigns(out, env.tariffs)) == len(out)
    assert len(env.pilot_history) >= 1
    orch = agent.last_orchestrator
    assert orch.stages.get("finalize") == "ok", orch.stages


def test_pilot_overlap_gain_counts_covered_pilot_subs(tmp_path):
    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="off")
    orch.dv = F.FakeDataView(None, None, None)
    orch.model = F.FakeArmModel(None, None, None)
    arm = (F.SUBS[0][0], F.SUBS[0][1], "tariff_3")
    orch.used_subs = {F.SUBS[0]: 2}
    orch.pilot_records = {F.SUBS[0]: [(arm, "sms", 2)]}
    camp = {"filter_current_tariff": "tariff_1", "filter_arpu_segment": "HIGH",
            "target_tariff": "tariff_3", "channel": "sms"}
    assert orch._pilot_overlap_gain([camp]) == pytest.approx(2 * 8500.0 * 0.2)
    other = dict(camp, filter_arpu_segment="LOW")
    assert orch._pilot_overlap_gain([other]) == 0.0


class _ArmRatioModel(F.FakeArmModel):
    """Posterior mean depends on target: tariff_2 -> 0.05, else 0.2."""

    def posterior(self, arm, channel):
        from agent_src.contract import Posterior

        return Posterior(mean=0.05 if arm[2] == "tariff_2" else 0.2, sd=0.05)


def test_pilot_overlap_uses_min_of_pilot_and_final_lift(tmp_path):
    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="off")
    orch.dv = F.FakeDataView(None, None, None)
    orch.model = _ArmRatioModel(None, None, None)
    sub = F.SUBS[0]
    orch.used_subs = {sub: 1}
    orch.pilot_records = {sub: [((sub[0], sub[1], "tariff_2"), "sms", 1)]}  # weak pilot arm, 1 of 2 customers
    camp = {"filter_current_tariff": "tariff_1", "filter_arpu_segment": "HIGH",
            "target_tariff": "tariff_3", "channel": "sms"}
    # over-count = covered(1) · mean_p · min(pilot 0.05, final 0.2)
    assert orch._pilot_overlap_gain([camp]) == pytest.approx(1 * 8500.0 * 0.05)


def test_repeated_pilots_in_same_sub_overlap_in_expectation(tmp_path):
    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="off")
    orch.dv = F.FakeDataView(None, None, None)
    orch.model = F.FakeArmModel(None, None, None)
    sub = F.SUBS[0]
    arm = (sub[0], sub[1], "tariff_3")
    orch.used_subs = {sub: 2}
    orch.pilot_records = {sub: [(arm, "sms", 1), (arm, "sms", 1)]}  # two random draws of 1 from 2
    cov = orch._pilot_coverage()
    assert cov[sub][0] == pytest.approx(2 * (1 - 0.5 * 0.5))


class _NanEnv(F.FakeEnv):
    def run_pilot(self, *a, **k):
        res = super().run_pilot(*a, **k)
        res["observed_lift_ratio"] = float("nan")
        return res


def test_non_finite_pilot_result_is_not_fed_to_model_but_spend_is_booked(tmp_path):
    env = _NanEnv()
    F.FakePlanner.max_specs = 2
    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="off")
    out = run_coro(orch.run(env))
    assert _valid(out)
    assert orch.model.observations == []
    assert len(env.calls) == 2
    assert orch.explore_money_spent == pytest.approx(16.0) and orch.explore_reach_spent == 4
    assert all("error" in r for _, r in orch.planner.registered)


def test_string_weight_keys_are_parsed(tmp_path):
    class _StrLLM(F.FakeLLM):
        async def assess_hypotheses(self, batches, tariffs):
            return {it["arm_id"]: 0.8 for b in batches.values() for it in b}

    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(LLMLayer=_StrLLM), llm_mode="decide")
    run_coro(orch.run(F.FakeEnv()))
    assert orch.planner.weights and all(isinstance(k, tuple) and len(k) == 3 for k in orch.planner.weights)


def test_variant_choice_prefers_within_limits(tmp_path):
    class _CapSim(F.FakeSimulator):
        def simulate(self, campaigns, ratio_fn, budget, contacts):
            r = super().simulate(campaigns, ratio_fn, budget, contacts)
            if len(campaigns) == 2:  # the richer "overlap_cost" plan is capped
                from dataclasses import replace

                return replace(r, within_limits=False)
            return r

    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(ScoreSimulator=_CapSim), llm_mode="off")
    run_coro(orch.run(F.FakeEnv()))
    assert orch.extra["allocate"]["chosen"] == "exclude"


def test_campaign_matches_treats_pd_na_as_missing():
    import pandas as pd

    assert _orc_campaign_matches({"filter_data_segment": pd.NA, "filter_current_tariff": "tariff_1"},
                                 ("tariff_1", "HIGH", "LITE", "LOW"))


def test_evidence_sd_treats_shared_posterior_as_correlated(tmp_path):
    from agent_src.contract import Option, Plan

    orch = Orchestrator(_cfg(tmp_path), components=F.fake_components(), llm_mode="off")
    orch.dv = F.FakeDataView(None, None, None)
    orch.model = F.FakeArmModel(None, None, None)
    orch.sim = F.FakeSimulator(orch.dv, None)
    s1 = ("tariff_1", "HIGH", "LITE", "LOW")
    s2 = ("tariff_1", "HIGH", "HEAVY", "LOW")
    opts = [Option(sub=s, target="tariff_3", channel="sms", n=2, cost=8.0, net_mean=100.0, net_sd=10.0,
                   net_lcb=90.0, p_pos=0.99) for s in (s1, s2)]
    orch.plan = Plan(options=opts, lambda_money=0.0, lambda_reach=0.0, total_net_lcb=180.0, total_cost=16.0,
                     total_contacts=4)
    camp = {"campaign_name": "c01", "filter_current_tariff": "tariff_1", "filter_arpu_segment": "HIGH",
            "target_tariff": "tariff_3", "channel": "sms"}
    ev = orch._evidence([camp], 1e5, 15000)
    assert ev[0]["net_mean"] == pytest.approx(200.0)
    assert ev[0]["net_sd"] == pytest.approx(20.0)  # not sqrt(200)
