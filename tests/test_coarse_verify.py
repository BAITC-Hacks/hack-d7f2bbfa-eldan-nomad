"""Coarse cell-level packing (m60), prior calibration (m30) and verification pilots (m90)."""
from __future__ import annotations

import math

import pytest

from agent_src.contract import Config, Observation, Prior
from agent_src.m10_dataview import DataView, load_history
from agent_src.m20_prior import PriorBuilder
from agent_src.m30_arm_model import ArmModel
from agent_src.m40_simulator import ScoreSimulator
from agent_src.m60_packer import CampaignPacker


@pytest.fixture(scope="module")
def world(repo_root):
    import os

    old = os.getcwd()
    os.chdir(repo_root)
    try:
        from mock_environment import make_mock_env

        env, _ = make_mock_env(seed=0)
    finally:
        os.chdir(old)
    cfg = Config()
    dv = DataView(env.customer_profile, env.tariffs, env.channels)
    priors = PriorBuilder(cfg).build(load_history([str(repo_root)]), dv)
    sim = ScoreSimulator(dv, env.customer_profile)
    return env, cfg, dv, priors, sim


def _ratio(model):
    def fn(cur, arpu, tg, ch):
        return float(model.posterior((cur, arpu, tg), ch).mean)
    return fn


def test_pack_coarse_valid_cell_wide_campaigns(world):
    env, cfg, dv, priors, sim = world
    model = ArmModel(cfg, dv, priors)
    packer = CampaignPacker(cfg, dv, sim)
    camps = packer.pack_coarse(model, 100_000.0, 15_000)
    assert 1 <= len(camps) <= 10
    names = [c["campaign_name"] for c in camps]
    assert len(set(names)) == len(names)
    for c in camps:
        assert c["filter_data_segment"] is None and c["filter_call_segment"] is None
        froms = c["filter_current_tariff"].split(";")
        assert c["target_tariff"] not in froms
        assert len(sim.segment(c)) <= 5000
    res = sim.simulate(camps, _ratio(model), 100_000.0, 15_000)
    assert res.within_limits and res.net > 0
    assert packer.last_report["mode"] == "coarse"


def test_pack_coarse_respects_tight_limits_and_is_deterministic(world):
    env, cfg, dv, priors, sim = world
    model = ArmModel(cfg, dv, priors)
    a = CampaignPacker(cfg, dv, sim).pack_coarse(model, 5_000.0, 1_500)
    b = CampaignPacker(cfg, dv, sim).pack_coarse(model, 5_000.0, 1_500)
    assert a == b
    res = sim.simulate(a, _ratio(model), 5_000.0, 1_500)
    assert res.contacts <= 1_500 and res.cost <= 5_000.0 + 1e-9
    assert res.within_limits


def test_pack_coarse_no_positive_arm_returns_empty(world):
    env, cfg, dv, priors, sim = world
    neg = {k: Prior(mu=-abs(p.mu) - 0.01, sd=0.001, share=p.share, n_hist=p.n_hist, source=p.source)
           for k, p in priors.items()}
    assert CampaignPacker(cfg, dv, sim).pack_coarse(ArmModel(cfg, dv, neg), 1e5, 15_000) == []
    assert CampaignPacker(cfg, dv, sim).pack_coarse(ArmModel(cfg, dv, priors), 1e5, 0) == []


def _obs(arm, y, n=200, ch="push"):
    return Observation(arm=arm, channel=ch, y=y, n=n, cost=0.0, sub=None, pilot_index=0)


def _synthetic_model(cfg, dv, mu, sd):
    arms = sorted((ck[0], ck[1], t) for ck in dv.cells for t in dv.targets_for(ck))
    priors = {a: Prior(mu=mu, sd=sd, share=0.5, n_hist=10, source="arm") for a in arms}
    return ArmModel(cfg, dv, priors), arms


def test_calibrate_neutral_when_prior_agrees(world):
    env, cfg, dv, _p, _s = world
    model, arms = _synthetic_model(cfg, dv, 0.05, 0.05)
    k = model.k_channel(arms[0], "push")
    for a in arms[:6]:
        model.add(_obs(a, 0.05 * k))
    cal = model.calibrate()
    assert cal["n_arms"] == 6
    assert cal["tau"] == 0.0
    assert abs(cal["beta"] - 1.0) < 0.05


def test_calibrate_detects_sign_flip_and_inflates_prior(world):
    env, cfg, dv, _p, _s = world
    model, arms = _synthetic_model(cfg, dv, 0.2, 0.01)
    k = model.k_channel(arms[0], "push")
    before = model.posterior(arms[-1], "push")
    for a in arms[:8]:
        model.add(_obs(a, -0.2 * k, n=10_000))
    cal = model.calibrate()
    assert cal["beta"] == 0.0 and cal["tau"] > 0.05
    after = model.posterior(arms[-1], "push")  # unobserved arm: prior shrunk and widened
    assert after.mean < before.mean and after.sd > before.sd


def test_calibrate_needs_min_arms(world):
    env, cfg, dv, _p, _s = world
    model, arms = _synthetic_model(cfg, dv, 0.2, 0.01)
    model.add(_obs(arms[0], -1.0))
    assert model.calibrate()["beta"] == 1.0
    assert math.isclose(model.posterior(arms[5], "push").mean, 0.2 * model.k_channel(arms[5], "push"))


def test_verify_spec_targets_deployed_arm(world, tmp_path):
    from agent_src.m90_orchestrator import Orchestrator

    env, cfg, dv, priors, sim = world
    orch = Orchestrator(cfg.replace(report_path=str(tmp_path / "r.md")), llm_mode="off")
    orch.dv, orch.sim, orch.model = dv, sim, ArmModel(cfg, dv, priors)

    class _Env:
        remaining_budget, remaining_contacts = 100_000.0, 15_000

    spec = orch._verify_spec(_Env())
    assert spec is not None
    assert spec.reason.startswith("verify")
    camps = CampaignPacker(cfg, dv, sim).pack_coarse(orch.model, 100_000.0, 15_000, z=0.5 * cfg.z_risk)
    deployed = {(t, c["filter_arpu_segment"], c["target_tariff"]) for c in camps
                for t in c["filter_current_tariff"].split(";")}
    assert tuple(spec.arm) in deployed
    assert spec.sub[:2] == spec.arm[:2]
    assert cfg.min_pilot <= spec.n <= cfg.max_pilot
    assert spec.channel in dv.channels
