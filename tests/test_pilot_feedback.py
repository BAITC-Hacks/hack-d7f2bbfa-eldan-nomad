"""Counterfactual pilots must change the deployed strategy, with identical priors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent_src.contract import Config, Prior
from agent_src.m30_arm_model import ArmModel
from agent_src.m90_orchestrator import Agent
from environment import make_environment
from scoring_core import CHANNELS, apply_filters, validate_strategy
from tests.fakes_planner import FakeDV, std_spec


class NeutralPriors:
    def __init__(self, cfg):
        pass

    def build(self, history, dv):
        return {(cur, seg, target): Prior(0.0, 0.3, 0.3, 0, "none")
                for cur, seg in dv.cells for target in dv.targets_for((cur, seg))}


def test_same_inputs_but_opposite_pilot_results_change_final_decision():
    profile = pd.DataFrame({
        "ID_NUMBER": np.arange(600), "current_tariff": "tariff_1",
        "arpu_segment": "MID", "data_segment": "LITE", "call_segment": "LOW",
        "predicted_arpu": 3000.0,
    })
    tariffs = pd.DataFrame({"tariff_plan_code": ["tariff_1", "tariff_2"],
                            "price_tariff": [3000.0, 4000.0]})
    deployed = []
    for effect in (0.8, -0.8):
        impact = pd.DataFrame([{"tariff_plan_code_from": "tariff_1",
                                "tariff_plan_code_to": "tariff_2", "arpu_segment": "MID",
                                "arpu_change_pct": effect, "conversion_rate": 1.0}])
        env, _ = make_environment(profile, impact, tariffs, CHANNELS, 100_000, 15_000,
                                  lambda *_: (0.0, 0.0), seed=42)
        agent = Agent(cfg=Config(report_path="", max_pilots=6),
                      components={"PriorBuilder": NeutralPriors}, llm_mode="off")
        campaigns = agent.act(env)
        validate_strategy(pd.DataFrame(campaigns), tariffs)
        orch = agent.last_orchestrator
        assert orch.model.n_obs(("tariff_1", "MID", "tariff_2")) > 0
        assert len(orch.model.observations) == len(env.pilot_history)
        deployed.append(sum(len(apply_filters(profile, pd.Series(c))) for c in campaigns))
    assert deployed[0] > 0
    assert deployed[1] == 0


def test_larger_pilot_reduces_posterior_uncertainty():
    dv = FakeDV(std_spec(), ["tariff_1", "tariff_2", "tariff_3"])
    arm = ("tariff_1", "HIGH", "tariff_2")
    model = ArmModel(Config(), dv, {arm: Prior(0.0, 0.2, 0.3, 0, "none")})
    small = model.posterior_after(arm, "push", 0.15, 30, "push")
    large = model.posterior_after(arm, "push", 0.15, 150, "push")
    assert large.sd < small.sd
    assert abs(large.mean - 0.15) < abs(small.mean - 0.15)
