"""Fake components and env for orchestrator tests (only INTERFACES.md surface)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from agent_src.contract import (
    Cell,
    Option,
    Plan,
    PilotSpec,
    Posterior,
    Prior,
    ReviewOutcome,
    SimResult,
    SubCell,
    campaign_dict,
)

SUBS = [("tariff_1", "HIGH", "LITE", "LOW"), ("tariff_2", "MID", "HEAVY", "HIGH")]


class FakeEnv:
    """Minimal env with a deterministic run_pilot."""

    def __init__(self, fail_pilots: bool = False) -> None:
        self.customer_profile = pd.DataFrame({
            "ID_NUMBER": [1, 2, 3, 4],
            "current_tariff": ["tariff_1", "tariff_1", "tariff_2", "tariff_2"],
            "arpu_segment": ["HIGH", "HIGH", "MID", "MID"],
            "data_segment": ["LITE", "LITE", "HEAVY", "HEAVY"],
            "call_segment": ["LOW", "LOW", "HIGH", "HIGH"],
            "predicted_arpu": [9000.0, 8000.0, 3000.0, 3500.0],
        })
        self.tariffs = pd.DataFrame({"tariff_plan_code": ["tariff_1", "tariff_2", "tariff_3"],
                                     "price_tariff": [1.0, 2.0, 3.0]})
        self.channels = {"push": {"cost_per_contact": 0, "conversion_multiplier": 0.5},
                         "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65}}
        self.remaining_budget = 100000.0
        self.remaining_contacts = 15000
        self.pilots_left = 20
        self.pilot_history: list[dict] = []
        self.fail_pilots = fail_pilots
        self.calls: list[dict] = []

    def run_pilot(self, target_tariff, channel, n_customers=100, filter_arpu_segment=None,
                  filter_data_segment=None, filter_call_segment=None, filter_current_tariff=None):
        self.calls.append(dict(target_tariff=target_tariff, channel=channel, n_customers=n_customers))
        if self.fail_pilots:
            raise RuntimeError("empty segment")
        n = min(int(n_customers), 2)  # env may return fewer customers than asked
        cost = n * self.channels[channel]["cost_per_contact"]
        self.remaining_budget -= cost
        self.remaining_contacts -= n
        self.pilots_left -= 1
        res = {"pilot": f"pilot_{len(self.pilot_history) + 1}", "n_customers": n, "cost": cost,
               "observed_lift_ratio": 0.25, "target_tariff": target_tariff, "channel": channel}
        self.pilot_history.append(res)
        return res


class Boom(Exception):
    """Raised by sabotaged fakes."""


class FakeDataView:
    def __init__(self, profile, tariffs, channels):
        self.subs = {}
        for s in SUBS:
            self.subs[s] = SubCell(key=s, n=2, sum_p=17000.0, mean_p=8500.0,
                                   ids=np.array([1, 2], dtype=np.int64), p=np.array([9000.0, 8000.0]))
        self.cells = {s[:2]: Cell(key=s[:2], n=2, sum_p=17000.0, mean_p=8500.0, subs=[s]) for s in SUBS}
        self.n_nan = 0
        self.tariff_codes = ["tariff_1", "tariff_2", "tariff_3"]
        self.tariff_price = {"tariff_1": 1.0, "tariff_2": 2.0, "tariff_3": 3.0}
        self.channels = ["push", "sms"]

    def tariff_info(self, code):
        return {"tariff_plan_code": code, "price_tariff": self.tariff_price[code]}

    def cost(self, ch):
        return {"push": 0.0, "sms": 4.0}[ch]

    def mult(self, ch):
        return {"push": 0.5, "sms": 0.65}[ch]

    def targets_for(self, cell):
        return [t for t in self.tariff_codes if t != cell[0]]


def fake_load_history(dirs):
    return pd.DataFrame({"x": [1, 2, 3]})


def fake_load_descriptions(dirs):
    return {"tariff_3": "big tariff"}


class FakePriorBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, history, dv):
        out = {}
        for cell in sorted(dv.cells):
            for t in dv.targets_for(cell):
                out[(cell[0], cell[1], t)] = Prior(mu=0.2, sd=0.1, share=0.3, n_hist=10, source="arm")
        return out


class FakeArmModel:
    def __init__(self, cfg, dv, priors):
        self.observations = []

    def add(self, obs):
        self.observations.append(obs)

    def posterior(self, arm, channel):
        return Posterior(mean=0.2, sd=0.05)

    def n_obs(self, arm):
        return sum(1 for o in self.observations if o.arm == arm)


class FakeSimulator:
    def __init__(self, dv, profile):
        self.dv = dv

    def simulate(self, campaigns, ratio_fn, budget, contacts):
        per = [{"name": c["campaign_name"], "n_segment": 2, "n_contacted": 2, "cost": 8.0, "gross": 100000.0,
                "capped_campaign": False, "capped_reach": False, "capped_money": False} for c in campaigns]
        gross = 100000.0 * len(campaigns)
        return SimResult(gross=gross, cost=8.0 * len(campaigns), contacts=2 * len(campaigns),
                         net=gross - 8.0 * len(campaigns), per_campaign=per, within_limits=True)


def _opt(sub, target="tariff_3", channel="sms"):
    return Option(sub=sub, target=target, channel=channel, n=2, cost=8.0, net_mean=100.0, net_sd=10.0,
                  net_lcb=90.0, p_pos=0.99)


class FakeAllocator:
    calls: list = []

    def __init__(self, cfg, dv, model):
        self.dv = dv

    def shadow_prices(self, budget, contacts):
        return (0.0, 1.0)

    def allocate(self, budget, contacts, exclude_subs=frozenset(), extra_cost=None):
        FakeAllocator.calls.append({"exclude": set(exclude_subs), "extra": dict(extra_cost or {})})
        opts = [_opt(s) for s in SUBS if s not in exclude_subs]
        return Plan(options=opts, lambda_money=0.0, lambda_reach=0.0, total_net_lcb=90.0 * len(opts),
                    total_cost=8.0 * len(opts), total_contacts=2 * len(opts))


class FakePacker:
    def __init__(self, cfg, dv, sim):
        pass

    def pack(self, plan, model, budget, contacts):
        out = []
        for i, o in enumerate(plan.options, 1):
            out.append(campaign_dict(campaign_name=f"c{i:02d}_{o.target}_{o.channel}_{o.sub[1]}",
                                     filter_arpu_segment=o.sub[1], filter_data_segment=o.sub[2],
                                     filter_call_segment=o.sub[3], filter_current_tariff=o.sub[0],
                                     target_tariff=o.target, channel=o.channel))
        return out

    def empty_campaign(self):
        return campaign_dict(campaign_name="c00_empty", filter_arpu_segment="LOW", filter_current_tariff="tariff_3",
                             target_tariff="tariff_1", channel="push")


class FakePlanner:
    """Plans `max_specs` pilots, then stops."""

    max_specs = 3
    delay_s = 0.0

    def __init__(self, cfg, dv, model, allocator):
        self.weights = None
        self.registered = []
        self.issued = 0

    def set_weights(self, w):
        self.weights = dict(w)

    def next_pilot(self, state):
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.issued >= self.max_specs:
            return None
        sub = SUBS[self.issued % len(SUBS)]
        self.issued += 1
        return PilotSpec(arm=(sub[0], sub[1], "tariff_3"), channel="sms", n=60, sub=sub,
                         filters={"filter_arpu_segment": sub[1], "filter_data_segment": sub[2],
                                  "filter_call_segment": sub[3], "filter_current_tariff": sub[0]},
                         score=1.0, reason="test")

    def register_result(self, spec, result):
        self.registered.append((spec, result))


class FakeLLM:
    veto: list = []
    applied = True

    def __init__(self, cfg, mode, cache_path, log):
        self.mode = mode
        self.batches = None
        self.what_if_result = None

    async def assess_hypotheses(self, batches, tariffs):
        self.batches = batches
        out = {}
        for items in batches.values():
            for it in items:
                arm = tuple(it["arm_id"].split("|"))
                out[arm] = 1.2
        return out

    async def review_plan(self, campaigns, evidence, what_if):
        self.what_if_result = what_if(list(self.veto))
        return ReviewOutcome(veto=list(self.veto), rationale="r", summary="s",
                             applied=self.applied and self.mode == "decide", source="llm")


class FakeGuardrails:
    def __init__(self, cfg, dv, sim):
        pass

    def validate(self, campaigns, budget, contacts):
        out = [dict(c) for c in campaigns][:10]
        if not out:
            out = [FakePacker(None, None, None).empty_campaign()]
        return out


class FakeReporter:
    written: list = []

    def __init__(self, cfg, path):
        self.path = path

    def write(self, log, campaigns, extra):
        FakeReporter.written.append({"path": self.path, "n": len(campaigns), "extra": extra})


def fake_components(**over: Any) -> dict:
    comps = {
        "DataView": FakeDataView,
        "load_history": fake_load_history,
        "load_tariff_descriptions": fake_load_descriptions,
        "PriorBuilder": FakePriorBuilder,
        "ArmModel": FakeArmModel,
        "ScoreSimulator": FakeSimulator,
        "Allocator": FakeAllocator,
        "CampaignPacker": FakePacker,
        "PilotPlanner": FakePlanner,
        "LLMLayer": FakeLLM,
        "Guardrails": FakeGuardrails,
        "Reporter": FakeReporter,
    }
    comps.update(over)
    return comps
