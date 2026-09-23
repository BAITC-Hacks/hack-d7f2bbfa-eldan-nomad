"""Tests for agent_src.m50_allocator.Allocator (with fake DataView / ArmModel)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from agent_src.contract import Config, Option, Posterior, RunLog, SubCell
from agent_src.m50_allocator import Allocator

COSTS = {"push": 0.0, "sms": 4.0, "digital_ads": 22.0, "call": 160.0}
MULTS = {"push": 0.5, "sms": 0.65, "digital_ads": 0.85, "call": 1.2}


class FakeDV:
    """Minimal DataView: subs, channels, cost/mult, targets_for."""

    def __init__(self, subs: dict, tariffs: list[str], channels=("push", "sms", "digital_ads", "call")):
        self.subs = subs
        self.tariff_codes = tariffs
        self.channels = list(channels)

    def cost(self, ch: str) -> float:
        return COSTS[ch]

    def mult(self, ch: str) -> float:
        return MULTS[ch]

    def cell_of(self, sub):
        return sub[:2]

    def targets_for(self, cell):
        return [t for t in self.tariff_codes if t != cell[0]]


class FakeModel:
    """Options from a table {(sub, target, channel): (net_lcb, p_pos)}; posterior mean from {arm: base}."""

    def __init__(self, dv: FakeDV, table: dict, base: dict | None = None):
        self.dv = dv
        self.table = table
        self.base = base or {}
        self.calls = 0

    def posterior(self, arm, channel) -> Posterior:
        return Posterior(self.base.get(arm, 0.0) * MULTS[channel], 0.1)

    def option(self, sub: SubCell, target: str, channel: str) -> Option:
        self.calls += 1
        lcb, p_pos = self.table.get((sub.key, target, channel), (-1.0, 0.0))
        n = sub.n
        return Option(sub.key, target, channel, n, COSTS[channel] * n, lcb + 1.0, 1.0, lcb, p_pos)


def _sub(key, n):
    ids = np.arange(n, dtype=np.int64)
    return SubCell(key, n, float(n) * 1000.0, 1000.0, ids, np.full(n, 1000.0))


def _random_instance(seed: int, n_subs: int = 7, targets=("t2", "t3")):
    rng = np.random.default_rng(seed)
    subs, table = {}, {}
    for i in range(n_subs):
        key = ("t1", "LOW", "LITE", f"S{i:02d}")
        n = int(rng.integers(20, 200))
        subs[key] = _sub(key, n)
        for t in targets:
            for ch in COSTS:
                lcb = float(rng.uniform(-500, 3000)) + (COSTS[ch] * n * rng.uniform(0, 0.6))
                table[(key, t, ch)] = (lcb, float(rng.uniform(0.5, 1.0)))
    dv = FakeDV(subs, ["t1", *targets])
    return dv, FakeModel(dv, table)


def _brute(alloc: Allocator, dv: FakeDV, budget: float, contacts: int) -> float:
    per = [alloc.options_for(dv.subs[k]) for k in sorted(dv.subs)]
    best = 0.0
    for combo in itertools.product(*[[None, *p] for p in per]):
        ch = [o for o in combo if o is not None]
        if sum(o.cost for o in ch) <= budget and sum(o.n for o in ch) <= contacts:
            best = max(best, sum(o.net_lcb for o in ch))
    return best


def _check_feasible(plan, budget, contacts):
    assert plan.total_cost <= budget + 1e-6
    assert plan.total_contacts <= contacts
    subs = [o.sub for o in plan.options]
    assert len(subs) == len(set(subs))
    assert plan.total_cost == pytest.approx(sum(o.cost for o in plan.options))
    assert plan.total_contacts == sum(o.n for o in plan.options)


def test_options_admissibility_and_sorting():
    key = ("t1", "MID", "HEAVY", "LOW")
    dv = FakeDV({key: _sub(key, 100)}, ["t1", "t2", "t3", "t4"])
    table = {
        (key, "t2", "push"): (50.0, 0.65),  # push: p_min_push 0.6 -> ok
        (key, "t2", "sms"): (400.0, 0.7),  # paid below p_min 0.8 -> dropped
        (key, "t2", "call"): (300.0, 0.95),
        (key, "t3", "digital_ads"): (-5.0, 0.99),  # negative lcb -> dropped
        (key, "t3", "sms"): (800.0, 0.9),
        (key, "t4", "sms"): (9999.0, 0.99),  # t4 not in top-2 targets
    }
    base = {("t1", "MID", "t2"): 0.3, ("t1", "MID", "t3"): 0.2, ("t1", "MID", "t4"): 0.1}
    alloc = Allocator(Config(top_targets_per_cell=2), dv, FakeModel(dv, table, base))
    opts = alloc.options_for(dv.subs[key])
    assert [(o.target, o.channel) for o in opts] == [("t3", "sms"), ("t2", "call"), ("t2", "push")]
    assert all(opts[i].net_lcb >= opts[i + 1].net_lcb for i in range(len(opts) - 1))


@pytest.mark.parametrize("seed", range(12))
def test_matches_bruteforce_within_2pct(seed):
    dv, model = _random_instance(seed)
    rng = np.random.default_rng(1000 + seed)
    total_n = sum(s.n for s in dv.subs.values())
    budget = float(rng.uniform(0.0, 40000.0))
    contacts = int(rng.uniform(0.2, 1.0) * total_n)
    alloc = Allocator(Config(), dv, model)
    plan = alloc.allocate(budget, contacts)
    _check_feasible(plan, budget, contacts)
    opt = _brute(alloc, dv, budget, contacts)
    assert plan.total_net_lcb >= 0.98 * opt - 1e-6


@pytest.mark.parametrize("seed", range(20))
def test_constraints_always_satisfied_larger(seed):
    dv, model = _random_instance(seed, n_subs=60, targets=("t2", "t3", "t4"))
    rng = np.random.default_rng(seed)
    budget = float(rng.choice([0.0, 500.0, 1e4, 1e5, 1e7]))
    contacts = int(rng.choice([0, 50, 1000, 3000, 10**6]))
    plan = Allocator(Config(), dv, model).allocate(budget, contacts)
    _check_feasible(plan, budget, contacts)
    assert all(o.net_lcb > 0 for o in plan.options)


def test_zero_budget_push_only():
    dv, model = _random_instance(3, n_subs=10)
    plan = Allocator(Config(), dv, model).allocate(0.0, 10**6)
    assert plan.options, "some push options should be admissible"
    assert all(o.channel == "push" and o.cost == 0.0 for o in plan.options)
    assert plan.total_cost == 0.0


def test_nothing_admissible_empty_plan():
    key = ("t1", "LOW", "LITE", "LOW")
    dv = FakeDV({key: _sub(key, 50)}, ["t1", "t2"])
    table = {(key, "t2", ch): (-10.0, 0.99) for ch in COSTS}
    table[(key, "t2", "sms")] = (100.0, 0.5)  # positive but p_pos too low
    plan = Allocator(Config(), dv, FakeModel(dv, table)).allocate(1e6, 10**6)
    assert plan.options == []
    assert plan.total_net_lcb == 0.0 and plan.total_cost == 0.0 and plan.total_contacts == 0


def test_empty_dataview():
    dv = FakeDV({}, ["t1", "t2"])
    plan = Allocator(Config(), dv, FakeModel(dv, {})).allocate(1e6, 1000)
    assert plan.options == [] and plan.lambda_money == 0.0 and plan.lambda_reach == 0.0


def test_unconstrained_picks_best_per_sub_and_zero_lambdas():
    dv, model = _random_instance(5, n_subs=8)
    alloc = Allocator(Config(), dv, model)
    plan = alloc.allocate(1e12, 10**9)
    assert plan.lambda_money == 0.0 and plan.lambda_reach == 0.0
    best = {k: alloc.options_for(s)[0] for k, s in dv.subs.items() if alloc.options_for(s)}
    assert {o.sub: (o.target, o.channel) for o in plan.options} == {
        k: (o.target, o.channel) for k, o in best.items()}


def test_binding_constraints_give_positive_shadow_prices():
    dv, model = _random_instance(7, n_subs=30)
    alloc = Allocator(Config(), dv, model)
    lm, lr = alloc.shadow_prices(2000.0, 10**9)
    assert lm > 0.0 and lr == 0.0
    total_n = sum(s.n for s in dv.subs.values())
    lm2, lr2 = alloc.shadow_prices(1e12, total_n // 5)
    assert lr2 > 0.0 and lm2 == 0.0


def test_exclude_subs_and_extra_cost():
    dv, model = _random_instance(11, n_subs=6)
    alloc = Allocator(Config(), dv, model)
    full = alloc.allocate(1e12, 10**9)
    chosen = sorted({o.sub for o in full.options})
    excl = {chosen[0]}
    plan = alloc.allocate(1e12, 10**9, exclude_subs=excl)
    assert all(o.sub not in excl for o in plan.options)
    # a huge overlap cost removes the sub; a small one reduces the objective by exactly that amount
    big = {chosen[1]: 1e12}
    plan2 = alloc.allocate(1e12, 10**9, extra_cost=big)
    assert chosen[1] not in {o.sub for o in plan2.options}
    small = {chosen[1]: 1.0}
    plan3 = alloc.allocate(1e12, 10**9, extra_cost=small)
    assert plan3.total_net_lcb == pytest.approx(full.total_net_lcb - 1.0)


def test_deterministic_and_logged():
    dv, model = _random_instance(2, n_subs=40, targets=("t2", "t3", "t4"))
    log = RunLog()
    a = Allocator(Config(), dv, model, log=log).allocate(20000.0, 2500)
    b = Allocator(Config(), dv, model).allocate(20000.0, 2500)
    assert [(o.sub, o.target, o.channel) for o in a.options] == [(o.sub, o.target, o.channel) for o in b.options]
    assert a.lambda_money == b.lambda_money and a.lambda_reach == b.lambda_reach
    assert log.events and log.events[-1]["kind"] == "allocate"


def test_nan_options_ignored():
    key = ("t1", "LOW", "LITE", "LOW")
    dv = FakeDV({key: _sub(key, 50)}, ["t1", "t2"])
    table = {(key, "t2", "push"): (float("nan"), 0.99), (key, "t2", "sms"): (100.0, 0.95)}
    plan = Allocator(Config(), dv, FakeModel(dv, table)).allocate(1e6, 1000)
    assert [(o.channel) for o in plan.options] == ["sms"]


@pytest.mark.parametrize("seed", range(6))
def test_fast_mode_feasible_and_close(seed):
    dv, model = _random_instance(100 + seed)
    total_n = sum(s.n for s in dv.subs.values())
    budget, contacts = 8000.0, total_n // 2
    alloc = Allocator(Config(), dv, model)
    fast = alloc.allocate(budget, contacts, polish=False)
    full = alloc.allocate(budget, contacts)
    _check_feasible(fast, budget, contacts)
    assert full.total_net_lcb >= fast.total_net_lcb - 1e-6
    assert fast.total_net_lcb >= 0.8 * _brute(alloc, dv, budget, contacts)


def test_tiny_contacts_limit():
    dv, model = _random_instance(9, n_subs=8)
    min_n = min(s.n for s in dv.subs.values())
    plan = Allocator(Config(), dv, model).allocate(1e9, min_n - 1)
    assert plan.options == []


@pytest.mark.parametrize("seed", range(8))
def test_both_constraints_binding_near_bruteforce(seed):
    """Money and reach bind simultaneously (coupled nested bisection) -> feasible and near-optimal."""
    dv, model = _random_instance(200 + seed)
    alloc = Allocator(Config(), dv, model)
    free = alloc.allocate(1e12, 10**9)
    budget, contacts = 0.4 * free.total_cost, int(0.5 * free.total_contacts)
    plan = alloc.allocate(budget, contacts)
    _check_feasible(plan, budget, contacts)
    assert plan.total_net_lcb >= 0.98 * _brute(alloc, dv, budget, contacts) - 1e-6
    lm, lr = alloc.shadow_prices(budget, contacts)
    assert lm >= 0.0 and lr >= 0.0 and np.isfinite(lm) and np.isfinite(lr)


def test_extra_cost_sanitised():
    """NaN / negative / -inf extra costs are ignored (no bonuses); +inf removes the sub."""
    dv, model = _random_instance(11, n_subs=6)
    alloc = Allocator(Config(), dv, model)
    full = alloc.allocate(1e12, 10**9)
    chosen = sorted({o.sub for o in full.options})
    for bad in (float("nan"), -5.0, float("-inf")):
        plan = alloc.allocate(1e12, 10**9, extra_cost={chosen[0]: bad})
        assert plan.total_net_lcb == pytest.approx(full.total_net_lcb)
        assert np.isfinite(plan.total_net_lcb)
    plan = alloc.allocate(1e12, 10**9, extra_cost={chosen[0]: float("inf")})
    assert chosen[0] not in {o.sub for o in plan.options}


def test_infinite_budget_means_no_money_limit_and_nan_is_zero():
    dv, model = _random_instance(5, n_subs=8)
    alloc = Allocator(Config(), dv, model)
    ref = alloc.allocate(1e12, 10**9)
    inf_plan = alloc.allocate(float("inf"), 10**9)
    assert inf_plan.total_net_lcb == pytest.approx(ref.total_net_lcb)
    nan_plan = alloc.allocate(float("nan"), 10**9)
    assert nan_plan.total_cost == 0.0


def test_target_ranking_uses_model_channel_scale():
    """With a saturation-aware k_channel the raw-multiplier proxy is replaced."""
    key = ("t1", "LOW", "LITE", "LOW")
    dv = FakeDV({key: _sub(key, 50)}, ["t1", "t2", "t3"], channels=("call",))

    class KModel(FakeModel):
        def posterior(self, arm, channel):
            # t2: saturated (k=1.0), mean 0.30 -> base 0.30; t3: k=1.2, mean 0.33 -> base 0.275
            return Posterior({"t2": 0.30, "t3": 0.33}[arm[2]], 0.1)

        def k_channel(self, arm, channel):
            return {"t2": 1.0, "t3": 1.2}[arm[2]]

    alloc = Allocator(Config(top_targets_per_cell=1), dv, KModel(dv, {}))
    assert alloc._alloc_rank_targets(key[:2]) == ["t2"]
