"""Tests for agent_src.m70_planner.PilotPlanner (fakes only; one optional smoke test with real modules)."""
from __future__ import annotations

import math
import time

import pytest

from agent_src.contract import Config, ExploreState, Observation, PilotSpec
from agent_src.m70_planner import PilotPlanner
from tests.fakes_planner import FakeAllocator, FakeDV, FakeModel, std_spec

TARIFFS = ["tariff_1", "tariff_2", "tariff_3"]


def make(priors=None, default=(0.0, 0.05), lm=0.0, lr=0.0, cfg=None, spec=None):
    dv = FakeDV(spec or std_spec(), TARIFFS)
    model = FakeModel(dv, priors or {}, default=default)
    planner = PilotPlanner(cfg or Config(), dv, model, FakeAllocator(lm, lr))
    return dv, model, planner


def state(**kw):
    base = dict(remaining_budget=100_000.0, remaining_contacts=15_000, pilots_left=20,
                explore_money_spent=0.0, explore_reach_spent=0)
    base.update(kw)
    return ExploreState(**base)


def run(planner, model, spec: PilotSpec, st: ExploreState, y: float, n=None):
    """Pretend the pilot ran: register, add observation, update the state."""
    n = spec.n if n is None else n
    cost = n * planner.dv.cost(spec.channel)
    planner.register_result(spec, {"n_customers": n, "cost": cost, "observed_lift_ratio": y})
    model.add(Observation(spec.arm, spec.channel, y, n, cost, spec.sub, len(model.observations)))
    st.remaining_budget -= cost
    st.remaining_contacts -= n
    st.pilots_left -= 1
    st.explore_money_spent += cost
    st.explore_reach_spent += n
    st.pilots_per_arm[spec.arm] = st.pilots_per_arm.get(spec.arm, 0) + 1
    if spec.sub is not None:
        st.used_subs[spec.sub] = st.used_subs.get(spec.sub, 0) + n


def test_always_one_pilot_when_nothing_is_worth_it():
    # everything clearly negative and certain -> KG ~ 0, still a first (cheap) pilot is returned
    _, _, planner = make(default=(-0.3, 0.001))
    spec = planner.next_pilot(state())
    assert spec is not None
    assert spec.channel == "push"  # cheapest
    assert planner.last_debug["decision"] == "forced_first"
    kw = spec.run_kwargs()
    assert kw["target_tariff"] == spec.arm[2] and kw["filter_current_tariff"] == spec.arm[0]


def test_stops_after_first_pilot_when_nothing_is_worth_it():
    _, model, planner = make(default=(-0.3, 0.001))
    st = state()
    spec = planner.next_pilot(st)
    run(planner, model, spec, st, y=-0.3)
    assert planner.next_pilot(st) is None
    assert planner.last_debug["decision"] == "stop"


def test_no_pilots_left_or_deadline():
    _, _, planner = make(default=(0.05, 0.2))
    assert planner.next_pilot(state(pilots_left=0)) is None
    _, model, planner = make(default=(0.05, 0.2))
    st = state()
    spec = planner.next_pilot(st)
    run(planner, model, spec, st, y=0.05)
    st.deadline = time.monotonic() - 1.0
    assert planner.next_pilot(st) is None


def test_prefers_high_value_uncertain_arm():
    a_unc = ("tariff_1", "HIGH", "tariff_3")
    priors = {a_unc: (0.0, 0.3)}
    _, _, planner = make(priors=priors, default=(0.0, 0.005))
    spec = planner.next_pilot(state())
    assert spec is not None
    assert spec.arm == a_unc
    assert planner.last_debug["kg"] > 0


def test_respects_max_pilots_per_arm():
    a_unc = ("tariff_1", "HIGH", "tariff_3")
    priors = {a_unc: (0.0, 0.3), ("tariff_2", "MID", "tariff_3"): (0.0, 0.2)}
    _, _, planner = make(priors=priors, default=(0.0, 0.005))
    st = state(pilots_per_arm={a_unc: 3}, explore_reach_spent=300)
    spec = planner.next_pilot(st)
    assert spec is None or spec.arm != a_unc


def test_respects_money_and_reach_reserve():
    priors = {("tariff_1", "HIGH", "tariff_3"): (0.02, 0.3)}
    _, _, planner = make(priors=priors, default=(0.0, 0.005))
    # money reserve: 25% of initial 100k = 25k, already spent 24.9k -> <= 100 money left
    st = state(remaining_budget=75_100.0, explore_money_spent=24_900.0, explore_reach_spent=10)
    spec = planner.next_pilot(st)
    assert spec is not None
    assert spec.n * planner.dv.cost(spec.channel) <= 100.0 + 1e-9
    # reach reserve: 30% of 15000 = 4500, spent 4495 -> < min_pilot left -> stop
    st2 = state(remaining_contacts=10_505, explore_reach_spent=4_495, pilots_per_arm={("x", "y", "z"): 1})
    assert planner.next_pilot(st2) is None


def test_pilot_sub_is_smallest_fitting_sub():
    a = ("tariff_1", "HIGH", "tariff_3")
    _, _, planner = make(priors={a: (0.0, 0.3)}, default=(0.0, 0.005))
    spec = planner.next_pilot(state())
    assert spec.sub is not None
    sub_n = planner.dv.subs[spec.sub].n
    fitting = [s.n for s in planner.dv.subs_of(spec.arm[:2]) if s.n >= spec.n]
    assert sub_n == min(fitting)
    assert spec.filters == {"filter_current_tariff": spec.sub[0], "filter_arpu_segment": spec.sub[1],
                            "filter_data_segment": spec.sub[2], "filter_call_segment": spec.sub[3]}


def test_pilot_is_always_a_sub_cell_sized_to_fit():
    # requested size 45 fits no sub -> pilot on the largest sub with its full size (never a whole-cell pilot)
    spec_cells = {("tariff_1", "HIGH"): [("LITE", "LOW", 20, 9000.0), ("HEAVY", "LOW", 25, 9000.0)]}
    a = ("tariff_1", "HIGH", "tariff_3")
    cfg = Config(pilot_sizes=(45,))
    _, _, planner = make(priors={a: (0.0, 0.5)}, default=(0.0, 0.005), cfg=cfg, spec=spec_cells)
    spec = planner.next_pilot(state())
    assert spec is not None and spec.sub == ("tariff_1", "HIGH", "HEAVY", "LOW") and spec.n == 25
    assert spec.filters["filter_data_segment"] == "HEAVY" and spec.filters["filter_call_segment"] == "LOW"


def test_no_pilot_when_every_sub_is_below_min_pilot():
    spec_cells = {("tariff_1", "HIGH"): [("LITE", "LOW", 5, 9000.0), ("HEAVY", "LOW", 8, 9000.0)]}
    _, _, planner = make(priors={("tariff_1", "HIGH", "tariff_3"): (0.0, 0.5)}, spec=spec_cells)
    assert planner.next_pilot(state()) is None
    assert planner.last_debug["reason"] == "no_feasible_pilot"


def test_register_error_marks_unavailable():
    a = ("tariff_1", "HIGH", "tariff_3")
    _, _, planner = make(priors={a: (0.0, 0.3)}, default=(0.0, 0.005))
    st = state()
    spec = planner.next_pilot(st)
    planner.register_result(spec, {"error": "boom"})
    assert (spec.arm, spec.sub) in planner.unavailable
    spec2 = planner.next_pilot(st)
    assert spec2 is not None
    assert (spec2.arm, spec2.sub) != (spec.arm, spec.sub)


def test_register_result_bookkeeping():
    _, _, planner = make(default=(0.0, 0.2))
    spec = planner.next_pilot(state())
    planner.register_result(spec, {"n_customers": 7, "cost": 28.0})
    assert planner.pilots_per_arm[spec.arm] == 1
    if spec.sub is not None:
        assert planner.used_subs[spec.sub] == 7


def test_deterministic():
    specs = []
    for _ in range(2):
        _, _, planner = make(default=(0.01, 0.1))
        s = planner.next_pilot(state())
        specs.append((s.arm, s.channel, s.n, s.sub, round(s.score, 6)))
    assert specs[0] == specs[1]


def test_learning_loop_eventually_stops_and_respects_limits():
    a = ("tariff_1", "HIGH", "tariff_3")
    _, model, planner = make(priors={a: (0.02, 0.1)}, default=(0.0, 0.02))
    st = state()
    count = 0
    while True:
        spec = planner.next_pilot(st)
        if spec is None:
            break
        truth = 0.02 * planner.dv.mult(spec.channel) if spec.arm == a else 0.0
        run(planner, model, spec, st, y=truth)
        count += 1
        assert count <= 20
    assert 1 <= count <= 20
    assert st.explore_money_spent <= 0.25 * 100_000 + 1e-6
    assert st.explore_reach_spent <= 0.30 * 15_000
    assert all(v <= 3 for v in st.pilots_per_arm.values())


def test_winner_curse_bonus_for_big_paid_plan():
    # strongly positive but poorly confirmed arm whose plan is paid and big -> bonus > 0
    a = ("tariff_1", "HIGH", "tariff_3")
    _, _, planner = make(priors={a: (0.1, 0.03)}, default=(-0.2, 0.001))
    ev = planner._cell_eval(("tariff_1", "HIGH"), 0.0, 0.0)
    t = ev.t_index["tariff_3"]
    info = planner._confirm_info(state(), a, ev, t, 100_000.0)
    assert info is not None and info[0] > 0
    spec = planner.next_pilot(state())
    assert spec.arm == a and planner.last_debug["bonus"] > 0


def test_weights_affect_ranking():
    cfg = Config(top_arms=1)
    b = ("tariff_2", "MID", "tariff_3")
    _, _, planner = make(default=(0.0, 0.1), cfg=cfg)
    planner.set_weights({b: 50.0})
    spec = planner.next_pilot(state())
    assert spec.arm == b


def test_allocator_failure_is_tolerated():
    dv = FakeDV(std_spec(), TARIFFS)
    model = FakeModel(dv, {}, default=(0.0, 0.1))
    planner = PilotPlanner(Config(), dv, model, FakeAllocator(fail=True))
    assert planner.next_pilot(state()) is not None


def test_compute_is_bounded_on_realistic_size():
    tariffs = [f"tariff_{i}" for i in range(1, 22)]
    spec = {}
    for i, t in enumerate(tariffs):
        for arpu, mp in (("LOW", 1500.0), ("MID", 3500.0), ("HIGH", 9000.0)):
            spec[(t, arpu)] = [(d, c, 30 + 40 * j + i, mp) for j, (d, c) in enumerate(
                [("NON_USER", "LOW"), ("LITE", "LOW"), ("LITE", "HIGH"), ("HEAVY", "LOW"), ("HEAVY", "MEDIUM"),
                 ("HEAVY", "HIGH")])]
    dv = FakeDV(spec, tariffs)
    model = FakeModel(dv, {}, default=(0.01, 0.1))
    planner = PilotPlanner(Config(), dv, model, FakeAllocator(1e-3, 1.0))
    t0 = time.monotonic()
    spec_ = planner.next_pilot(state())
    assert spec_ is not None
    assert time.monotonic() - t0 < 3.0  # realistic size finishes far below the internal cap (5 s after a fixed arm prefix)


def test_smoke_with_real_modules(mock_env_factory):
    try:
        from agent_src.m10_dataview import DataView, load_history
        from agent_src.m20_prior import PriorBuilder
        from agent_src.m30_arm_model import ArmModel
        from agent_src.m50_allocator import Allocator
    except ImportError:  # teammates' modules not ready
        pytest.skip("real modules missing")
    env = mock_env_factory(0)
    cfg = Config()
    dv = DataView(env.customer_profile, env.tariffs, env.channels)
    priors = PriorBuilder(cfg).build(load_history(["."]), dv)
    model = ArmModel(cfg, dv, priors)
    planner = PilotPlanner(cfg, dv, model, Allocator(cfg, dv, model))
    st = ExploreState(env.remaining_budget, env.remaining_contacts, env.pilots_left, 0.0, 0,
                      deadline=time.monotonic() + 60)
    n = 0
    while n < 4:
        t0 = time.monotonic()
        spec = planner.next_pilot(st)
        assert time.monotonic() - t0 < 3.0  # allocator.shadow_prices dominates on real data
        if spec is None:
            break
        res = env.run_pilot(**spec.run_kwargs())
        planner.register_result(spec, res)
        model.add(Observation(spec.arm, spec.channel, res["observed_lift_ratio"], res["n_customers"], res["cost"],
                              spec.sub, len(env.pilot_history) - 1))
        st.remaining_budget, st.remaining_contacts = env.remaining_budget, env.remaining_contacts
        st.pilots_left = env.pilots_left
        st.explore_money_spent += res["cost"]
        st.explore_reach_spent += res["n_customers"]
        st.pilots_per_arm[spec.arm] = st.pilots_per_arm.get(spec.arm, 0) + 1
        n += 1
    assert n >= 1
    assert math.isfinite(st.remaining_budget)


# ---------------------------------------------------------------- cross-check additions


def _ref_value(planner, model, ev, arm, post_fn, cfg, lm=0.0, lr=0.0):
    """Brute-force cell value: per sub, best admissible (target, channel) by lcb score, valued at the mean.

    Only the allocator's top `top_targets_per_cell` targets (by max_c mean_c / mult_c) may be deployed.
    """
    ck = arm[:2]
    tg = planner.dv.targets_for(ck)
    prox = [max(post_fn((ck[0], ck[1], x), ch)[0] / planner.dv.mult(ch) for ch in planner.dv.channels) for x in tg]
    elig = [tg[i] for i in sorted(range(len(tg)), key=lambda i: (-prox[i], i))[: cfg.top_targets_per_cell]]
    total = 0.0
    for si, sc in enumerate(planner.dv.subs_of(ck)):
        best = None
        for tgt in elig:
            a = (ck[0], ck[1], tgt)
            for ch in planner.dv.channels:
                m, sd = post_fn(a, ch)
                cost = planner.dv.cost(ch) * sc.n
                net = m * sc.sum_p - cost
                nsd = abs(sd * sc.sum_p)
                lcb = net - cfg.z_risk * nsd
                p_min = cfg.p_min_push if ch == "push" else cfg.p_min
                p_pos = (0.5 * (1 + math.erf(net / nsd / math.sqrt(2)))) if nsd > 0 else float(net > 0)
                if lcb <= 0 or p_pos < p_min - 1e-9:
                    continue
                score = lcb - lm * cost - lr * sc.n
                if score > 0 and (best is None or score > best[0]):
                    best = (score, net - lm * cost - lr * sc.n)
        total += 0.0 if best is None else best[1]
    return total


@pytest.mark.parametrize("k", [3, 1])
def test_kg_matches_bruteforce_enumeration(k):
    import numpy as np

    a = ("tariff_1", "HIGH", "tariff_3")
    b = ("tariff_1", "HIGH", "tariff_2")
    cfg = Config(gh_nodes=7, top_targets_per_cell=k)
    _, model, planner = make(priors={a: (0.01, 0.08), b: (0.012, 0.004)}, default=(0.0, 0.005), cfg=cfg)
    ck = ("tariff_1", "HIGH")
    ev = planner._cell_eval(ck, 0.0, 0.0)
    t = ev.t_index["tariff_3"]
    others = [i for i in ev.rank if i != t]
    comp_in, comp_out = planner._competitor(ev, others[: k - 1]), planner._competitor(ev, others[:k])
    thr = (float(ev.proxy[others[k - 1]]), others[k - 1]) if len(others) >= k else (-math.inf, math.inf)
    pj = planner._channels.index("sms")
    n = 100
    m_p, sd_p = model.posterior(a, "sms").mean, model.posterior(a, "sms").sd
    kg, imm = planner._kg(a, ev, t, pj, "sms", n, m_p, sd_p, cfg.noise_sd ** 2, comp_in, comp_out, thr, 3, 1.0,
                          0.0, 0.0)
    # reference: explicit Gauss-Hermite over y with exact posterior_after calls
    x, w = np.polynomial.hermite.hermgauss(cfg.gh_nodes)
    pred = math.sqrt(sd_p ** 2 + cfg.noise_sd ** 2 / n)
    before = _ref_value(planner, model, ev, a, lambda arm, ch: (model.posterior(arm, ch).mean,
                                                                 model.posterior(arm, ch).sd), cfg)
    after = 0.0
    for xi, wi in zip(x, w):
        y = m_p + pred * xi * math.sqrt(2)

        def post(arm, ch, y=y):
            p = model.posterior_after(arm, "sms", y, n, ch) if arm == a else model.posterior(arm, ch)
            return p.mean, p.sd

        after += wi / math.sqrt(math.pi) * _ref_value(planner, model, ev, a, post, cfg)
    assert kg == pytest.approx(after - before, rel=1e-6, abs=1e-3)
    assert kg > 0
    assert math.isfinite(imm)


def test_immediate_not_double_counted_when_arm_already_deployed():
    # certain, strongly positive arm deployed on push everywhere: a push pilot adds no lift beyond the final plan
    a = ("tariff_1", "HIGH", "tariff_3")
    _, model, planner = make(priors={a: (0.2, 1e-4)}, default=(-0.3, 1e-4))
    ev = planner._cell_eval(("tariff_1", "HIGH"), 0.0, 0.0)
    t = ev.t_index["tariff_3"]
    assert (ev.top1 == t).all()
    others = [i for i in ev.rank if i != t]
    k = planner.cfg.top_targets_per_cell
    comp = planner._competitor(ev, others[: k - 1]), planner._competitor(ev, others[:k])
    pj = planner._channels.index("push")
    m_p, sd_p = model.posterior(a, "push").mean, model.posterior(a, "push").sd
    kg, imm = planner._kg(a, ev, t, pj, "push", 30, m_p, sd_p, 0.804 ** 2, comp[0], comp[1], (-math.inf, math.inf),
                          0, 1.0, 0.0, 0.0)
    assert abs(imm) < 1e-6 * 30 * 9000
    assert abs(kg) < 1.0


def test_only_allocator_top_targets_compete():
    # tariff_3 has the better lcb but lower mean; with top_targets_per_cell=1 only tariff_2 may be deployed
    priors = {("tariff_1", "HIGH", "tariff_3"): (0.04, 0.001), ("tariff_1", "HIGH", "tariff_2"): (0.05, 0.02)}
    ck = ("tariff_1", "HIGH")
    _, _, p2 = make(priors=priors, cfg=Config(top_targets_per_cell=2))
    ev2 = p2._cell_eval(ck, 0.0, 0.0)
    assert (ev2.top1 == ev2.t_index["tariff_3"]).any()
    _, _, p1 = make(priors=priors, cfg=Config(top_targets_per_cell=1))
    ev1 = p1._cell_eval(ck, 0.0, 0.0)
    assert not (ev1.top1 == ev1.t_index["tariff_3"]).any()
    assert list(ev1.elig) == [i == ev1.t_index["tariff_2"] for i in range(len(ev1.targets))]


def test_weight_promotes_arm_even_with_negative_ucb():
    cfg = Config(top_arms=1)
    b = ("tariff_2", "MID", "tariff_3")
    _, _, planner = make(default=(-0.3, 0.01), cfg=cfg)
    planner.set_weights({b: 50.0})
    assert planner._rank_arms(state()) == [b]


def test_register_result_sanitizes_bad_numbers():
    _, _, planner = make(default=(0.0, 0.2))
    spec = planner.next_pilot(state())
    planner.register_result(spec, {"n_customers": -5, "cost": float("nan")})
    assert planner._money_spent == 0.0 and planner._reach_spent == 0
    planner.register_result(spec, {"n_customers": 10, "cost": float("inf")})
    assert planner._money_spent == pytest.approx(10 * planner.dv.cost(spec.channel))


def test_max_pilots_counts_state_history():
    _, _, planner = make(default=(0.05, 0.2), cfg=Config(max_pilots=3))
    st = state(pilots_per_arm={("tariff_1", "HIGH", "tariff_2"): 3}, explore_reach_spent=300)
    assert planner.next_pilot(st) is None
    assert planner.last_debug["reason"] == "no_pilots_left"


def test_deterministic_under_slow_clock(monkeypatch):
    import agent_src.m70_planner as mod

    # More than the former 12-arm clock-independent prefix, so this catches
    # truncation that the old two-cell/four-arm fixture could not expose.
    cells = std_spec(cells=tuple((f"tariff_{i}", "HIGH") for i in range(1, 9)),
                     arpu=(9000.0,) * 8)
    _, _, fast = make(default=(0.01, 0.1), spec=cells)
    ref = fast.next_pilot(state())

    class SlowClock:
        t = 0.0

        def monotonic(self):
            SlowClock.t += 100.0
            return SlowClock.t

    monkeypatch.setattr(mod, "time", SlowClock())
    _, _, slow = make(default=(0.01, 0.1), spec=cells)
    got = slow.next_pilot(state())
    assert fast.last_debug["n_arms_evaluated"] > 12
    assert slow.last_debug["n_arms_evaluated"] == fast.last_debug["n_arms_evaluated"]
    assert (got.arm, got.channel, got.n, got.sub) == (ref.arm, ref.channel, ref.n, ref.sub)
    assert got.score == pytest.approx(ref.score)


def test_calibration_refreshes_unobserved_arm_before_next_decision():
    a = ("tariff_1", "HIGH", "tariff_3")
    _, model, planner = make(priors={a: (0.01, 0.1)})
    planner.next_pilot(state())
    assert model.n_obs(a) == 0
    before = planner._post(a, "push")
    # Like global calibration: estimates change with no new observation on a.
    model.priors[a] = (0.4, 0.04)
    planner.next_pilot(state())
    assert model.n_obs(a) == 0
    after = planner._post(a, "push")
    assert after != before
    assert after == pytest.approx((0.2, 0.02))


def test_repeat_pilot_in_used_sub_discounts_immediate():
    a = ("tariff_1", "HIGH", "tariff_3")
    spec_cells = {("tariff_1", "HIGH"): [("LITE", "LOW", 400, 9000.0)]}
    _, _, planner = make(priors={a: (0.05, 0.05)}, default=(-0.3, 1e-4), spec=spec_cells)
    fresh = planner.next_pilot(state())
    imm_fresh = planner.last_debug["immediate"]
    used = planner.next_pilot(state(used_subs={("tariff_1", "HIGH", "LITE", "LOW"): 200}, explore_reach_spent=200))
    imm_used = planner.last_debug["immediate"]
    assert (fresh.arm, fresh.channel, fresh.n, fresh.sub) == (used.arm, used.channel, used.n, used.sub)
    assert imm_fresh != 0.0
    assert imm_used == pytest.approx(0.5 * imm_fresh)
