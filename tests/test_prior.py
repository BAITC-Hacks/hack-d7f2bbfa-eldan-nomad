"""Tests for agent_src.m20_prior.PriorBuilder (fake DataView, synthetic history)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_src.contract import Config, Prior
from agent_src.m20_prior import PriorBuilder

REPO = Path(__file__).resolve().parent.parent


class FakeDV:
    """Minimal DataView stand-in: cells, targets_for, tariff_price."""

    def __init__(self, cells: list[tuple[str, str]], prices: dict[str, float]):
        self.cells = {c: object() for c in cells}
        self.tariff_price = dict(prices)
        self.tariff_codes = sorted(prices, key=lambda s: int(s.split("_")[1]))

    def targets_for(self, cell):
        return [t for t in self.tariff_codes if t != cell[0]]


PRICES = {"tariff_1": 0.0, "tariff_2": 3140.0, "tariff_3": 5990.0, "tariff_4": 9990.0}
CELLS = [("tariff_1", "LOW"), ("tariff_2", "MID"), ("tariff_3", "HIGH"), ("tariff_9", "MID")]


def _rows(frm, to, prev, nxt, n):
    return [{"TIME_KEY": "2026-10-01", "AVG_ARPU_PREV_3M": prev, "AVG_ARPU_NEXT_3M": nxt,
             "ID_NUMBER": i, "tariff_plan_code_from": frm, "tariff_plan_code_to": to} for i in range(n)]


def _history(n_up: int = 30, nxt_up: float = 3000.0) -> pd.DataFrame:
    rows = []
    rows += _rows("tariff_2", "tariff_3", 2000.0, nxt_up, n_up)          # MID, +50%
    rows += _rows("tariff_2", "tariff_1", 2000.0, 1500.0, 10)            # MID, -25%
    rows += _rows("tariff_2", "tariff_4", 2000.0, 2000.0, 60)            # MID, 0
    rows += _rows("tariff_1", "tariff_2", 500.0, 600.0, 5)               # LOW
    rows += _rows("tariff_1", "tariff_2", 50.0, 5000.0, 5)               # PREV < 100 -> dropped
    return pd.DataFrame(rows)


@pytest.fixture()
def dv():
    return FakeDV(CELLS, PRICES)


def _expected_arms(dv):
    return {(c[0], c[1], t) for c in dv.cells for t in dv.targets_for(c)}


def test_covers_every_arm_no_history(dv):
    pr = PriorBuilder(Config()).build(None, dv)
    assert set(pr) == _expected_arms(dv)
    for arm, p in pr.items():
        assert isinstance(p, Prior)
        assert arm[0] != arm[2]
        assert p.sd == pytest.approx(0.25)
        assert 0 < p.share <= 1
        assert p.n_hist == 0


def test_no_history_price_sign(dv):
    pr = PriorBuilder(Config()).build(None, dv)
    up, down = pr[("tariff_2", "MID", "tariff_4")], pr[("tariff_2", "MID", "tariff_1")]
    assert up.mu > 0 > down.mu
    assert abs(up.mu) < 0.05 and up.source == "price"
    unknown = pr[("tariff_9", "MID", "tariff_1")]  # tariff_9 has no price
    assert unknown.mu == 0.0 and unknown.source == "none"


def test_bad_history_falls_back(dv):
    for bad in (pd.DataFrame(), pd.DataFrame({"x": [1]}), "garbage", _history().iloc[:0]):
        pr = PriorBuilder(Config()).build(bad, dv)
        assert set(pr) == _expected_arms(dv)
        assert all(p.sd == pytest.approx(0.25) for p in pr.values())


def test_history_sources_and_counts(dv):
    pr = PriorBuilder(Config()).build(_history(), dv)
    assert set(pr) == _expected_arms(dv)
    a = pr[("tariff_2", "MID", "tariff_3")]
    assert a.source == "arm" and a.n_hist == 30
    assert pr[("tariff_1", "LOW", "tariff_2")].n_hist == 5  # PREV<100 rows dropped
    assert pr[("tariff_1", "LOW", "tariff_3")].source == "cell"
    assert pr[("tariff_3", "HIGH", "tariff_1")].source == "global"
    for p in pr.values():
        assert math.isfinite(p.mu) and math.isfinite(p.sd) and p.sd >= Config().tau - 1e-9
        assert 0 < p.share <= 1


def test_share_beta_mean_and_mu_formula(dv):
    cfg = Config()
    pr = PriorBuilder(cfg).build(_history(), dv)
    a = pr[("tariff_2", "MID", "tariff_3")]
    tot, c = 100, 30
    assert a.share == pytest.approx((c + 1) / (tot + 2))
    # shrunk change lies strictly between the arm mean (0.5) and the cell mean
    m = a.mu / a.share
    cell_mean = (30 * 0.5 + 10 * -0.25 + 60 * 0.0) / 100
    assert cell_mean < m < 0.5


def test_monotonic_in_observed_change(dv):
    b = PriorBuilder(Config())
    mus = [b.build(_history(nxt_up=x), dv)[("tariff_2", "MID", "tariff_3")].mu for x in (1000.0, 2000.0, 3000.0, 6000.0)]
    assert all(x < y for x, y in zip(mus, mus[1:]))


def test_monotonic_in_count(dv):
    b = PriorBuilder(Config(tau=0.0))
    ps = [b.build(_history(n_up=n), dv)[("tariff_2", "MID", "tariff_3")] for n in (5, 20, 80, 300)]
    shares = [p.share for p in ps]
    assert all(x < y for x, y in zip(shares, shares[1:]))
    # more evidence -> change estimate closer to the arm mean 0.5
    ms = [p.mu / p.share for p in ps]
    assert all(x < y for x, y in zip(ms, ms[1:])) and ms[-1] < 0.5


def test_shrink_k_pulls_to_cell(dv):
    arm = ("tariff_2", "MID", "tariff_3")
    m_lo = PriorBuilder(Config(shrink_k=1.0)).build(_history(), dv)[arm]
    m_hi = PriorBuilder(Config(shrink_k=500.0)).build(_history(), dv)[arm]
    assert m_hi.mu / m_hi.share < m_lo.mu / m_lo.share


def test_tau_sets_sd_floor(dv):
    arm = ("tariff_2", "MID", "tariff_3")
    s0 = PriorBuilder(Config(tau=0.0)).build(_history(), dv)[arm].sd
    s1 = PriorBuilder(Config(tau=0.2)).build(_history(), dv)[arm].sd
    assert s1 == pytest.approx(math.sqrt(s0 ** 2 + 0.04))


def test_deterministic(dv):
    a = PriorBuilder(Config()).build(_history(), dv)
    b = PriorBuilder(Config()).build(_history().sample(frac=1.0, random_state=3), dv)
    assert list(a.items()) == list(b.items())  # bit-exact, same order


def test_deterministic_real_history_shuffled():
    path = REPO / "data" / "change_tariff.csv"
    if not path.exists():
        pytest.skip("no history file")
    hist = pd.read_csv(path)
    codes = sorted(set(hist["tariff_plan_code_from"]))
    rdv = FakeDV([(t, s) for t in codes for s in ("LOW", "MID", "HIGH")], {t: 1.0 for t in codes})
    a = PriorBuilder(Config()).build(hist, rdv)
    b = PriorBuilder(Config()).build(hist.sample(frac=1.0, random_state=7), rdv)
    assert list(a.items()) == list(b.items())


def test_exact_sd_formula_with_within_arm_variance(dv):
    cfg = Config()
    k, tau = cfg.shrink_k, cfg.tau
    rows = _rows("tariff_2", "tariff_3", 2000.0, 3000.0, 20) + _rows("tariff_2", "tariff_3", 2000.0, 1000.0, 10)
    rows += _rows("tariff_2", "tariff_4", 2000.0, 2000.0, 70)
    pr = PriorBuilder(cfg).build(pd.DataFrame(rows), dv)[("tariff_2", "MID", "tariff_3")]
    chg = np.array([0.5] * 20 + [-0.5] * 10 + [0.0] * 70)
    g_mean, g_var = chg.mean(), chg.var()
    arm = chg[:30]
    c, tot = 30, 100
    m_cell_s = (tot * chg.mean() + k * g_mean) / (tot + k)
    m_s = (c * arm.mean() + k * m_cell_s) / (c + k)
    s2 = ((c - 1) * arm.var(ddof=1) + k * g_var) / (c - 1 + k)
    var_m = s2 / (c + k)
    a_, b_ = c + 1.0, tot - c + 1.0
    share = a_ / (a_ + b_)
    var_share = a_ * b_ / ((a_ + b_) ** 2 * (a_ + b_ + 1))
    assert pr.share == pytest.approx(share)
    assert pr.mu == pytest.approx(m_s * share)
    assert pr.sd == pytest.approx(math.sqrt(var_m * share ** 2 + m_s ** 2 * var_share + tau ** 2))


def test_unobserved_arm_uses_global_share_and_same_price_shift(dv):
    arm = ("tariff_1", "LOW", "tariff_3")
    pr = PriorBuilder(Config(tau=0.0)).build(_history(), dv)[arm]
    g_share = float(np.median([30 / 100, 10 / 100, 60 / 100, 5 / 5]))
    assert pr.n_hist == 0 and pr.source == "cell" and pr.share == pytest.approx(g_share)
    # price shift is added in base-ratio units: same +0.02 as the no-history prior
    flat = FakeDV(CELLS, {**PRICES, "tariff_3": PRICES["tariff_1"]})
    pr_flat = PriorBuilder(Config(tau=0.0)).build(_history(), flat)[arm]
    assert pr.mu - pr_flat.mu == pytest.approx(0.02)
    assert PriorBuilder(Config()).build(None, dv)[arm].mu == pytest.approx(0.02)


@pytest.mark.parametrize("col,val", [("AVG_ARPU_NEXT_3M", np.inf), ("AVG_ARPU_NEXT_3M", -np.inf),
                                     ("AVG_ARPU_PREV_3M", np.inf), ("AVG_ARPU_NEXT_3M", np.nan),
                                     ("AVG_ARPU_PREV_3M", "abc")])
def test_non_finite_history_rows_dropped(dv, col, val):
    base = _history()
    bad = pd.DataFrame(_rows("tariff_2", "tariff_3", 2000.0, 3000.0, 50))
    bad[col] = val
    pr_clean = PriorBuilder(Config()).build(base, dv)
    pr_bad = PriorBuilder(Config()).build(pd.concat([base, bad], ignore_index=True), dv)
    assert list(pr_clean.items()) == list(pr_bad.items())


@pytest.mark.parametrize("over", [dict(tau=float("nan")), dict(tau=float("inf")), dict(shrink_k=float("nan")),
                                  dict(conv_prior_default=float("nan")), dict(conv_prior_default=0.0),
                                  dict(conv_prior_default=5.0), dict(conv_prior_default="x"), dict(tau=-1.0)])
@pytest.mark.parametrize("hist", [None, "hist"])
def test_bad_config_never_raises_and_valid(dv, over, hist):
    h = _history() if hist else None
    pr = PriorBuilder(Config(**over)).build(h, dv)
    assert set(pr) == _expected_arms(dv)
    for p in pr.values():
        assert math.isfinite(p.mu) and math.isfinite(p.sd) and p.sd > 0
        assert 0 < p.share <= 1


def test_broken_dv_never_raises():
    class Broken:
        cells = {("tariff_1", "LOW"): None}

        def targets_for(self, cell):
            raise RuntimeError("boom")

    assert PriorBuilder(Config()).build(_history(), Broken()) == {}


def test_real_history_file(dv):
    path = REPO / "data" / "change_tariff.csv"
    if not path.exists():
        pytest.skip("no history file")
    hist = pd.read_csv(path)
    tariffs = pd.read_csv(REPO / "tariff_dictionary.csv")
    prices = dict(zip(tariffs["tariff_plan_code"], tariffs["price_tariff"]))
    codes = sorted(prices)
    cells = [(t, s) for t in codes for s in ("LOW", "MID", "HIGH")]
    rdv = FakeDV(cells, prices)
    pr = PriorBuilder(Config()).build(hist, rdv)
    assert len(pr) == len(cells) * (len(codes) - 1)
    srcs = {p.source for p in pr.values()}
    assert "arm" in srcs
    mus = np.array([p.mu for p in pr.values()])
    assert np.isfinite(mus).all() and np.abs(mus).max() < 3.0
