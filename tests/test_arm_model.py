"""Tests for agent_src.m30_arm_model.ArmModel (analytic conjugate results + MC calibration)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from agent_src.contract import Config, Observation, Prior, SubCell, norm_cdf
from agent_src.m30_arm_model import ArmModel

CHANNELS = {
    "push": (0.0, 0.50),
    "sms": (4.0, 0.65),
    "digital_ads": (22.0, 0.85),
    "call": (160.0, 1.20),
}
ARM = ("tariff_1", "MID", "tariff_2")


class FakeDV:
    """Minimal DataView stand-in: channel cost / multiplier lookups."""

    channels = list(CHANNELS)

    def cost(self, ch: str) -> float:
        return CHANNELS[ch][0]

    def mult(self, ch: str) -> float:
        return CHANNELS[ch][1]


def make_model(mu=0.05, sd=0.1, share=0.3, cfg=None, arm=ARM) -> ArmModel:
    cfg = cfg or Config()
    return ArmModel(cfg, FakeDV(), {arm: Prior(mu=mu, sd=sd, share=share, n_hist=50, source="arm")})


def obs(y, n, ch="sms", arm=ARM, i=0):
    return Observation(arm=arm, channel=ch, y=y, n=n, cost=0.0, sub=None, pilot_index=i)


def conj(m0, v0, y, v):
    prec = 1 / v0 + 1 / v
    return (m0 / v0 + y / v) / prec, 1 / prec


# ------------------------------------------------------------------ prior / transfer


def test_prior_scaling_by_channel():
    m = make_model(mu=0.06, sd=0.1, share=0.3)
    for ch, (_, mult) in CHANNELS.items():
        k = min(mult * 0.3, 1.0) / 0.3
        p = m.posterior(ARM, ch)
        assert p.mean == pytest.approx(0.06 * k)
        assert p.sd == pytest.approx(0.1 * k)


def test_prior_saturation_caps_k():
    m = make_model(share=0.9)
    # call: 1.2*0.9 > 1 -> k = 1/0.9
    assert m.k_channel(ARM, "call") == pytest.approx(1 / 0.9)
    assert m.k_channel(ARM, "sms") == pytest.approx(0.65)


def test_default_share_when_prior_share_invalid():
    cfg = Config()
    m = ArmModel(cfg, FakeDV(), {ARM: Prior(0.05, 0.1, float("nan"), 0, "none")})
    assert m.share(ARM) == pytest.approx(cfg.conv_prior_default)


def test_k_transfer_up_and_down():
    m = make_model(share=0.3)
    km, ks = m.k_transfer(ARM, "call", "push")  # upscale
    k = (1.2 * 0.3) / (0.5 * 0.3)
    assert ks == pytest.approx(k)
    assert km == pytest.approx(k * 0.9)
    km, ks = m.k_transfer(ARM, "push", "call")  # downscale, exact
    assert km == pytest.approx(1 / k) and ks == pytest.approx(1 / k)
    assert m.k_transfer(ARM, "sms", "sms") == (1.0, 1.0)


# ------------------------------------------------------------------ conjugate updates


def test_same_channel_single_update_matches_conjugate():
    cfg = Config()
    m = make_model(mu=0.05, sd=0.1, share=0.3, cfg=cfg)
    k = 0.65
    m.add(obs(0.12, 100, "sms"))
    exp_m, exp_v = conj(0.05 * k, (0.1 * k) ** 2, 0.12, cfg.noise_sd ** 2 / 100)
    p = m.posterior(ARM, "sms")
    assert p.mean == pytest.approx(exp_m)
    assert p.sd == pytest.approx(math.sqrt(exp_v))
    assert m.n_obs(ARM) == 1 and len(m.observations) == 1


def test_two_observations_equal_pooled():
    """Sequential updates == one update with pooled mean and n1+n2."""
    cfg = Config()
    m = make_model(cfg=cfg)
    m.add(obs(0.10, 60, "sms", i=0))
    m.add(obs(0.20, 140, "sms", i=1))
    pooled = (0.10 * 60 + 0.20 * 140) / 200
    k = 0.65
    exp_m, exp_v = conj(0.05 * k, (0.1 * k) ** 2, pooled, cfg.noise_sd ** 2 / 200)
    p = m.posterior(ARM, "sms")
    assert p.mean == pytest.approx(exp_m)
    assert p.sd == pytest.approx(math.sqrt(exp_v))


def test_downscale_transfer_consistent_with_latent_model():
    """Without upscaling, evaluating on a lower channel equals latent-base update then scale."""
    cfg = Config()
    m = make_model(mu=0.05, sd=0.1, share=0.3, cfg=cfg)
    m.add(obs(0.3, 100, "call"))
    kc, kp = 0.65, 1.2  # k(sms), k(call) at share .3 (no saturation)
    # base theta: prior N(.05,.1^2); obs y/kp with sd s/sqrt(n)/kp
    bm, bv = conj(0.05, 0.01, 0.3 / kp, (cfg.noise_sd / 10 / kp) ** 2)
    p = m.posterior(ARM, "sms")
    assert p.mean == pytest.approx(bm * kc)
    assert p.sd == pytest.approx(math.sqrt(bv) * kc)


def test_upscale_is_discounted_and_variance_inflated():
    cfg = Config()
    m = make_model(mu=0.0, sd=0.1, share=0.3, cfg=cfg)
    m.add(obs(0.2, 100, "push"))
    k = 1.2 / 0.5
    pr_m, pr_v = 0.0, (0.1 * 1.2) ** 2
    exp_m, exp_v = conj(pr_m, pr_v, 0.2 * k * cfg.kappa_up, (cfg.noise_sd / 10 * k) ** 2)
    p = m.posterior(ARM, "call")
    assert p.mean == pytest.approx(exp_m)
    assert p.sd == pytest.approx(math.sqrt(exp_v))
    # Evidence weight is lower than an undiscounted exact transfer would give.
    m_nodisc = make_model(mu=0.0, sd=0.1, share=0.3, cfg=cfg.replace(kappa_up=1.0))
    m_nodisc.add(obs(0.2, 100, "push"))
    assert p.mean < m_nodisc.posterior(ARM, "call").mean


def test_invalid_observation_ignored():
    m = make_model()
    before = m.posterior(ARM, "sms")
    m.add(obs(float("nan"), 100))
    m.add(obs(0.3, 0))
    after = m.posterior(ARM, "sms")
    assert after.mean == pytest.approx(before.mean) and after.sd == pytest.approx(before.sd)
    assert m.n_obs(ARM) == 2


def test_cache_invalidated_on_add():
    m = make_model()
    p0 = m.posterior(ARM, "sms")
    m.add(obs(0.5, 200))
    p1 = m.posterior(ARM, "sms")
    assert p1.mean > p0.mean and p1.sd < p0.sd


def test_posterior_after_is_pure_and_matches_add():
    m = make_model()
    m.add(obs(0.1, 50, "sms"))
    snap = (m.posterior(ARM, "call").mean, m.posterior(ARM, "call").sd, len(m.observations))
    hyp = m.posterior_after(ARM, "digital_ads", 0.15, 120, "call")
    assert (m.posterior(ARM, "call").mean, m.posterior(ARM, "call").sd, len(m.observations)) == snap
    m.add(obs(0.15, 120, "digital_ads", i=1))
    real = m.posterior(ARM, "call")
    assert hyp.mean == pytest.approx(real.mean) and hyp.sd == pytest.approx(real.sd)


def test_unknown_arm_posterior_and_option():
    m = make_model()
    other = ("tariff_9", "LOW", "tariff_1")
    for ch in CHANNELS:
        p = m.posterior(other, ch)
        assert p.mean == 0.0 and p.sd == pytest.approx(0.25)
    sub = SubCell(key=("tariff_9", "LOW", "LITE", "LOW"), n=10, sum_p=1000.0, mean_p=100.0,
                  ids=np.arange(10), p=np.full(10, 100.0))
    o = m.option(sub, "tariff_1", "push")
    assert o.net_mean == 0.0 and o.net_sd == pytest.approx(m.posterior(other, "push").sd * 1000.0)
    assert o.p_pos == pytest.approx(0.5)
    # An unusable observation must not switch the unknown arm to a different convention.
    m.add(obs(float("nan"), 50, "push", arm=other))
    o2 = m.option(sub, "tariff_1", "push")
    assert o2.net_sd == pytest.approx(o.net_sd)


def test_upscale_negative_evidence_not_discounted():
    cfg = Config()
    m = make_model(mu=0.0, sd=0.1, share=0.3, cfg=cfg)
    m.add(obs(-0.2, 100, "push"))
    k = 1.2 / 0.5
    exp_m, exp_v = conj(0.0, (0.1 * 1.2) ** 2, -0.2 * k, (cfg.noise_sd / 10 * k) ** 2)
    p = m.posterior(ARM, "call")
    assert p.mean == pytest.approx(exp_m) and p.sd == pytest.approx(math.sqrt(exp_v))
    m_nodisc = make_model(mu=0.0, sd=0.1, share=0.3, cfg=cfg.replace(kappa_up=1.0))
    m_nodisc.add(obs(-0.2, 100, "push"))
    assert p.mean == pytest.approx(m_nodisc.posterior(ARM, "call").mean)
    hyp = make_model(mu=0.0, sd=0.1, share=0.3, cfg=cfg).posterior_after(ARM, "push", -0.2, 100, "call")
    assert hyp.mean == pytest.approx(p.mean) and hyp.sd == pytest.approx(p.sd)


def test_malformed_prior_and_config_values_are_safe():
    cfg = Config().replace(noise_sd=float("nan"), kappa_up=float("inf"))
    m = ArmModel(cfg, FakeDV(), {ARM: Prior(mu=None, sd=float("nan"), share=0.3, n_hist=0, source="x")})
    m.add(obs(0.1, 100, "push"))
    for ch in CHANNELS:
        p = m.posterior(ARM, ch)
        assert math.isfinite(p.mean) and math.isfinite(p.sd) and p.sd > 0
        assert math.isfinite(m.predictive_sd(ARM, ch, 50))
    km, _ = m.k_transfer(ARM, "call", "push")
    assert km <= m.k_transfer(ARM, "call", "push")[1]


def test_observations_property_is_a_copy():
    m = make_model()
    m.add(obs(0.1, 100))
    m.observations.append(obs(9.0, 100))
    assert len(m.observations) == 1 and m.n_obs(ARM) == 1


def test_option_unknown_channel_cost_is_not_viable():
    class BadDV(FakeDV):
        def cost(self, ch):
            raise KeyError(ch)

    m = ArmModel(Config(), BadDV(), {ARM: Prior(0.5, 0.01, 0.3, 50, "arm")})
    sub = SubCell(key=("tariff_1", "MID", "LITE", "LOW"), n=5, sum_p=5000.0, mean_p=1000.0,
                  ids=np.arange(5), p=np.full(5, 1000.0))
    o = m.option(sub, "tariff_2", "sms")
    assert o.net_lcb == -math.inf and o.p_pos == 0.0 and o.cost == math.inf


# ------------------------------------------------------------------ option


def test_option_formulae():
    cfg = Config()
    m = make_model(cfg=cfg)
    m.add(obs(0.2, 200, "sms"))
    sub = SubCell(key=("tariff_1", "MID", "LITE", "LOW"), n=50, sum_p=50 * 3000.0, mean_p=3000.0,
                  ids=np.arange(50), p=np.full(50, 3000.0))
    o = m.option(sub, "tariff_2", "sms")
    post = m.posterior(ARM, "sms")
    net = post.mean * sub.sum_p - 4.0 * 50
    assert o.n == 50 and o.cost == pytest.approx(200.0)
    assert o.net_mean == pytest.approx(net)
    assert o.net_sd == pytest.approx(post.sd * sub.sum_p)
    assert o.net_lcb == pytest.approx(net - cfg.z_risk * o.net_sd)
    assert o.p_pos == pytest.approx(norm_cdf(net / o.net_sd))
    assert o.sub == sub.key and o.target == "tariff_2" and o.channel == "sms"


def test_option_zero_sd_edge():
    m = make_model()
    sub = SubCell(key=("tariff_1", "MID", "LITE", "LOW"), n=0, sum_p=0.0, mean_p=0.0,
                  ids=np.arange(0), p=np.zeros(0))
    o = m.option(sub, "tariff_2", "call")
    assert o.net_sd == 0.0 and o.p_pos == 0.0


# ------------------------------------------------------------------ Monte Carlo calibration


def _coverage(eval_ch, pilot_ch, trials=3000, share=0.3, conv=0.3, seed=0, cfg=None):
    """Fraction of trials where truth on eval_ch lies in the 90% posterior interval."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    mu0, sd0 = 0.05, 0.1
    z = 1.6448536269514722
    hits = 0
    for _ in range(trials):
        theta = rng.normal(mu0, sd0)  # base ratio at mult 1: change * conv
        change = theta / conv

        def r(ch):
            return change * min(conv * CHANNELS[ch][1], 1.0)

        m = make_model(mu=mu0, sd=sd0, share=share, cfg=cfg)
        for i in range(2):
            n = int(rng.integers(30, 201))
            y = r(pilot_ch) + rng.normal(0, cfg.noise_sd / math.sqrt(n))
            m.add(obs(y, n, pilot_ch, i=i))
        p = m.posterior(ARM, eval_ch)
        hits += abs(r(eval_ch) - p.mean) <= z * p.sd
    return hits / trials


def test_mc_calibration_same_channel():
    cov = _coverage("sms", "sms")
    assert abs(cov - 0.90) < 0.02


def test_mc_calibration_downscale_transfer():
    cov = _coverage("push", "call")
    assert abs(cov - 0.90) < 0.02


def test_mc_calibration_upscale_transfer_not_badly_undercovered():
    cov = _coverage("call", "push")
    assert 0.85 < cov < 0.97  # conservative, but not arbitrarily wide


def test_mc_calibration_upscale_under_true_saturation():
    """True conversion saturates on call (share prior is wrong): kappa_up keeps coverage >= nominal."""
    cov = _coverage("call", "push", share=0.3, conv=1.0)
    assert 0.88 < cov < 0.99
    # Control: without the upscale discount, coverage under saturation is worse.
    cov_nodisc = _coverage("call", "push", share=0.3, conv=1.0, cfg=Config().replace(kappa_up=1.0))
    assert cov_nodisc <= cov
