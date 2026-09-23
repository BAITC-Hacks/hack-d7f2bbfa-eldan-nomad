from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip

import math

import numpy as np

# Posterior of the per-arm lift ratio, per evaluation channel.
#
# Model (plan, "Posterior и перенос между каналами"):
#   true ratio on channel c:  r_c = change * min(conv * mult_c, 1)
#   prior base ratio (mult 1): mu ~= change * share, with share = c_hat (prior conversion estimate)
#   => prior on channel c:     r_c ~ N(mu * k(c), (sd * k(c))^2),   k(c) = min(mult_c * c_hat, 1) / c_hat
#   pilot on channel p:        y = r_p + eps,  eps ~ N(0, s^2 / n_actual)
#   transfer p -> c:           r_c = r_p * k(c, p),  k(c, p) = min(mult_c * c_hat, 1) / min(mult_p * c_hat, 1)
#
# Upscaling (k(c,p) > 1) is where saturation (min(., 1)) may bite with the true conversion, so the transfer
# is made conservative in two ways:
#   * mean: a *positive* y is multiplied by k_eff = kappa_up * k(c, p) (discount towards 0); negative
#     evidence is transferred undiscounted (y * k) so harmful arms are never made to look less harmful;
#   * variance: the transferred sd is noise_sd / sqrt(n) * k(c, p) (the *undiscounted* factor), i.e. the
#     variance is inflated by 1/kappa_up^2 relative to a plain rescale by k_eff. This lowers the weight of
#     upscaled evidence, reflecting uncertainty about the saturation point.
# Downscaling (k <= 1) and same-channel evidence are transferred exactly (k_eff = k, sd * k).
# Each evaluation channel gets its own Gaussian conjugate update (normal-normal, known noise variance).

_AM_MIN_SHARE = 1e-3
_AM_MIN_SD = 1e-9
_AM_DEFAULT_SD = 0.25


def _am_pos_float(x: Any, default: float) -> float:
    """float(x) if finite and > 0, else default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) and v > 0 else default


def _am_finite(x: Any, default: float) -> float:
    """float(x) if finite, else default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class ArmModel:
    """Gaussian conjugate posterior of lift ratio per (arm, channel) with cross-channel transfer."""

    def __init__(self, cfg: Config, dv: Any, priors: dict[ArmKey, Prior]):
        self.cfg = cfg
        self.dv = dv
        self.priors: dict[ArmKey, Prior] = dict(priors or {})
        self._obs: list[Observation] = []
        self._by_arm: dict[ArmKey, list[Observation]] = {}
        self._cache: dict[tuple[ArmKey, str], Posterior] = {}
        self._mult_cache: dict[str, float] = {}
        self._noise_sd = _am_pos_float(getattr(cfg, "noise_sd", None), 0.804)
        kap = _am_pos_float(getattr(cfg, "kappa_up", None), 1.0)
        self._kappa_up = min(kap, 1.0)
        # empirical-Bayes prior calibration (extension): prior mean *= beta, prior var += tau2 (base scale)
        self._beta = 1.0
        self._tau2 = 0.0
        self.calibration: dict = {"beta": 1.0, "tau": 0.0, "n_arms": 0}

    # ------------------------------------------------------------------ data
    @property
    def observations(self) -> list[Observation]:
        """All observations in add() order (a copy; mutate via add() only)."""
        return list(self._obs)

    def add(self, obs: Observation) -> None:
        """Register a pilot observation (invalidates cached posteriors of its arm)."""
        arm = tuple(obs.arm)
        self._obs.append(obs)
        self._by_arm.setdefault(arm, []).append(obs)
        for key in [k for k in self._cache if k[0] == arm]:
            del self._cache[key]

    def n_obs(self, arm: ArmKey) -> int:
        """Number of observations recorded for an arm (any channel)."""
        return len(self._by_arm.get(tuple(arm), ()))

    def observations_for(self, arm: ArmKey) -> list[Observation]:
        """Observations of one arm, in add() order (extension)."""
        return list(self._by_arm.get(tuple(arm), ()))

    # ------------------------------------------------------------- transfer
    def _mult(self, ch: str) -> float:
        m = self._mult_cache.get(ch)
        if m is None:
            try:
                m = float(self.dv.mult(ch))
            except Exception:
                m = 1.0
            if not math.isfinite(m) or m <= 0:
                m = 1.0
            self._mult_cache[ch] = m
        return m

    def share(self, arm: ArmKey) -> float:
        """Prior conversion estimate c_hat used for channel transfer (extension)."""
        pr = self.priors.get(tuple(arm))
        s = pr.share if pr is not None else self.cfg.conv_prior_default
        try:
            s = float(s)
        except (TypeError, ValueError):
            s = self.cfg.conv_prior_default
        if not math.isfinite(s) or s <= 0:
            s = self.cfg.conv_prior_default
        return min(max(s, _AM_MIN_SHARE), 1.0)

    def k_channel(self, arm: ArmKey, channel: str) -> float:
        """k(c) = min(mult_c * c_hat, 1) / c_hat: scale from base ratio (mult 1) to channel c (extension)."""
        s = self.share(arm)
        return min(self._mult(channel) * s, 1.0) / s

    def k_transfer(self, arm: ArmKey, eval_channel: str, pilot_channel: str) -> tuple[float, float]:
        """(k_mean, k_sd) for moving evidence from pilot_channel to eval_channel (extension).

        k_mean includes the kappa_up discount when upscaling (applied to positive evidence only, see
        _transfer); k_sd is the undiscounted factor.
        """
        if eval_channel == pilot_channel:
            return 1.0, 1.0
        s = self.share(arm)
        k = min(self._mult(eval_channel) * s, 1.0) / min(self._mult(pilot_channel) * s, 1.0)
        if k > 1.0 + 1e-12:
            return k * self._kappa_up, k
        return k, k

    # ------------------------------------------------------------ posterior
    def prior_on(self, arm: ArmKey, channel: str) -> Posterior:
        """Prior of the lift ratio on a channel (extension). Unknown arm -> Posterior(0, 0.25) on every channel."""
        arm = tuple(arm)
        pr = self.priors.get(arm)
        if pr is None:
            return Posterior(0.0, _AM_DEFAULT_SD)
        k = self.k_channel(arm, channel)
        mu = _am_finite(getattr(pr, "mu", None), 0.0) * self._beta
        sd = math.sqrt(_am_pos_float(getattr(pr, "sd", None), _AM_DEFAULT_SD) ** 2 + self._tau2)
        return Posterior(mu * k, max(sd * k, _AM_MIN_SD))

    def calibrate(self, min_arms: int = 3, beta_prior_sd: float = 0.5, max_tau: float = 0.25) -> dict:
        """Check the history prior against pilot evidence (extension; empirical Bayes, deterministic).

        The history describes another population, so the prior may be biased or overconfident. Using the first
        observation of every arm with a history prior: y_i = beta·m_i + e_i, var(e_i) = k_i²(sd_i² + tau²) +
        noise_i². beta ~ N(1, beta_prior_sd²) is fitted by weighted least squares (clipped to [0, 2]); tau² by the
        method of moments on the residuals (capped at max_tau²). Fewer than `min_arms` arms -> no calibration.
        """
        rows = []
        for arm in sorted(self._by_arm):
            pr = self.priors.get(arm)
            obs = self._by_arm[arm]
            if pr is None or not obs:
                continue
            ob = obs[0]
            try:
                y, n = float(ob.y), float(ob.n)
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(y) and math.isfinite(n)) or n <= 0:
                continue
            k = self.k_channel(arm, ob.channel)
            m = _am_finite(getattr(pr, "mu", None), 0.0) * k
            s2 = (_am_pos_float(getattr(pr, "sd", None), _AM_DEFAULT_SD) * k) ** 2
            rows.append((y, m, s2, self._noise_sd ** 2 / n, k * k))
        beta, tau2 = 1.0, 0.0
        if len(rows) >= max(int(min_arms), 1):
            w0 = 1.0 / max(beta_prior_sd, 1e-6) ** 2
            num, den = w0, w0
            for y, m, s2, nv, _k2 in rows:
                v = s2 + nv
                num += m * y / v
                den += m * m / v
            beta = min(max(num / den, 0.0), 2.0)
            r2 = sum((y - beta * m) ** 2 for y, m, _s2, _nv, _k2 in rows)
            ev = sum(s2 + nv for _y, _m, s2, nv, _k2 in rows)
            k2 = sum(k2 for *_x, k2 in rows)
            if k2 > 0:
                tau2 = min(max((r2 - ev) / k2, 0.0), max_tau * max_tau)
        if beta != self._beta or tau2 != self._tau2:
            self._beta, self._tau2 = beta, tau2
            self._cache.clear()
        self.calibration = {"beta": round(beta, 4), "tau": round(math.sqrt(tau2), 4), "n_arms": len(rows)}
        return dict(self.calibration)

    def _transfer(self, arm: ArmKey, pilot_channel: str, y: float, n: float,
                  eval_channel: str) -> tuple[float, float] | None:
        """Transferred (value, variance) of one observation on eval_channel, or None if unusable."""
        try:
            y = float(y)
            n = float(n)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(y) and math.isfinite(n)) or n <= 0:
            return None
        km, ks = self.k_transfer(arm, eval_channel, pilot_channel)
        if y < 0:
            km = ks  # never discount harmful evidence towards 0
        sd = self._noise_sd / math.sqrt(n) * ks
        val = y * km
        if not math.isfinite(val) or not math.isfinite(sd):
            return None
        return val, max(sd * sd, _AM_MIN_SD ** 2)

    @staticmethod
    def _update(mean: float, var: float, y: float, obs_var: float) -> tuple[float, float]:
        """Normal-normal conjugate update with known observation variance."""
        prec = 1.0 / var + 1.0 / obs_var
        new_var = 1.0 / prec
        return new_var * (mean / var + y / obs_var), new_var

    def posterior(self, arm: ArmKey, channel: str) -> Posterior:
        """Posterior of the lift ratio of `arm` on `channel` (channel multiplier included)."""
        arm = tuple(arm)
        key = (arm, channel)
        hit = self._cache.get(key)
        if hit is not None:
            return Posterior(hit.mean, hit.sd)
        pr = self.prior_on(arm, channel)
        mean, var = pr.mean, pr.sd * pr.sd
        for ob in self._by_arm.get(arm, ()):
            t = self._transfer(arm, ob.channel, ob.y, ob.n, channel)
            if t is not None:
                mean, var = self._update(mean, var, t[0], t[1])
        post = Posterior(float(mean), float(math.sqrt(max(var, 0.0))))
        self._cache[key] = post
        return Posterior(post.mean, post.sd)

    def posterior_after(self, arm: ArmKey, pilot_channel: str, y: float, n: int,
                        eval_channel: str) -> Posterior:
        """Posterior on eval_channel if a pilot on pilot_channel returned y with n customers (no mutation)."""
        arm = tuple(arm)
        cur = self.posterior(arm, eval_channel)
        t = self._transfer(arm, pilot_channel, y, n, eval_channel)
        if t is None:
            return cur
        mean, var = self._update(cur.mean, max(cur.sd * cur.sd, _AM_MIN_SD ** 2), t[0], t[1])
        return Posterior(float(mean), float(math.sqrt(var)))

    def predictive_sd(self, arm: ArmKey, pilot_channel: str, n: int) -> float:
        """Sd of the predictive distribution of a pilot's y on pilot_channel (extension, for KG)."""
        post = self.posterior(arm, pilot_channel)
        n = max(float(n), 1.0)
        return math.sqrt(post.sd ** 2 + self._noise_sd ** 2 / n)

    # --------------------------------------------------------------- option
    def option(self, sub: SubCell, target: str, channel: str) -> Option:
        """Net-value option of contacting the whole sub-cell with (target, channel).

        A channel whose cost cannot be read (or is negative / non-finite) yields a non-viable option
        (cost = inf, net = -inf, p_pos = 0) instead of silently assuming a free channel.
        """
        key = tuple(sub.key)
        arm = (key[0], key[1], target)
        post = self.posterior(arm, channel)
        n = int(sub.n)
        try:
            cost_ch = float(self.dv.cost(channel))
        except Exception:
            cost_ch = math.nan
        if not math.isfinite(cost_ch) or cost_ch < 0:
            return Option(sub=key, target=target, channel=channel, n=n, cost=math.inf,
                          net_mean=-math.inf, net_sd=0.0, net_lcb=-math.inf, p_pos=0.0)
        cost = cost_ch * n
        sum_p = _am_finite(sub.sum_p, 0.0)
        net = post.mean * sum_p - cost
        net_sd = abs(post.sd * sum_p)
        net_lcb = net - float(self.cfg.z_risk) * net_sd
        if net_sd > 0:
            p_pos = norm_cdf(net / net_sd)
        else:
            p_pos = 1.0 if net > 0 else 0.0
        return Option(sub=key, target=target, channel=channel, n=n, cost=cost,
                      net_mean=float(net), net_sd=float(net_sd), net_lcb=float(net_lcb), p_pos=float(p_pos))
