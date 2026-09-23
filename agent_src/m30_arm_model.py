from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip

import math

import numpy as np

# Апостериорное распределение lift ratio для каждого arm по каждому каналу оценки.
#
# Модель (план, "Posterior и перенос между каналами"):
#   истинный ratio на канале c:  r_c = change * min(conv * mult_c, 1)
#   априорный базовый ratio (mult 1): mu ~= change * share, где share = c_hat (априорная оценка конверсии)
#   => prior на канале c:     r_c ~ N(mu * k(c), (sd * k(c))^2),   k(c) = min(mult_c * c_hat, 1) / c_hat
#   пилот на канале p:        y = r_p + eps,  eps ~ N(0, s^2 / n_actual)
#   перенос p -> c:           r_c = r_p * k(c, p),  k(c, p) = min(mult_c * c_hat, 1) / min(mult_p * c_hat, 1)
#
# При масштабировании вверх (k(c,p) > 1) может сработать насыщение (min(., 1)) при истинной конверсии, поэтому перенос
# делается консервативным двумя способами:
#   * среднее: *положительный* y умножается на k_eff = kappa_up * k(c, p) (дисконт к 0); отрицательные
#     данные переносятся без дисконта (y * k), чтобы вредные arm никогда не выглядели менее вредными;
#   * дисперсия: перенесённое sd = noise_sd / sqrt(n) * k(c, p) (*недисконтированный* множитель), т.е.
#     дисперсия увеличена в 1/kappa_up^2 раз относительно простого масштабирования на k_eff. Это снижает вес
#     масштабированных вверх данных, отражая неопределённость точки насыщения.
# Масштабирование вниз (k <= 1) и данные того же канала переносятся точно (k_eff = k, sd * k).
# Каждый канал оценки получает собственное гауссово сопряжённое обновление (normal-normal, известная дисперсия шума).

_AM_MIN_SHARE = 1e-3
_AM_MIN_SD = 1e-9
_AM_DEFAULT_SD = 0.25


def _am_pos_float(x: Any, default: float) -> float:
    """float(x) если конечно и > 0, иначе default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) and v > 0 else default


def _am_finite(x: Any, default: float) -> float:
    """float(x) если конечно, иначе default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class ArmModel:
    """Гауссов сопряжённый posterior lift ratio по (arm, channel) с переносом между каналами."""

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
        # калибровка prior методом эмпирического Байеса (расширение): prior mean *= beta, prior var += tau2 (базовая шкала)
        self._beta = 1.0
        self._tau2 = 0.0
        self.calibration: dict = {"beta": 1.0, "tau": 0.0, "n_arms": 0}

    # ------------------------------------------------------------------ data
    @property
    def observations(self) -> list[Observation]:
        """Все наблюдения в порядке add() (копия; изменять только через add())."""
        return list(self._obs)

    def add(self, obs: Observation) -> None:
        """Регистрирует наблюдение пилота (сбрасывает кэш posterior его arm)."""
        arm = tuple(obs.arm)
        self._obs.append(obs)
        self._by_arm.setdefault(arm, []).append(obs)
        for key in [k for k in self._cache if k[0] == arm]:
            del self._cache[key]

    def n_obs(self, arm: ArmKey) -> int:
        """Число наблюдений, записанных для arm (любой канал)."""
        return len(self._by_arm.get(tuple(arm), ()))

    def observations_for(self, arm: ArmKey) -> list[Observation]:
        """Наблюдения одного arm в порядке add() (расширение)."""
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
        """Априорная оценка конверсии c_hat для переноса между каналами (расширение)."""
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
        """k(c) = min(mult_c * c_hat, 1) / c_hat: масштаб от базового ratio (mult 1) к каналу c (расширение)."""
        s = self.share(arm)
        return min(self._mult(channel) * s, 1.0) / s

    def k_transfer(self, arm: ArmKey, eval_channel: str, pilot_channel: str) -> tuple[float, float]:
        """(k_mean, k_sd) для переноса данных с pilot_channel на eval_channel (расширение).

        k_mean включает дисконт kappa_up при масштабировании вверх (только для положительных данных, см.
        _transfer); k_sd — недисконтированный множитель.
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
        """Prior lift ratio на канале (расширение). Неизвестный arm -> Posterior(0, 0.25) на каждом канале."""
        arm = tuple(arm)
        pr = self.priors.get(arm)
        if pr is None:
            return Posterior(0.0, _AM_DEFAULT_SD)
        k = self.k_channel(arm, channel)
        mu = _am_finite(getattr(pr, "mu", None), 0.0) * self._beta
        sd = math.sqrt(_am_pos_float(getattr(pr, "sd", None), _AM_DEFAULT_SD) ** 2 + self._tau2)
        return Posterior(mu * k, max(sd * k, _AM_MIN_SD))

    def calibrate(self, min_arms: int = 3, beta_prior_sd: float = 0.5, max_tau: float = 0.25) -> dict:
        """Проверяет исторический prior по данным пилотов (расширение; эмпирический Байес, детерминированно).

        История описывает другую популяцию, поэтому prior может быть смещён или излишне уверен. По первому
        наблюдению каждого arm с историческим prior: y_i = beta·m_i + e_i, var(e_i) = k_i²(sd_i² + tau²) +
        noise_i². beta ~ N(1, beta_prior_sd²) оценивается взвешенным МНК (с обрезкой до [0, 2]); tau² — методом
        моментов по остаткам (не больше max_tau²). Меньше `min_arms` arm -> без калибровки.
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
        """Перенесённые (value, variance) одного наблюдения на eval_channel или None, если оно непригодно."""
        try:
            y = float(y)
            n = float(n)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(y) and math.isfinite(n)) or n <= 0:
            return None
        km, ks = self.k_transfer(arm, eval_channel, pilot_channel)
        if y < 0:
            km = ks  # никогда не дисконтируем вредные данные к 0
        sd = self._noise_sd / math.sqrt(n) * ks
        val = y * km
        if not math.isfinite(val) or not math.isfinite(sd):
            return None
        return val, max(sd * sd, _AM_MIN_SD ** 2)

    @staticmethod
    def _update(mean: float, var: float, y: float, obs_var: float) -> tuple[float, float]:
        """Сопряжённое normal-normal обновление при известной дисперсии наблюдения."""
        prec = 1.0 / var + 1.0 / obs_var
        new_var = 1.0 / prec
        return new_var * (mean / var + y / obs_var), new_var

    def posterior(self, arm: ArmKey, channel: str) -> Posterior:
        """Posterior lift ratio для `arm` на `channel` (с учётом множителя канала)."""
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
        """Posterior на eval_channel, если пилот на pilot_channel вернул y при n клиентах (без мутации)."""
        arm = tuple(arm)
        cur = self.posterior(arm, eval_channel)
        t = self._transfer(arm, pilot_channel, y, n, eval_channel)
        if t is None:
            return cur
        mean, var = self._update(cur.mean, max(cur.sd * cur.sd, _AM_MIN_SD ** 2), t[0], t[1])
        return Posterior(float(mean), float(math.sqrt(var)))

    def predictive_sd(self, arm: ArmKey, pilot_channel: str, n: int) -> float:
        """Sd предиктивного распределения y пилота на pilot_channel (расширение, для KG)."""
        post = self.posterior(arm, pilot_channel)
        n = max(float(n), 1.0)
        return math.sqrt(post.sd ** 2 + self._noise_sd ** 2 / n)

    # --------------------------------------------------------------- option
    def option(self, sub: SubCell, target: str, channel: str) -> Option:
        """Опция чистой ценности контакта всей sub-cell с (target, channel).

        Канал, стоимость которого нельзя прочитать (или она отрицательна / не конечна), даёт нежизнеспособную опцию
        (cost = inf, net = -inf, p_pos = 0), вместо молчаливого допущения бесплатного канала.
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
