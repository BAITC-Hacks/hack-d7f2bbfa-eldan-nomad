"""Fakes for PilotPlanner tests: DataView, Gaussian ArmModel, Allocator with fixed shadow prices."""
from __future__ import annotations

import math

import numpy as np

from agent_src.contract import Cell, Observation, Posterior, SubCell

FAKE_CHANNELS = {
    "push": (0.0, 0.50),
    "sms": (4.0, 0.65),
    "digital_ads": (22.0, 0.85),
    "call": (160.0, 1.20),
}


class FakeDV:
    """Minimal DataView: cells of given sub sizes/arpu, all tariffs as targets."""

    def __init__(self, spec: dict, tariffs: list[str]):
        """spec: {(tariff, arpu): [(data, call, n, mean_p), ...]}."""
        self.channels = list(FAKE_CHANNELS)
        self.tariff_codes = list(tariffs)
        self.cells: dict = {}
        self.subs: dict = {}
        next_id = 1
        for ck, rows in spec.items():
            keys = []
            for data, call, n, mp in rows:
                key = (ck[0], ck[1], data, call)
                ids = np.arange(next_id, next_id + n, dtype=np.int64)
                next_id += n
                p = np.full(n, float(mp))
                self.subs[key] = SubCell(key, n, float(p.sum()), float(mp), ids, p)
                keys.append(key)
            keys.sort()
            n_tot = sum(self.subs[k].n for k in keys)
            sp = sum(self.subs[k].sum_p for k in keys)
            self.cells[ck] = Cell(ck, n_tot, sp, sp / max(n_tot, 1), keys)

    def cost(self, ch: str) -> float:
        return FAKE_CHANNELS[ch][0]

    def mult(self, ch: str) -> float:
        return FAKE_CHANNELS[ch][1]

    def cell_of(self, sub):
        return sub[:2]

    def subs_of(self, cell):
        return [self.subs[k] for k in sorted(self.cells[cell].subs)]

    def targets_for(self, cell):
        return [t for t in self.tariff_codes if t != cell[0]]


class FakeModel:
    """Gaussian model on base ratio r0 per arm; r_c = r0 * mult_c; conjugate updates."""

    def __init__(self, dv: FakeDV, priors: dict, default=(0.0, 0.05), noise_sd: float = 0.804):
        self.dv = dv
        self.priors = dict(priors)  # arm -> (mu, sd) at multiplier 1
        self.default = default
        self.noise_sd = noise_sd
        self._obs: list = []
        self.calls = 0

    @property
    def observations(self):
        return list(self._obs)

    def add(self, obs: Observation) -> None:
        self._obs.append(obs)

    def n_obs(self, arm) -> int:
        return sum(1 for o in self._obs if o.arm == arm)

    def _base(self, arm):
        mu, sd = self.priors.get(arm, self.default)
        var = sd * sd
        for o in self._obs:
            if o.arm != arm:
                continue
            m = self.dv.mult(o.channel)
            ov = (self.noise_sd / math.sqrt(o.n) / m) ** 2
            mu, var = self._upd(mu, var, o.y / m, ov)
        return mu, var

    @staticmethod
    def _upd(mu, var, x, ov):
        if var <= 0:
            return mu, 0.0
        k = var / (var + ov)
        return mu + k * (x - mu), var * (1 - k)

    def posterior(self, arm, channel) -> Posterior:
        self.calls += 1
        mu, var = self._base(arm)
        m = self.dv.mult(channel)
        return Posterior(mu * m, math.sqrt(var) * m)

    def posterior_after(self, arm, pilot_channel, y, n, eval_channel) -> Posterior:
        self.calls += 1
        mu, var = self._base(arm)
        mp = self.dv.mult(pilot_channel)
        ov = (self.noise_sd / math.sqrt(n) / mp) ** 2
        mu, var = self._upd(mu, var, y / mp, ov)
        m = self.dv.mult(eval_channel)
        return Posterior(mu * m, math.sqrt(var) * m)


class FakeAllocator:
    """Only shadow_prices is used by the planner."""

    def __init__(self, lm: float = 0.0, lr: float = 0.0, fail: bool = False):
        self.lm, self.lr, self.fail = lm, lr, fail

    def shadow_prices(self, budget, contacts):
        if self.fail:
            raise RuntimeError("boom")
        return self.lm, self.lr


def std_spec(cells=(("tariff_1", "HIGH"), ("tariff_2", "MID")), arpu=(9000.0, 3000.0)):
    """Two cells with 4 subs of sizes 40/80/300/1500."""
    spec = {}
    for ck, mp in zip(cells, arpu):
        spec[ck] = [("LITE", "LOW", 40, mp), ("LITE", "HIGH", 80, mp), ("HEAVY", "LOW", 300, mp),
                    ("HEAVY", "HIGH", 1500, mp)]
    return spec
