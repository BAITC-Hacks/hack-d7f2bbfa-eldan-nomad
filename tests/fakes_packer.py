"""Fakes for CampaignPacker tests: DataView, ScoreSimulator and ArmModel stand-ins."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from agent_src.contract import (
    CHANNELS_ORDER,
    Option,
    Posterior,
    SimResult,
    SubCell,
)

SEG_COLS = ["current_tariff", "arpu_segment", "data_segment", "call_segment"]


class FakeDataView:
    """Minimal DataView: subs from a profile, natural tariff codes, channel table."""

    def __init__(self, profile: pd.DataFrame, tariff_codes: list[str], channels: dict):
        self.profile = profile
        self.tariff_codes = list(tariff_codes)
        self._ch = channels
        self.channels = [c for c in CHANNELS_ORDER if c in channels]
        ok = profile.dropna(subset=SEG_COLS)
        self.subs: dict = {}
        for key, g in ok.groupby(SEG_COLS, sort=True):
            g = g.sort_values("ID_NUMBER")
            p = g["predicted_arpu"].to_numpy(float)
            self.subs[tuple(key)] = SubCell(tuple(key), len(g), float(p.sum()), float(p.mean()),
                                            g["ID_NUMBER"].to_numpy(np.int64), p)

    def cost(self, ch: str) -> float:
        return float(self._ch[ch]["cost_per_contact"])

    def mult(self, ch: str) -> float:
        return float(self._ch[ch]["conversion_multiplier"])


class FakeSim:
    """Simple replica of score_campaigns mechanics (filters, caps, dedup by max)."""

    def __init__(self, profile: pd.DataFrame, channels: dict):
        self.profile = profile
        self.channels = channels
        self.n_simulate = 0

    def segment(self, c: dict) -> pd.DataFrame:
        r = self.profile
        for col, key in (("arpu_segment", "filter_arpu_segment"), ("data_segment", "filter_data_segment"),
                         ("call_segment", "filter_call_segment")):
            if pd.notna(c.get(key)):
                r = r[r[col] == c[key]]
        if pd.notna(c.get("filter_current_tariff")):
            ts = [t.strip() for t in str(c["filter_current_tariff"]).split(";") if t.strip()]
            r = r[r["current_tariff"].isin(ts)]
        return r.sort_values("ID_NUMBER")

    def simulate(self, campaigns: list[dict], ratio_fn: Callable, budget: float, contacts: int) -> SimResult:
        self.n_simulate += 1
        reach, money = int(contacts), float(budget)
        best: dict = {}
        cost_total, n_total, per = 0.0, 0, []
        for c in campaigns:
            cpc = float(self.channels[c["channel"]]["cost_per_contact"])
            seg = self.segment(c).iloc[:5000]
            seg = seg.iloc[:max(reach, 0)]
            if cpc > 0:
                seg = seg.iloc[:max(int(money // cpc), 0)]
            reach -= len(seg)
            money -= len(seg) * cpc
            cost_total += len(seg) * cpc
            n_total += len(seg)
            combos = {k: ratio_fn(k[0], k[1], c["target_tariff"], c["channel"])
                      for k in set(zip(seg["current_tariff"], seg["arpu_segment"]))}
            ratio = np.array([combos[k] for k in zip(seg["current_tariff"], seg["arpu_segment"])], float)
            lifts = ratio * seg["predicted_arpu"].to_numpy(float)
            gross = float(lifts.sum())
            for i, lift in zip(seg["ID_NUMBER"].tolist(), lifts.tolist()):
                best[i] = max(best.get(i, -np.inf), lift)
            per.append({"name": c["campaign_name"], "n_contacted": len(seg), "gross": gross,
                        "cost": len(seg) * cpc})
        g = float(sum(best.values()))
        return SimResult(g, cost_total, n_total, g - cost_total, per, True)


class FakeModel:
    """ArmModel stand-in: posterior mean = ratio[(from, arpu, target)] * channel mult."""

    def __init__(self, ratios: dict, channels: dict, default: float = 0.0):
        self.ratios = ratios
        self.channels = channels
        self.default = default

    def posterior(self, arm: tuple, channel: str) -> Posterior:
        base = self.ratios.get(tuple(arm), self.default)
        return Posterior(base * float(self.channels[channel]["conversion_multiplier"]), 0.01)


def make_option(dv: FakeDataView, sub: tuple, target: str, channel: str, net: float | None = None) -> Option:
    """Option covering the whole sub; net defaults to a positive value."""
    s = dv.subs[sub]
    cost = dv.cost(channel) * s.n
    nm = float(net) if net is not None else 100.0 + s.sum_p * 0.01
    return Option(sub, target, channel, s.n, cost, nm, 1.0, nm - 1.0, 0.99)
