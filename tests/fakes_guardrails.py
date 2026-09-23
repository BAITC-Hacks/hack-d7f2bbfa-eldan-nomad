"""Fakes for guardrail tests: a minimal DataView and a cost/contacts-only ScoreSimulator replica."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent_src.contract import CHANNELS_ORDER, SimResult

SEG_COLS = ("current_tariff", "arpu_segment", "data_segment", "call_segment")


@dataclass
class FakeSub:
    key: tuple
    n: int


class FakeDV:
    """Only what Guardrails needs: tariff_codes, channels, subs, cost()."""

    def __init__(self, profile: pd.DataFrame, tariffs: pd.DataFrame, channels: dict):
        self.tariff_codes = sorted(tariffs["tariff_plan_code"].astype(str), key=lambda s: (len(s), s))
        self.channels = [c for c in CHANNELS_ORDER if c in channels] + sorted(set(channels) - set(CHANNELS_ORDER))
        self._channels = channels
        ok = profile.dropna(subset=list(SEG_COLS))
        counts = ok.groupby(list(SEG_COLS)).size()
        self.subs = {tuple(k): FakeSub(tuple(k), int(v)) for k, v in counts.items()}

    def cost(self, ch: str) -> float:
        return float(self._channels[ch]["cost_per_contact"])


class FakeSim:
    """Replicates scoring_core caps (5000 / reach / money, push not money-capped); lift from ratio_fn."""

    def __init__(self, dv: FakeDV, profile: pd.DataFrame, channels: dict, raise_error: bool = False):
        self.dv, self.profile, self.channels, self.raise_error = dv, profile, channels, raise_error
        self.calls = 0

    def segment(self, c: dict) -> pd.DataFrame:
        r = self.profile
        for col, pcol in (("filter_arpu_segment", "arpu_segment"), ("filter_data_segment", "data_segment"),
                          ("filter_call_segment", "call_segment")):
            if pd.notna(c.get(col)):
                r = r[r[pcol] == c[col]]
        if pd.notna(c.get("filter_current_tariff")):
            ts = [t.strip() for t in str(c["filter_current_tariff"]).split(";") if t.strip()]
            r = r[r["current_tariff"].isin(ts)]
        return r.sort_values("ID_NUMBER")

    def simulate(self, campaigns, ratio_fn, budget, contacts) -> SimResult:
        self.calls += 1
        if self.raise_error:
            raise RuntimeError("boom")
        reach, money = int(contacts), float(budget)
        tot_cost, tot_n, gross, per, ok = 0.0, 0, 0.0, [], len(campaigns) <= 10
        for c in campaigns:
            seg = self.segment(c)
            cpc = float(self.channels[c["channel"]]["cost_per_contact"])
            capped_c = len(seg) > 5000
            seg = seg.iloc[:5000]
            capped_r = len(seg) > reach
            seg = seg.iloc[:max(reach, 0)]
            capped_m = False
            if cpc > 0:
                aff = int(money // cpc)
                if len(seg) > aff:
                    capped_m, seg = True, seg.iloc[:max(aff, 0)]
            n = len(seg)
            cost = n * cpc
            reach -= n
            money -= cost
            tot_cost += cost
            tot_n += n
            ok = ok and not capped_r and not capped_m
            per.append({"name": c["campaign_name"], "n_segment": n, "n_contacted": n, "cost": cost, "gross": 0.0,
                        "capped_campaign": capped_c, "capped_reach": capped_r, "capped_money": capped_m})
        return SimResult(gross=gross, cost=tot_cost, contacts=tot_n, net=gross - tot_cost, per_campaign=per,
                         within_limits=ok)


def zero_sub_count(dv: FakeDV) -> int:
    return int(np.sum([s.n for s in dv.subs.values()]))
