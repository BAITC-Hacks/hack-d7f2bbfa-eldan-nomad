#!/usr/bin/env python3
"""Decompose the gap (oracle_net - agent_net) into loss categories.

Method (exact, residual ~ 0 by construction)
--------------------------------------------
Let each targetable sub-cell s = (tariff, arpu, data, call) choose at most one
option o = (target, channel) and let the two shared resources (reach R=15000,
money M=100000) be priced by shadow prices (lam_R, lam_M) that minimise the
Lagrangian dual

    D(lam) = sum_s rho_s(lam)^+ + lam_R*R + lam_M*M,
    rho_s  = max_o [ gross_so - cost_so - lam_M*cost_so - lam_R*n_s ].

D(lam*) is the LP (fluid) upper bound of any plan (no slot limit, no grouping,
fractional sub-cells).  For ANY plan with a contact ledger (pilots + finals,
duplicates included) the identity

    D - net = lam_R*(R - K) + lam_M*(M - C)                 [h unused]
            + sum_uncontacted_i rho_i^+                       [e missed]
            + sum_contacted_i (rho_i^+ - r_eff_i)             [a/b/f/k/c/d/i]
            + sum_noneffective_k (cost_k*(1+lam_M) + lam_R)  [g dedup]

holds exactly (rho_i = rho_s/n_s, r_k = lift_k - cost_k*(1+lam_M) - lam_R).
Every category is computed for the agent AND for the oracle plan with the
TRUE world impact model; the gap per category is agent_regret - oracle_regret,
so the categories sum to oracle_net - agent_net (residual = float noise).

Per contacted customer, the effective (max-lift) contact's slack
rho_i^+ - r_eff is attributed hierarchically:
  a  effective contact is a pilot contact (pilot net cost + opportunity)
  b  lift <= 0 (zero/negative gross: pure reach/money waste)
  f  deployed option has negative true net on its (tariff, arpu) cell
  k  sub-cell not worth contacting at shadow prices (over-deployment)
  c  wrong target: best target for the chosen channel vs chosen target
  d  wrong channel: best (target, channel) of the cell vs best target at chosen channel
  i  coarse grouping / selection: sub-cell optimum vs cell-level optimum
Non-effective (duplicate) contacts -> g (split pilot+final / final+final).
The 10-slot limit (j) shows up as oracle-side slack; it is reported separately
as oracle_noslot - oracle (the value the benchmark itself loses to the cap).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment import make_environment  # noqa: E402
from mock_environment import MAX_TOTAL_CONTACTS, TOTAL_BUDGET, _mock_impact_model  # noqa: E402
from scoring_core import (  # noqa: E402
    CHANNELS,
    MAX_CAMPAIGNS,
    MAX_CUSTOMERS_PER_CAMPAIGN,
    apply_filters,
    sanitize_campaigns,
    score_campaigns,
)
import tools.oracle_eval as oracle_mod  # noqa: E402
from tools.stress_eval import FILTER_COLUMNS, build_worlds, _select_worlds  # noqa: E402

CATS = ["a_pilot", "b_zero_gross", "c_wrong_target", "d_wrong_channel", "e_missed",
        "f_neg_net_cell", "g_dup_pilot_final", "g_dup_final_final", "h_unused",
        "i_grouping", "k_overdeploy"]
CH_LIST = list(CHANNELS)


# ---------------------------------------------------------------- true model
class TrueModel:
    """Per-(cur, arpu, target, channel) true lift ratio, mirroring scoring_core."""

    def __init__(self, impact: pd.DataFrame, fallback, tariffs: pd.DataFrame, profile: pd.DataFrame):
        self.codes = sorted(tariffs["tariff_plan_code"].dropna().astype(str).unique(),
                            key=lambda c: (len(c), c))
        fb_conv = float(impact["conversion_rate"].median())
        arm = {(str(r.tariff_plan_code_from), str(r.arpu_segment), str(r.tariff_plan_code_to)):
               (float(r.arpu_change_pct), float(r.conversion_rate))
               for r in impact.itertuples(index=False)}
        cells = (profile[["current_tariff", "arpu_segment"]].dropna().drop_duplicates()
                 .astype(str).itertuples(index=False, name=None))
        self.ratio: dict[tuple[str, str, str, str], float] = {}
        for cur, seg in cells:
            for t in self.codes:
                known = arm.get((cur, seg, t))
                if known is None or not np.isfinite(known[0]):
                    ch_, cv = fallback(cur, t, seg, tariffs, fb_conv)
                else:
                    ch_, cv = known
                for ch in CH_LIST:
                    eff = min(float(cv) * CHANNELS[ch]["conversion_multiplier"], 1.0) \
                        if np.isfinite(cv) else np.nan
                    val = float(ch_) * eff
                    self.ratio[(cur, seg, t, ch)] = val if np.isfinite(val) else 0.0

    def r(self, cur, seg, t, ch) -> float:
        return self.ratio.get((str(cur), str(seg), str(t), str(ch)), 0.0)


# ---------------------------------------------------------------- ledger
def contact_ledger(campaigns: pd.DataFrame, profile: pd.DataFrame, tm: TrueModel) -> pd.DataFrame:
    """Replicate score_campaigns' caps loop; one row per (campaign, customer) contact."""
    rows = []
    rem_reach, rem_money = MAX_TOTAL_CONTACTS, TOTAL_BUDGET
    for idx, camp in campaigns.reset_index(drop=True).iterrows():
        ch = camp["channel"]
        cpc = CHANNELS[ch]["cost_per_contact"]
        seg = apply_filters(profile, camp).sort_values("ID_NUMBER")
        seg = seg.iloc[:MAX_CUSTOMERS_PER_CAMPAIGN]
        seg = seg.iloc[:max(rem_reach, 0)]
        if cpc > 0:
            seg = seg.iloc[:max(int(rem_money // cpc), 0)]
        rem_reach -= len(seg)
        rem_money -= len(seg) * cpc
        if not len(seg):
            continue
        ratio = np.array([tm.r(c, s, camp["target_tariff"], ch)
                          for c, s in zip(seg["current_tariff"], seg["arpu_segment"])])
        lift = np.nan_to_num(ratio * seg["predicted_arpu"].to_numpy(dtype=float), nan=0.0)
        name = camp.get("campaign_name") or f"campaign_{idx}"
        is_pilot = isinstance(camp.get("explicit_ids"), (list, tuple, np.ndarray)) and \
            len(camp.get("explicit_ids")) > 0
        rows.append(pd.DataFrame({
            "ID_NUMBER": seg["ID_NUMBER"].to_numpy(), "camp": name, "order": idx,
            "pilot": is_pilot, "target": camp["target_tariff"], "channel": ch,
            "cost": float(cpc), "lift": lift,
        }))
    if not rows:
        return pd.DataFrame(columns=["ID_NUMBER", "camp", "order", "pilot", "target",
                                     "channel", "cost", "lift"])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- LP dual
class Bound:
    """Sub-cell option tables + dual minimisation for the LP upper bound."""

    def __init__(self, profile: pd.DataFrame, tm: TrueModel):
        keys = ["current_tariff", "arpu_segment", "data_segment", "call_segment"]
        clean = profile.dropna(subset=keys)
        g = clean.groupby(keys, observed=True, sort=True)
        agg = g.agg(n=("ID_NUMBER", "size"), s=("predicted_arpu", lambda x: float(x.sum(skipna=True))))
        self.sub_keys = [tuple(map(str, k)) for k in agg.index]
        self.n = agg["n"].to_numpy(dtype=float)
        self.s = agg["s"].to_numpy(dtype=float)
        self.sub_index = {k: i for i, k in enumerate(self.sub_keys)}
        # option matrix per sub: gross[s, o], cost[s, o] ; o over (target, channel), target != cur
        self.opts = [(t, ch) for t in tm.codes for ch in CH_LIST]
        S, O = len(self.sub_keys), len(self.opts)
        self.gross = np.full((S, O), -np.inf)
        self.cost = np.zeros((S, O))
        for i, (cur, seg, _d, _c) in enumerate(self.sub_keys):
            for j, (t, ch) in enumerate(self.opts):
                if t == cur:
                    continue
                self.gross[i, j] = tm.r(cur, seg, t, ch) * self.s[i]
                self.cost[i, j] = CHANNELS[ch]["cost_per_contact"] * self.n[i]
        self.net = self.gross - self.cost

    def rho(self, lr: float, lm: float) -> np.ndarray:
        red = self.net - lm * self.cost - lr * self.n[:, None]
        return np.maximum(red.max(axis=1), 0.0)

    def dual(self, lr: float, lm: float) -> float:
        return float(self.rho(lr, lm).sum() + lr * MAX_TOTAL_CONTACTS + lm * TOTAL_BUDGET)

    def minimise(self) -> tuple[float, float, float]:
        best = (np.inf, 0.0, 0.0)
        lrs = np.concatenate([[0.0], np.geomspace(1, 5000, 60)])
        lms = np.concatenate([[0.0], np.geomspace(0.01, 200, 60)])
        for lr in lrs:
            for lm in lms:
                v = self.dual(lr, lm)
                if v < best[0]:
                    best = (v, lr, lm)
        v, lr, lm = best
        step_r, step_m = max(lr, 1.0) * 0.5, max(lm, 0.01) * 0.5
        for _ in range(60):  # coordinate pattern search refinement
            improved = False
            for dr, dm in ((step_r, 0), (-step_r, 0), (0, step_m), (0, -step_m)):
                nr, nm = max(lr + dr, 0.0), max(lm + dm, 0.0)
                nv = self.dual(nr, nm)
                if nv < v - 1e-9:
                    v, lr, lm, improved = nv, nr, nm, True
            if not improved:
                step_r *= 0.5
                step_m *= 0.5
        return v, lr, lm


# ---------------------------------------------------------------- decompose
def decompose(ledger: pd.DataFrame, profile: pd.DataFrame, tm: TrueModel, bd: Bound,
              lr: float, lm: float) -> dict:
    rho_sub = bd.rho(lr, lm)
    keys = ["current_tariff", "arpu_segment", "data_segment", "call_segment"]
    prof = profile[["ID_NUMBER", *keys, "predicted_arpu"]].copy()
    sub_i = np.array([bd.sub_index.get(tuple(map(str, k)), -1) if not any(pd.isna(x) for x in k) else -1
                      for k in prof[keys].itertuples(index=False, name=None)])
    rho_i = np.where(sub_i >= 0, rho_sub[np.maximum(sub_i, 0)] / bd.n[np.maximum(sub_i, 0)], 0.0)
    prof["rho"] = rho_i
    prof["sub"] = sub_i
    out = {c: 0.0 for c in CATS}

    # cell-level option tables (for f / c / d)
    cell_n = prof.dropna(subset=keys[:2]).groupby(keys[:2])["ID_NUMBER"].size()
    cell_s = prof.dropna(subset=keys[:2]).groupby(keys[:2])["predicted_arpu"].sum()

    def cell_red(cur, seg, t, ch):
        n = float(cell_n.get((cur, seg), 0)); s = float(cell_s.get((cur, seg), 0.0))
        c = CHANNELS[ch]["cost_per_contact"]
        return tm.r(cur, seg, t, ch) * s - n * (c * (1 + lm) + lr), tm.r(cur, seg, t, ch) * s - n * c

    cache: dict = {}

    def cell_best(cur, seg, ch):
        key = (cur, seg, ch)
        if key not in cache:
            cands = [t for t in tm.codes if t != cur]
            t_ch = max(cands, key=lambda t: (cell_red(cur, seg, t, ch)[0], t))
            o_all = max(((t, c) for t in cands for c in CH_LIST),
                        key=lambda o: (cell_red(cur, seg, o[0], o[1])[0], o))
            cache[key] = (t_ch, o_all)
        return cache[key]

    K = len(ledger)
    C = float(ledger["cost"].sum()) if K else 0.0
    out["h_unused"] = lr * (MAX_TOTAL_CONTACTS - K) + lm * (TOTAL_BUDGET - C)
    contacted = set(ledger["ID_NUMBER"]) if K else set()
    out["e_missed"] = float(prof.loc[~prof["ID_NUMBER"].isin(contacted), "rho"].clip(lower=0).sum())

    by_camp: dict = {}

    def add(camp, cat, v):
        d = by_camp.setdefault(camp, {})
        d[cat] = d.get(cat, 0.0) + float(v)
        if cat in out:
            out[cat] += v

    info = {"zero_gross_final_contacts": 0, "dup_contacts": 0, "pilot_contacts": 0,
            "pilot_cost": 0.0, "pilot_gross_eff": 0.0}
    if K:
        led = ledger.merge(prof, on="ID_NUMBER", how="left")
        led["r"] = led["lift"] - led["cost"] * (1 + lm) - lr
        # effective contact: max lift, ties -> final over pilot, then earliest
        led = led.sort_values(["ID_NUMBER", "lift", "pilot", "order"],
                              ascending=[True, False, True, True], kind="mergesort")
        led["eff"] = ~led.duplicated("ID_NUMBER", keep="first")
        has_pilot = led.groupby("ID_NUMBER")["pilot"].transform("any")
        has_final = led.groupby("ID_NUMBER")["pilot"].transform(lambda x: (~x).any())
        info["pilot_contacts"] = int(led["pilot"].sum())
        info["pilot_cost"] = float(led.loc[led["pilot"], "cost"].sum())
        info["pilot_gross_eff"] = float(led.loc[led["pilot"] & led["eff"], "lift"].sum())
        info["dup_contacts"] = int((~led["eff"]).sum())
        dup = led[~led["eff"]]
        w = dup["cost"] * (1 + lm) + lr
        pf = (has_pilot & has_final)[~led["eff"]]
        out["g_dup_pilot_final"] = float(w[pf].sum())
        out["g_dup_final_final"] = float(w[~pf].sum())
        eff = led[led["eff"]]
        slack = eff["rho"].clip(lower=0) - eff["r"]
        piloted = eff["pilot"].to_numpy()
        out["a_pilot"] = float(slack[piloted].sum())
        fin = eff[~piloted]
        for row, sl in zip(fin.itertuples(index=False), slack[~piloted]):
            cur, seg, t, ch = row.current_tariff, row.arpu_segment, row.target, row.channel
            add(row.camp, "_n", 1)
            if row.lift <= 0:
                add(row.camp, "b_zero_gross", sl)
                info["zero_gross_final_contacts"] += 1
                continue
            if pd.isna(cur) or pd.isna(seg) or cell_red(cur, seg, t, ch)[1] < 0:
                add(row.camp, "f_neg_net_cell", sl)
                continue
            if row.sub < 0 or row.rho <= 0:
                add(row.camp, "k_overdeploy", sl)
                continue
            t_ch, (t_all, c_all) = cell_best(cur, seg, ch)
            arpu = 0.0 if pd.isna(row.predicted_arpu) else float(row.predicted_arpu)

            def rk(tt, cc):
                return tm.r(cur, seg, tt, cc) * arpu - CHANNELS[cc]["cost_per_contact"] * (1 + lm) - lr
            r0, r1, r2 = row.r, rk(t_ch, ch), rk(t_all, c_all)
            add(row.camp, "c_wrong_target", r1 - r0)
            add(row.camp, "d_wrong_channel", r2 - r1)
            add(row.camp, "i_grouping", row.rho - r2)
            add(row.camp, "best:" + t_all + "/" + c_all, 1)
    info["by_campaign"] = by_camp
    out["_info"] = info
    return out


# ---------------------------------------------------------------- runners
def score(campaigns: pd.DataFrame, profile, world, tariffs) -> dict:
    for col in ["target_tariff", "channel", *FILTER_COLUMNS]:
        if col not in campaigns.columns:
            campaigns[col] = None
    with contextlib.redirect_stdout(io.StringIO()):
        return score_campaigns(campaigns, profile, world.impact, tariffs,
                               float(profile["predicted_arpu"].sum()), world.fallback)


def run_agent(agent_class, world, seed, profile, tariffs) -> tuple[pd.DataFrame, int]:
    env, internals = make_environment(
        customer_profile=profile, impact_model=world.impact, dict_tariff=tariffs,
        channels=oracle_mod.CHANNELS, total_budget=TOTAL_BUDGET,
        max_total_contacts=MAX_TOTAL_CONTACTS, fallback_predict=world.fallback, seed=seed)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            finals = agent_class().act(env)
    except Exception as exc:  # pragma: no cover
        print(f"[!] agent failed: {exc}", file=sys.stderr)
        finals = []
    with contextlib.redirect_stdout(io.StringIO()):
        finals = sanitize_campaigns(finals, tariffs)[:MAX_CAMPAIGNS]
    pilots = internals.executed_pilot_campaigns()
    return pd.DataFrame(pilots + finals), len(finals)


def fmt(v: float) -> str:
    return f"{v:>12,.0f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worlds", default="mock,flip_sign_30,permute_targets,prior_misspecified")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--agent-module", default="agent")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--detail", action="store_true", help="per-final-campaign attribution")
    args = ap.parse_args(argv)
    os.environ["AGENT_LLM_MODE"] = "off"
    import importlib
    agent_class = importlib.import_module(args.agent_module).Agent
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    profile = pd.read_csv(ROOT / "customer_profile.csv")
    tariffs = pd.read_csv(ROOT / "data" / "dict_tariff.csv")
    base = _mock_impact_model(pd.read_csv(ROOT / "data" / "change_tariff.csv"))
    worlds = _select_worlds(args.worlds, build_worlds(base, tariffs))
    payload = {}
    for world in worlds:
        t0 = time.time()
        tm = TrueModel(world.impact, world.fallback, tariffs, profile)
        bd = Bound(profile, tm)
        D, lr, lm = bd.minimise()
        with contextlib.redirect_stdout(io.StringIO()):
            o_camps, o_res = oracle_mod.oracle_plan(world.impact, world.fallback, profile, tariffs)
            saved = oracle_mod.MAX_CAMPAIGNS
            oracle_mod.MAX_CAMPAIGNS = 1000
            try:
                _, o_res_ns = oracle_mod.oracle_plan(world.impact, world.fallback, profile, tariffs)
            finally:
                oracle_mod.MAX_CAMPAIGNS = saved
        o_net = float(o_res["net_arpu_gain"])
        o_ns = float(o_res_ns["net_arpu_gain"])
        o_led = contact_ledger(pd.DataFrame(o_camps), profile, tm)
        o_dec = decompose(o_led, profile, tm, bd, lr, lm)
        o_sum = sum(o_dec[c] for c in CATS)
        print(f"\n=== world={world.name}  LP bound D={D:,.0f} (lam_R={lr:.2f}/contact, lam_M={lm:.3f}/unit)"
              f"  oracle={o_net:,.0f}  oracle_noslot={o_ns:,.0f}  (j) oracle slot loss={o_ns - o_net:,.0f}"
              f"  oracle check D-net-sum={D - o_net - o_sum:,.1f}")
        per_seed = []
        for seed in seeds:
            camps, n_final = run_agent(agent_class, world, seed, profile, tariffs)
            res = score(camps.copy(), profile, world, tariffs)
            a_net = float(res["net_arpu_gain"])
            led = contact_ledger(camps, profile, tm)
            ledger_net = float(led.groupby("ID_NUMBER")["lift"].max().sum() - led["cost"].sum()) if len(led) else 0.0
            dec = decompose(led, profile, tm, bd, lr, lm)
            delta = {c: dec[c] - o_dec[c] for c in CATS}
            gap = o_net - a_net
            resid = gap - sum(delta.values())
            per_seed.append({"seed": seed, "agent_net": a_net, "ledger_check": ledger_net - a_net,
                             "gap": gap, "residual": resid, "n_final": n_final,
                             "agent": {c: dec[c] for c in CATS}, "delta": delta, "info": dec["_info"]})
            info = {k: v for k, v in dec["_info"].items() if k != "by_campaign"}
            print(f"  seed {seed}: agent={a_net:,.0f} gap={gap:,.0f} residual={resid:,.2f} "
                  f"ledger-scorer diff={ledger_net - a_net:,.2f} finals={n_final} info={info}")
            if args.detail:
                for name, d in dec["_info"]["by_campaign"].items():
                    print(f"      {name:<40} " + " ".join(
                        f"{k}={v:,.0f}" for k, v in sorted(d.items()) if abs(v) >= 1))
        # table
        print(f"  {'category':<20}{'agent_regret':>14}{'oracle_regret':>14}{'gap_share(mean)':>16}"
              + "".join(f"{'s' + str(s):>12}" for s in seeds))
        rows = []
        for c in CATS:
            ag = float(np.mean([p["agent"][c] for p in per_seed]))
            dl = [p["delta"][c] for p in per_seed]
            rows.append((c, ag, o_dec[c], float(np.mean(dl)), dl))
        for c, ag, orr, dm, dl in sorted(rows, key=lambda r: -r[3]):
            print(f"  {c:<20}{fmt(ag):>14}{fmt(orr):>14}{fmt(dm):>16}" + "".join(fmt(x) for x in dl))
        gaps = [p["gap"] for p in per_seed]
        res_ = [p["residual"] for p in per_seed]
        print(f"  {'TOTAL gap':<20}{'':>14}{'':>14}{fmt(np.mean(gaps)):>16}" + "".join(fmt(x) for x in gaps))
        print(f"  {'residual':<20}{'':>14}{'':>14}{np.mean(res_):>16,.2f}" + "".join(f"{x:>12,.2f}" for x in res_))
        print(f"  ({time.time() - t0:.0f}s)")
        payload[world.name] = {"D": D, "lam_R": lr, "lam_M": lm, "oracle": o_net, "oracle_noslot": o_ns,
                               "oracle_regret": {c: o_dec[c] for c in CATS}, "runs": per_seed}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
