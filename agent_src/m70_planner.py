from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip
# PilotPlanner: knowledge-gradient choice of the next pilot (arm x channel x size x pilot sub-cell).
#
# Value of a cell V = sum over its sub-cells of the value of the option the allocator would pick:
#   decision per sub = argmax over admissible options of  net_lcb - lm*cost - lr*n   (or nothing if <= 0),
#   admissible       = net_lcb > 0 and p_pos >= p_min (push: p_min_push),
#   realised value   = net_mean - lm*cost - lr*n of the chosen option (risk-neutral valuation of a risk-averse rule).
# KG(arm, p, n) = E_y[V_after(y)] - V_before, y ~ N(m_p, sd_p^2 + s^2/n) integrated with Gauss-Hermite nodes;
# the posterior after y comes from model.posterior_after (linear in y for the Gaussian model -> 2 calls per channel).
# Total score = KG + immediate - n*cost_p*(1 + lm) - lr*n + confirm bonus, where immediate is the pilot's own
# expected incremental lift over the final plan chosen after y (pilot customers keep max(pilot lift, final lift)),
# integrated over the same Gauss-Hermite nodes, scaled by the expected share of not-yet-piloted customers.
# Eligible targets mirror the allocator: top `top_targets_per_cell` per cell by max_c mean_c / mult_c (re-ranked
# per node for the piloted arm). Pilot unit = a sub-cell (smallest with enough unused contacts).
# Stop: best total <= 0, pilots/reserve/time exhausted. With zero pilots done a pilot is always returned.

import math
import time

import numpy as np

_KG_CONFIRM_FRAC = 0.25  # winner's-curse bonus = frac * min(net_total, sd_total) of the committed paid plan
_KG_CONFIRM_BUDGET_FRAC = 0.10  # paid plan of an arm above this share of the initial budget needs confirmation
_KG_CONFIRM_MIN_PILOTS = 2
_KG_CONFIRM_SD_RATIO = 0.5
_KG_MAX_FAILURES = 2
_KG_EPS = 1e-9


def _kg_norm_ppf(p: float) -> float:
    """Inverse standard normal CDF by bisection on norm_cdf (scalar, called once per planner)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    lo, hi = -40.0, 40.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class _KGCellEval:
    """Current decision in one cell: per-target best option per sub + top-2 targets per sub."""

    targets: list
    t_index: dict
    n: np.ndarray  # (S,) contacts per sub
    sum_p: np.ndarray  # (S,)
    mean_p: np.ndarray  # (S,)
    score: np.ndarray  # (T, S) best admissible score per target (-inf = none)
    value: np.ndarray  # (T, S) mean-valued score of that option
    gross_pc: np.ndarray  # (T, S) posterior-mean gross per contact of that option
    chan: np.ndarray  # (T, S) channel index of that option
    net: np.ndarray  # (T, S)
    net_sd: np.ndarray  # (T, S)
    cost: np.ndarray  # (T, S) money cost of that option
    top1: np.ndarray  # (S,) chosen target index or -1
    top1_score: np.ndarray
    top1_value: np.ndarray
    top2_score: np.ndarray  # best score among targets != top1 (or -inf)
    top2_value: np.ndarray
    v_before: float
    proxy: np.ndarray = None  # (T,) allocator ranking score max_c mean_c / mult_c
    rank: list = None  # target indices by (-proxy, index)
    elig: np.ndarray = None  # (T,) bool: in the allocator's top_targets_per_cell


@dataclass
class _KGCand:
    """One evaluated pilot candidate."""

    arm: tuple
    channel: str
    n: int
    sub: Optional[tuple]
    filters: dict
    kg: float
    immediate: float
    cost_money: float
    penalty: float
    bonus: float
    total: float


class PilotPlanner:
    """Knowledge-gradient pilot planner with reserves, per-arm caps and a winner's-curse confirm bonus."""

    def __init__(self, cfg: Config, dv: Any, model: Any, allocator: Any) -> None:
        self.cfg = cfg
        self.dv = dv
        self.model = model
        self.allocator = allocator
        self._w: dict = {}
        self._pilots_per_arm: dict = {}
        self._used_subs: dict = {}
        self._failures: dict = {}
        self._unavailable: set = set()  # (arm, sub-or-None)
        self._dead_arms: set = set()
        self._n_done = 0
        self._money_spent = 0.0
        self._reach_spent = 0
        self._post_cache: dict = {}
        self.last_debug: dict = {}
        self.history: list = []

        self._channels: list = list(dv.channels)
        self._ch_cost = np.array([float(dv.cost(c)) for c in self._channels], dtype=float)
        mults = []
        for c in self._channels:
            try:
                m = float(dv.mult(c))
            except Exception:  # noqa: BLE001 - duck-typed DataView
                m = 1.0
            mults.append(m if math.isfinite(m) and m > 0 else 1.0)
        self._ch_mult = np.array(mults, dtype=float)
        self._q = np.array(
            [_kg_norm_ppf(cfg.p_min_push if c == "push" else cfg.p_min) for c in self._channels], dtype=float)
        x, w = np.polynomial.hermite.hermgauss(max(1, int(cfg.gh_nodes)))
        self._gh_x = x * math.sqrt(2.0)
        self._gh_w = w / math.sqrt(math.pi)
        sizes = sorted({int(min(max(s, cfg.min_pilot), cfg.max_pilot)) for s in cfg.pilot_sizes})
        self._sizes: list = sizes or [int(cfg.min_pilot)]

        self._cells: list = sorted(dv.cells)
        self._cell_subs: dict = {}
        self._cell_arr: dict = {}
        self._targets: dict = {}
        for ck in self._cells:
            subs = sorted(dv.subs_of(ck), key=lambda s: s.key)
            self._cell_subs[ck] = subs
            self._cell_arr[ck] = (
                np.array([float(s.n) for s in subs], dtype=float),
                np.array([float(s.sum_p) for s in subs], dtype=float),
                np.array([float(s.mean_p) for s in subs], dtype=float),
            )
            self._targets[ck] = list(dv.targets_for(ck))

    # ------------------------------------------------------------------ public API

    def set_weights(self, w: dict) -> None:
        """LLM plausibility weights per arm (default 1.0); used only for candidate ranking."""
        self._w = {}
        for k, v in (w or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv) and fv > 0:
                self._w[tuple(k)] = fv

    @property
    def pilots_per_arm(self) -> dict:
        """Pilots registered per arm (copy)."""
        return dict(self._pilots_per_arm)

    @property
    def used_subs(self) -> dict:
        """Contacts consumed by registered pilots per sub-cell (copy)."""
        return dict(self._used_subs)

    @property
    def unavailable(self) -> set:
        """(arm, sub-or-None) pairs that failed in run_pilot (copy)."""
        return set(self._unavailable)

    def register_result(self, spec: PilotSpec, result: dict) -> None:
        """Bookkeeping after env.run_pilot: usage/counters on success, unavailability on error."""
        arm = tuple(spec.arm)
        if not isinstance(result, dict) or "error" in result:
            self._unavailable.add((arm, spec.sub))
            self._failures[arm] = self._failures.get(arm, 0) + 1
            if spec.sub is None or self._failures[arm] >= _KG_MAX_FAILURES:
                self._dead_arms.add(arm)
            self.history.append({"arm": arm, "channel": spec.channel, "sub": spec.sub, "error": True})
            return
        try:
            n = int(result.get("n_customers", spec.n))
        except (TypeError, ValueError, OverflowError):
            n = int(spec.n)
        n = max(0, n)
        try:
            cost = float(result.get("cost", n * float(self.dv.cost(spec.channel))))
        except (TypeError, ValueError):
            cost = float("nan")
        if not math.isfinite(cost) or cost < 0:
            cost = n * float(self.dv.cost(spec.channel))
        self._pilots_per_arm[arm] = self._pilots_per_arm.get(arm, 0) + 1
        self._n_done += 1
        self._money_spent += cost
        self._reach_spent += n
        if spec.sub is not None:
            self._used_subs[spec.sub] = self._used_subs.get(spec.sub, 0) + n
        self.history.append({"arm": arm, "channel": spec.channel, "sub": spec.sub, "n": n, "cost": cost})

    def next_pilot(self, state: ExploreState) -> Optional[PilotSpec]:
        """Best pilot by KG total score, or None to stop exploring."""
        t_start = time.monotonic()
        # Calibration updates every arm, including arms without a new observation.
        # Keep memoization within a decision only so all candidates use fresh evidence.
        self._post_cache.clear()
        zero_done = self._zero_done(state)
        n_done = max(self._n_done, sum(int(v) for v in (state.pilots_per_arm or {}).values()))
        if state.pilots_left <= 0 or n_done >= int(self.cfg.max_pilots):
            return self._stop("no_pilots_left")
        time_left = state.time_left()
        if time_left <= 0 and not zero_done:
            return self._stop("deadline")
        lm, lr = self._shadow(state)
        cands = self._evaluate(state, lm, lr, relax=False, t_start=t_start)
        best = self._pick(cands)
        if best is not None and best.total > 0:
            return self._spec(best, "kg")
        if not zero_done:
            return self._stop("best_total_nonpositive", best)
        if not cands:
            cands = self._evaluate(state, lm, lr, relax=True, t_start=time.monotonic())
        forced = self._pick_cheapest(cands)
        if forced is None:
            forced = self._fallback_candidate(state)
        if forced is None:
            return self._stop("no_feasible_pilot")
        return self._spec(forced, "forced_first")

    # ------------------------------------------------------------------ bookkeeping helpers

    def _zero_done(self, state: ExploreState) -> bool:
        """True when no pilot has been run yet (by any account)."""
        if self._n_done > 0:
            return False
        if sum(int(v) for v in (state.pilots_per_arm or {}).values()) > 0:
            return False
        if state.explore_reach_spent > 0:
            return False
        try:
            return len(self.model.observations) == 0
        except Exception:  # noqa: BLE001 - duck-typed model
            return True

    def _arm_pilots(self, state: ExploreState, arm: tuple) -> int:
        return max(int((state.pilots_per_arm or {}).get(arm, 0)), int(self._pilots_per_arm.get(arm, 0)))

    def _sub_used(self, state: ExploreState, sub: tuple) -> int:
        return max(int((state.used_subs or {}).get(sub, 0)), int(self._used_subs.get(sub, 0)))

    def _stop(self, why: str, best: Optional[_KGCand] = None) -> None:
        self.last_debug = {"decision": "stop", "reason": why,
                           "best_total": None if best is None else round(best.total, 2)}
        return None

    def _shadow(self, state: ExploreState) -> tuple:
        try:
            lm, lr = self.allocator.shadow_prices(state.remaining_budget, state.remaining_contacts)
            lm, lr = float(lm), float(lr)
            if not (math.isfinite(lm) and math.isfinite(lr)):
                return 0.0, 0.0
            return max(0.0, lm), max(0.0, lr)
        except Exception:  # noqa: BLE001 - allocator failure must not stop planning
            return 0.0, 0.0

    # ------------------------------------------------------------------ posteriors

    def _post(self, arm: tuple, ch: str) -> tuple:
        """(mean, sd) cached within one decision and invalidated on new arm observations."""
        try:
            nobs = int(self.model.n_obs(arm))
        except Exception:  # noqa: BLE001
            nobs = -1
        key = (arm, ch)
        hit = self._post_cache.get(key)
        if hit is not None and hit[0] == nobs and nobs >= 0:
            return hit[1], hit[2]
        try:
            p = self.model.posterior(arm, ch)
            m, s = float(p.mean), float(p.sd)
        except Exception:  # noqa: BLE001
            m, s = 0.0, 0.25
        if not math.isfinite(m):
            m = 0.0
        if not math.isfinite(s) or s < 0:
            s = 0.25
        self._post_cache[key] = (nobs, m, s)
        return m, s

    def _after_linear(self, arm: tuple, p_ch: str, n: int, y_lo: float, y_hi: float) -> Optional[tuple]:
        """Posterior after y on p_ch for each eval channel as (mean(y_lo), slope, sd) arrays, or None."""
        C = len(self._channels)
        a = np.zeros(C)
        b = np.zeros(C)
        s = np.zeros(C)
        dy = y_hi - y_lo
        for j, c in enumerate(self._channels):
            try:
                lo = self.model.posterior_after(arm, p_ch, y_lo, n, c)
                hi = self.model.posterior_after(arm, p_ch, y_hi, n, c)
                lo_m, hi_m, sd = float(lo.mean), float(hi.mean), float(hi.sd)
            except Exception:  # noqa: BLE001
                return None
            if not (math.isfinite(lo_m) and math.isfinite(hi_m) and math.isfinite(sd)):
                return None
            a[j] = lo_m
            b[j] = (hi_m - lo_m) / dy if dy > 0 else 0.0
            s[j] = max(sd, 0.0)
        return a, b, s

    # ------------------------------------------------------------------ option arithmetic

    def _best_over_channels(self, M: np.ndarray, D: np.ndarray, n: np.ndarray, sp: np.ndarray,
                            lm: float, lr: float) -> tuple:
        """Best admissible channel per (K, S) given posterior means M (K,C) and sds D (K,C).

        Returns score, value, chan, net, net_sd, cost, gross_pc arrays of shape (K, S).
        """
        cost_c = self._ch_cost[None, :, None]
        net = M[:, :, None] * sp[None, None, :] - cost_c * n[None, None, :]
        nsd = np.abs(D[:, :, None] * sp[None, None, :])
        lcb = net - float(self.cfg.z_risk) * nsd
        q = self._q[None, :, None]
        with np.errstate(invalid="ignore"):
            ok_p = np.where(nsd > 0, net >= q * nsd, net > 0)
        adm = (lcb > 0) & ok_p & (n[None, None, :] > 0)
        money = cost_c * n[None, None, :]
        pen = lm * money + lr * n[None, None, :]
        score = np.where(adm, lcb - pen, -np.inf)
        value = net - pen
        ci = np.argmax(score, axis=1)  # (K, S)
        take = ci[:, None, :]
        b_score = np.take_along_axis(score, take, axis=1)[:, 0, :]
        b_value = np.take_along_axis(value, take, axis=1)[:, 0, :]
        b_net = np.take_along_axis(net, take, axis=1)[:, 0, :]
        b_sd = np.take_along_axis(nsd, take, axis=1)[:, 0, :]
        b_cost = np.take_along_axis(np.broadcast_to(money, net.shape), take, axis=1)[:, 0, :]
        mp = np.where(n > 0, sp / np.maximum(n, 1.0), 0.0)
        gross_pc = np.take_along_axis(M, ci, axis=1) * mp[None, :]
        return b_score, b_value, ci, b_net, b_sd, b_cost, gross_pc

    def _cell_eval(self, ck: tuple, lm: float, lr: float) -> _KGCellEval:
        targets = self._targets[ck]
        n, sp, mp = self._cell_arr[ck]
        T, C, S = len(targets), len(self._channels), len(n)
        M = np.zeros((T, C))
        D = np.zeros((T, C))
        for i, tgt in enumerate(targets):
            arm = (ck[0], ck[1], tgt)
            for j, c in enumerate(self._channels):
                M[i, j], D[i, j] = self._post(arm, c)
        proxy = (M / self._ch_mult[None, :]).max(axis=1) if T and C else np.full(T, -np.inf)
        rank = sorted(range(T), key=lambda i: (-float(proxy[i]), i))
        k = max(0, int(self.cfg.top_targets_per_cell))
        elig = np.zeros(T, dtype=bool)
        elig[rank[:k]] = True
        t_index = {t: i for i, t in enumerate(targets)}
        if T == 0 or S == 0 or C == 0:
            z = np.zeros((T, S))
            neg = np.full((T, S), -np.inf)
            zs = np.zeros(S)
            return _KGCellEval(targets, t_index, n, sp, mp, neg, z, np.full((T, S), np.nan),
                               np.zeros((T, S), dtype=int), z, z, z, np.full(S, -1), np.full(S, -np.inf), zs,
                               np.full(S, -np.inf), zs, 0.0, proxy, rank, elig)
        score, value, chan, net, nsd, cost, gross_pc = self._best_over_channels(M, D, n, sp, lm, lr)
        score_e = np.where(elig[:, None], score, -np.inf)  # only allocator-eligible targets compete
        order = np.argsort(-score_e, axis=0, kind="stable")  # ties -> lower target index
        cols = np.arange(S)
        i1 = order[0]
        s1 = score_e[i1, cols]
        v1 = value[i1, cols]
        if T > 1:
            i2 = order[1]
            s2 = score_e[i2, cols]
            v2 = value[i2, cols]
        else:
            s2 = np.full(S, -np.inf)
            v2 = np.zeros(S)
        chosen = s1 > 0
        top1 = np.where(chosen, i1, -1)
        v_before = float(np.sum(np.where(chosen, v1, 0.0)))
        return _KGCellEval(targets, t_index, n, sp, mp, score, value, gross_pc,
                           chan, net, nsd, cost, top1, s1, v1, s2, v2, v_before, proxy, rank, elig)

    @staticmethod
    def _competitor(ev: _KGCellEval, idx: list) -> tuple:
        """Best admissible option per sub among targets `idx`: (score, value, gross_pc) arrays of shape (S,)."""
        S = len(ev.n)
        if not idx or S == 0:
            return np.full(S, -np.inf), np.zeros(S), np.full(S, np.nan)
        cols = np.arange(S)
        sub_sc = ev.score[idx]
        j = np.argmax(sub_sc, axis=0)  # ties -> earlier in idx (allocator rank order)
        s = sub_sc[j, cols]
        ok = np.isfinite(s) & (s > 0)
        rows = np.asarray(idx)[j]
        return (np.where(ok, s, -np.inf), np.where(ok, ev.value[rows, cols], 0.0),
                np.where(ok, ev.gross_pc[rows, cols], np.nan))

    # ------------------------------------------------------------------ candidates

    def _rank_arms(self, state: ExploreState) -> list:
        """Top cfg.top_arms arms by sum_p_cell * max_c(mean + sd) * w, excluding capped/failed arms."""
        scored = []
        cap = int(self.cfg.max_pilots_per_arm)
        for ck in self._cells:
            cell = self.dv.cells[ck]
            sp = float(cell.sum_p)
            for tgt in self._targets[ck]:
                arm = (ck[0], ck[1], tgt)
                if arm in self._dead_arms or self._arm_pilots(state, arm) >= cap:
                    continue
                ucb = max(sum(self._post(arm, c)) for c in self._channels) if self._channels else 0.0
                w = self._w.get(arm, 1.0)
                s = sp * ucb * (w if ucb >= 0 else 1.0 / w)  # a weight > 1 always promotes the arm
                if math.isfinite(s):
                    scored.append((-s, arm))
        scored.sort()
        return [a for _, a in scored[: max(0, int(self.cfg.top_arms))]]

    def _limits(self, state: ExploreState, relax: bool) -> tuple:
        """(money_cap, reach_cap) available for the next pilot."""
        money = max(0.0, float(state.remaining_budget))
        reach = max(0, int(state.remaining_contacts))
        if relax:
            return money, reach
        init_money = money + max(0.0, float(state.explore_money_spent))
        init_reach = reach + max(0, int(state.explore_reach_spent))
        m_res = float(self.cfg.explore_money_frac) * init_money - float(state.explore_money_spent)
        r_res = int(math.floor(float(self.cfg.explore_reach_frac) * init_reach)) - int(state.explore_reach_spent)
        return max(0.0, min(money, m_res)), max(0, min(reach, r_res))

    def _sub_avail(self, state: ExploreState, arm: tuple, ck: tuple) -> list:
        """[(avail, key, sub_index, sub)] of sub-cells still usable for this arm."""
        out = []
        for si, sc in enumerate(self._cell_subs[ck]):
            if (arm, sc.key) in self._unavailable:
                continue
            out.append((int(sc.n) - self._sub_used(state, sc.key), sc.key, si, sc))
        return out

    def _pilot_sub(self, state: ExploreState, arm: tuple, ck: tuple, n: int,
                   avail: Optional[list] = None) -> Optional[tuple]:
        """(sub_key, filters, sub_index) of the smallest sub-cell with >= n unused contacts, or None."""
        best = None
        for av, key, si, sc in (avail if avail is not None else self._sub_avail(state, arm, ck)):
            if av >= n and (best is None or (av, key) < best[0]):
                best = ((av, key), sc, si)
        if best is None:
            return None
        sc = best[1]
        filt = {"filter_current_tariff": sc.key[0], "filter_arpu_segment": sc.key[1],
                "filter_data_segment": sc.key[2], "filter_call_segment": sc.key[3]}
        return sc.key, filt, best[2]

    def _sizes_for(self, afford: int) -> list:
        sizes = [s for s in self._sizes if s <= afford]
        if afford >= int(self.cfg.min_pilot) and afford < self._sizes[-1] and afford not in sizes:
            sizes.append(int(afford))
        return sorted(sizes)

    def _evaluate(self, state: ExploreState, lm: float, lr: float, relax: bool, t_start: float) -> list:
        """Evaluate the fixed top_arms shortlist; CPU speed must not change the candidates.

        The orchestrator still enforces the overall run deadline between pilot decisions.
        """
        money_cap, reach_cap = self._limits(state, relax)
        if reach_cap < int(self.cfg.min_pilot):
            return []
        arms = self._rank_arms(state)
        init_budget = max(0.0, float(state.remaining_budget)) + max(0.0, float(state.explore_money_spent))
        k = max(0, int(self.cfg.top_targets_per_cell))
        evals: dict = {}
        cands: list = []
        n_eval = 0
        s2 = float(self.cfg.noise_sd) ** 2
        for ai, arm in enumerate(arms):
            n_eval = ai + 1
            ck = (arm[0], arm[1])
            ev = evals.get(ck)
            if ev is None:
                ev = self._cell_eval(ck, lm, lr)
                evals[ck] = ev
            t = ev.t_index.get(arm[2])
            if t is None or len(ev.n) == 0 or k == 0:
                continue
            avail = self._sub_avail(state, arm, ck)
            max_avail = max((a[0] for a in avail), default=0)
            if max_avail < int(self.cfg.min_pilot):
                continue
            others = [i for i in ev.rank if i != t]
            comp_in = self._competitor(ev, others[: k - 1])
            comp_out = self._competitor(ev, others[:k])
            thr = (float(ev.proxy[others[k - 1]]), others[k - 1]) if len(others) >= k else (-math.inf, math.inf)
            bonus_info = self._confirm_info(state, arm, ev, t, init_budget)
            for pj, p_ch in enumerate(self._channels):
                cost_p = float(self._ch_cost[pj])
                afford = reach_cap if cost_p <= 0 else min(reach_cap, int(money_cap // cost_p))
                afford = min(afford, int(self.cfg.max_pilot), max_avail)
                sizes = self._sizes_for(afford)
                if not sizes:
                    continue
                m_p, sd_p = self._post(arm, p_ch)
                for n in sizes:
                    ps = self._pilot_sub(state, arm, ck, n, avail)
                    if ps is None:
                        continue
                    sub_key, filt, si = ps
                    sub_n = float(ev.n[si])
                    fresh = max(0.0, 1.0 - self._sub_used(state, sub_key) / sub_n) if sub_n > 0 else 0.0
                    res = self._kg(arm, ev, t, pj, p_ch, n, m_p, sd_p, s2, comp_in, comp_out, thr, si, fresh,
                                   lm, lr)
                    if res is None:
                        continue
                    kg, imm = res
                    money = n * cost_p
                    penalty = money * (1.0 + lm) + lr * n
                    bonus = 0.0
                    if bonus_info is not None:
                        amount, dom_ch = bonus_info
                        bonus = amount if p_ch == dom_ch else 0.5 * amount
                    total = kg + imm - penalty + bonus
                    cands.append(_KGCand(arm, p_ch, int(n), sub_key, filt, float(kg), float(imm), float(money),
                                         float(penalty), float(bonus), float(total)))
        self.last_debug = {"n_arms_ranked": len(arms), "n_arms_evaluated": n_eval, "n_candidates": len(cands),
                           "lambda_money": lm, "lambda_reach": lr, "money_cap": money_cap, "reach_cap": reach_cap,
                           "relax": relax, "compute_s": round(time.monotonic() - t_start, 3)}
        return cands

    def _kg(self, arm: tuple, ev: _KGCellEval, t: int, pj: int, p_ch: str, n: int, m_p: float, sd_p: float,
            s2: float, comp_in: tuple, comp_out: tuple, thr: tuple, si: int, fresh: float,
            lm: float, lr: float) -> Optional[tuple]:
        """(KG, immediate) of a pilot of size n on channel p_ch in sub si; KG = E_y[V_after] - V_before.

        comp_in / comp_out: best competitor per sub when the arm is / is not in the allocator's top-k targets;
        thr = (proxy, index) of the k-th other target (the arm is eligible iff it ranks above it).
        """
        pilot_mp = float(ev.mean_p[si])
        pred_sd = math.sqrt(max(sd_p, 0.0) ** 2 + s2 / max(n, 1))
        if not math.isfinite(pred_sd) or pred_sd <= 0:
            return 0.0, n * fresh * m_p * pilot_mp
        lin = self._after_linear(arm, p_ch, n, m_p - pred_sd, m_p + pred_sd)
        if lin is None:
            return None
        a, b, sd_after = lin
        ys = m_p + pred_sd * self._gh_x  # (G,)
        M = a[None, :] + b[None, :] * (ys[:, None] - (m_p - pred_sd))  # (G, C)
        D = np.broadcast_to(sd_after[None, :], M.shape)
        sc, val, _, _, _, _, gpc = self._best_over_channels(M, D, ev.n, ev.sum_p, lm, lr)  # (G, S)
        prox = (M / self._ch_mult[None, :]).max(axis=1)  # (G,)
        el = (prox > thr[0]) | ((prox == thr[0]) & (t < thr[1]))
        el_c = el[:, None]
        o_sc = np.where(el_c, comp_in[0][None, :], comp_out[0][None, :])
        o_val = np.where(el_c, comp_in[1][None, :], comp_out[1][None, :])
        o_gpc = np.where(el_c, comp_in[2][None, :], comp_out[2][None, :])
        wins = el_c & (sc > 0) & (sc > o_sc)
        v_after = np.where(wins, val, o_val).sum(axis=1)  # (G,)
        e_after = float(np.dot(self._gh_w, v_after))
        comp_b = comp_in if bool(ev.elig[t]) else comp_out
        wins_b = bool(ev.elig[t]) & (ev.score[t] > 0) & (ev.score[t] > comp_b[0])
        v_before = float(np.where(wins_b, ev.value[t], comp_b[1]).sum())
        # immediate: pilot customers keep max(pilot lift, final lift of the plan chosen after y)
        final_pc = np.where(wins[:, si], gpc[:, si], o_gpc[:, si])  # (G,) nan = sub not deployed
        pilot_pc = M[:, pj] * pilot_mp
        inc = np.where(np.isfinite(final_pc), np.maximum(0.0, pilot_pc - np.nan_to_num(final_pc)), pilot_pc)
        imm = n * fresh * float(np.dot(self._gh_w, inc))
        return e_after - v_before, imm

    def _confirm_info(self, state: ExploreState, arm: tuple, ev: _KGCellEval, t: int,
                      init_budget: float) -> Optional[tuple]:
        """(bonus_amount, deployment_channel) if the arm's committed paid plan needs confirmation."""
        if len(ev.n) == 0:
            return None
        mine = ev.top1 == t
        if not np.any(mine):
            return None
        cost_total = float(ev.cost[t][mine].sum())
        if cost_total <= 0 or cost_total <= _KG_CONFIRM_BUDGET_FRAC * init_budget:
            return None
        net_total = float(ev.net[t][mine].sum())
        sd_total = float(ev.net_sd[t][mine].sum())
        if net_total <= 0:
            return None
        pilots = self._arm_pilots(state, arm)
        if pilots >= _KG_CONFIRM_MIN_PILOTS and sd_total / net_total < _KG_CONFIRM_SD_RATIO:
            return None
        ch_cost = {}
        for ci, c in zip(ev.chan[t][mine], ev.cost[t][mine]):
            ch_cost[int(ci)] = ch_cost.get(int(ci), 0.0) + float(c)
        dom = max(sorted(ch_cost), key=lambda k: ch_cost[k])
        return _KG_CONFIRM_FRAC * min(net_total, sd_total), self._channels[dom]

    # ------------------------------------------------------------------ selection

    @staticmethod
    def _cand_key(c: _KGCand) -> tuple:
        return (c.arm, c.channel, c.n, c.sub or ("",))

    def _pick(self, cands: list) -> Optional[_KGCand]:
        best = None
        for c in cands:
            if best is None or c.total > best.total + _KG_EPS or (
                    abs(c.total - best.total) <= _KG_EPS and self._cand_key(c) < self._cand_key(best)):
                best = c
        return best

    def _pick_cheapest(self, cands: list) -> Optional[_KGCand]:
        """Informative first (kg > 0), then cheapest money, then best total."""
        if not cands:
            return None
        return min(cands, key=lambda c: (0 if c.kg > _KG_EPS else 1, round(c.cost_money, 6), -c.total,
                                         self._cand_key(c)))

    def _fallback_candidate(self, state: ExploreState) -> Optional[_KGCand]:
        """Minimal feasible pilot (largest cell, first target, cheapest affordable channel)."""
        money, reach = self._limits(state, relax=True)
        n = int(self.cfg.min_pilot)
        if reach < n or not self._channels:
            return None
        order = sorted(range(len(self._channels)), key=lambda j: (self._ch_cost[j], j))
        for ck in sorted(self._cells, key=lambda k: (-int(self.dv.cells[k].n), k)):
            for tgt in self._targets[ck]:
                arm = (ck[0], ck[1], tgt)
                if arm in self._dead_arms:
                    continue
                for j in order:
                    if n * self._ch_cost[j] > money:
                        continue
                    ps = self._pilot_sub(state, arm, ck, n)
                    if ps is None:
                        continue
                    return _KGCand(arm, self._channels[j], n, ps[0], ps[1], 0.0, 0.0, float(n * self._ch_cost[j]),
                                   0.0, 0.0, 0.0)
        return None

    def _spec(self, c: _KGCand, mode: str) -> PilotSpec:
        reason = (f"{mode}: kg={c.kg:.1f} imm={c.immediate:.1f} pen={c.penalty:.1f} bonus={c.bonus:.1f} "
                  f"total={c.total:.1f}")
        self.last_debug = {**self.last_debug, "decision": mode, "arm": list(c.arm), "channel": c.channel, "n": c.n,
                           "sub": None if c.sub is None else list(c.sub), "kg": c.kg, "immediate": c.immediate,
                           "penalty": c.penalty, "bonus": c.bonus, "total": c.total}
        return PilotSpec(arm=tuple(c.arm), channel=c.channel, n=int(c.n), sub=c.sub, filters=dict(c.filters),
                         score=float(c.total), reason=reason)
