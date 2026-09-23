from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip
# Allocator: choose at most one (target, channel) option per sub-cell under money and reach limits.
# Method: per-sub admissible options -> Lagrangian relaxation over (money, reach) with nested bisection
# (outer lambda_reach, inner lambda_money) -> feasibility repair -> greedy fill of leftover resources
# -> drop-and-refill local search. Deterministic (ties broken by sub key order, then option order).

import math

import numpy as np

_ALLOC_BISECT_ITERS = 48
_ALLOC_EPS = 1e-9


class Allocator:
    """Budget/contacts-constrained assignment of options to sub-cells."""

    def __init__(self, cfg: Config, dv: Any, model: Any, log: Optional[RunLog] = None) -> None:
        self.cfg = cfg
        self.dv = dv
        self.model = model
        self.log = log

    # ------------------------------------------------------------------ options

    def _alloc_rank_targets(self, cell: CellKey) -> list[str]:
        """Top `top_targets_per_cell` targets of a cell by posterior mean normalised to multiplier 1.

        Score of a target = max over channels of posterior(arm, ch).mean / k(arm, ch), where k is the
        model's channel scale (`model.k_channel`, saturation-aware) or, if absent, the raw multiplier.
        Ties -> target code order.
        """
        k_fn = getattr(self.model, "k_channel", None)
        scored: list[tuple[float, int, str]] = []
        for i, tgt in enumerate(self.dv.targets_for(cell)):
            arm = (cell[0], cell[1], tgt)
            best = -math.inf
            for ch in self.dv.channels:
                m = float(k_fn(arm, ch)) if callable(k_fn) else float(self.dv.mult(ch))
                if not (math.isfinite(m) and m > 0.0):
                    m = 1.0
                mean = float(self.model.posterior(arm, ch).mean)
                if math.isfinite(mean):
                    best = max(best, mean / m)
            scored.append((best, i, tgt))
        scored.sort(key=lambda t: (-t[0], t[1]))
        k = max(0, int(self.cfg.top_targets_per_cell))
        return [t for _, _, t in scored[:k]]

    def _alloc_admissible(self, opt: Option) -> bool:
        """p_pos / net_lcb admissibility of one option."""
        if not (math.isfinite(opt.net_lcb) and math.isfinite(opt.cost) and math.isfinite(opt.p_pos)):
            return False
        if opt.n <= 0 or opt.net_lcb <= 0.0:
            return False
        p_min = self.cfg.p_min_push if opt.channel == "push" else self.cfg.p_min
        return opt.p_pos >= p_min

    def options_for(self, sub: SubCell, targets: Optional[list[str]] = None) -> list[Option]:
        """Admissible options of one sub-cell, sorted by net_lcb desc (ties: target rank, channel order)."""
        cell = sub.key[:2]
        if targets is None:
            targets = self._alloc_rank_targets(cell)
        rows: list[tuple[float, int, int, Option]] = []
        for ti, tgt in enumerate(targets):
            for ci, ch in enumerate(self.dv.channels):
                opt = self.model.option(sub, tgt, ch)
                if self._alloc_admissible(opt):
                    rows.append((-float(opt.net_lcb), ti, ci, opt))
        rows.sort(key=lambda r: (r[0], r[1], r[2]))
        return [r[3] for r in rows]

    # ------------------------------------------------------------------ allocate

    def _alloc_build(self, exclude_subs: Iterable[SubKey], extra_cost: Optional[dict]) -> tuple[
            list[SubKey], list[list[Option]], np.ndarray, np.ndarray, np.ndarray]:
        """Matrices (subs x options) of value, cost, contacts; padding has value -inf."""
        excl = set(exclude_subs or ())
        extra = extra_cost or {}
        rank_cache: dict[CellKey, list[str]] = {}
        keys: list[SubKey] = []
        opts: list[list[Option]] = []
        vals: list[list[float]] = []
        for key in sorted(self.dv.subs):
            if key in excl:
                continue
            sub = self.dv.subs[key]
            cell = key[:2]
            if cell not in rank_cache:
                rank_cache[cell] = self._alloc_rank_targets(cell)
            ex = float(extra.get(key, 0.0) or 0.0)
            if math.isnan(ex) or ex < 0.0:  # penalties only; +inf excludes the sub
                ex = 0.0
            row_o: list[Option] = []
            row_v: list[float] = []
            for o in self.options_for(sub, rank_cache[cell]):
                v = float(o.net_lcb) - ex
                if v > 0.0:
                    row_o.append(o)
                    row_v.append(v)
            if row_o:
                keys.append(key)
                opts.append(row_o)
                vals.append(row_v)
        s = len(keys)
        k = max((len(r) for r in opts), default=0)
        V = np.full((s, max(k, 1)), -np.inf)
        C = np.zeros((s, max(k, 1)))
        N = np.zeros((s, max(k, 1)))
        for i, (ro, rv) in enumerate(zip(opts, vals)):
            for j, (o, v) in enumerate(zip(ro, rv)):
                V[i, j] = v
                C[i, j] = max(0.0, float(o.cost))
                N[i, j] = float(o.n)
        return keys, opts, V, C, N

    @staticmethod
    def _alloc_choose(V: np.ndarray, C: np.ndarray, N: np.ndarray, lm: float, lr: float) -> np.ndarray:
        """Per-sub argmax of V - lm*C - lr*N if positive, else -1 (first max wins ties)."""
        if V.shape[0] == 0:
            return np.zeros(0, dtype=np.int64)
        adj = V - lm * C - lr * N
        j = np.argmax(adj, axis=1)
        best = adj[np.arange(V.shape[0]), j]
        return np.where(best > 0.0, j, -1).astype(np.int64)

    @staticmethod
    def _alloc_totals(ch: np.ndarray, V: np.ndarray, C: np.ndarray, N: np.ndarray) -> tuple[float, float, float]:
        """(value, cost, contacts) of a choice vector."""
        on = ch >= 0
        if not on.any():
            return 0.0, 0.0, 0.0
        r = np.nonzero(on)[0]
        c = ch[on]
        return float(V[r, c].sum()), float(C[r, c].sum()), float(N[r, c].sum())

    def _alloc_min_lm(self, V, C, N, lr: float, budget: float) -> tuple[float, np.ndarray]:
        """Smallest lambda_money (bisection) whose choice fits the money budget, given lambda_reach."""
        ch = self._alloc_choose(V, C, N, 0.0, lr)
        if self._alloc_totals(ch, V, C, N)[1] <= budget + _ALLOC_EPS:
            return 0.0, ch
        paid = (C > 0) & np.isfinite(V)
        hi = float(np.max(V[paid] / C[paid])) * 1.01 + 1e-6
        lo = 0.0
        best = self._alloc_choose(V, C, N, hi, lr)
        for _ in range(_ALLOC_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            ch = self._alloc_choose(V, C, N, mid, lr)
            if self._alloc_totals(ch, V, C, N)[1] <= budget + _ALLOC_EPS:
                hi, best = mid, ch
            else:
                lo = mid
        return hi, best

    def _alloc_lagrange(self, V, C, N, budget: float, contacts: float) -> tuple[float, float, np.ndarray]:
        """Nested bisection: outer lambda_reach, inner lambda_money. Returns feasible (lm, lr, choice)."""
        lm, ch = self._alloc_min_lm(V, C, N, 0.0, budget)
        if self._alloc_totals(ch, V, C, N)[2] <= contacts + _ALLOC_EPS:
            return lm, 0.0, ch
        fin = np.isfinite(V) & (N > 0)
        lo, hi = 0.0, float(np.max(V[fin] / N[fin])) * 1.01 + 1e-6
        best_lm, best_ch = self._alloc_min_lm(V, C, N, hi, budget)
        # contacts need not be monotone in lambda_reach once lambda_money is re-solved, so also keep
        # the best-valued feasible iterate as the primal seed (the multipliers stay the bisection ones)
        seed, seed_v = best_ch, self._alloc_totals(best_ch, V, C, N)[0]
        for _ in range(_ALLOC_BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            lm_mid, ch_mid = self._alloc_min_lm(V, C, N, mid, budget)
            val, _, cnt = self._alloc_totals(ch_mid, V, C, N)
            if cnt <= contacts + _ALLOC_EPS:
                hi, best_lm, best_ch = mid, lm_mid, ch_mid
                if val > seed_v + 1e-9:
                    seed, seed_v = ch_mid, val
            else:
                lo = mid
        return best_lm, hi, seed

    def _alloc_repair(self, ch: np.ndarray, V, C, N, budget: float, contacts: float,
                      keep: int = -1) -> np.ndarray:
        """Drop the option with the lowest value per unit of the binding resource until feasible.

        Row `keep` (if >= 0) is dropped only when nothing else is left to drop.
        """
        ch = ch.copy()
        for _ in range(ch.size + 1):
            _, cost, cnt = self._alloc_totals(ch, V, C, N)
            over_m = cost - budget
            over_r = cnt - contacts
            if over_m <= _ALLOC_EPS and over_r <= _ALLOC_EPS:
                break
            r = np.nonzero(ch >= 0)[0]
            c = ch[r]
            v = V[r, c]
            if over_m > _ALLOC_EPS and (over_m / max(budget, 1.0)) >= (over_r / max(contacts, 1.0)):
                res = C[r, c]
            else:
                res = N[r, c]
            cand = res > 0
            if not cand.any():  # binding resource cannot be freed by any chosen option
                res, cand = C[r, c] + N[r, c], np.ones(r.size, dtype=bool)
            ratio = np.where(cand, v / np.maximum(res, 1e-12), np.inf)
            if keep >= 0 and r.size > 1:
                ratio = np.where(r == keep, np.inf, ratio)
                if not np.isfinite(ratio).any():
                    ratio = np.where(r == keep, np.inf, 0.0)
            ch[r[int(np.argmin(ratio))]] = -1
        return ch

    def _alloc_fill(self, ch: np.ndarray, V, C, N, budget: float, contacts: float,
                    by_ratio: bool = False, frozen: Optional[np.ndarray] = None) -> np.ndarray:
        """Greedy: repeatedly apply the feasible add/switch move with the best score.

        Score = value gain, or (by_ratio) gain per unit of normalised resource use
        (money share of the leftover budget + reach share of the leftover contacts).
        Rows in `frozen` (bool mask) are not changed.
        """
        ch = ch.copy()
        s = ch.size
        if s == 0:
            return ch
        rows = np.arange(s)
        frz = np.zeros(s, dtype=bool) if frozen is None else frozen
        for _ in range(4 * s * V.shape[1] + 1):
            _, cost, cnt = self._alloc_totals(ch, V, C, N)
            on = ch >= 0
            cv = np.where(on, V[rows, np.maximum(ch, 0)], 0.0)
            cc = np.where(on, C[rows, np.maximum(ch, 0)], 0.0)
            cn = np.where(on, N[rows, np.maximum(ch, 0)], 0.0)
            gain = V - cv[:, None]
            dc = C - cc[:, None]
            dn = N - cn[:, None]
            ok = (np.isfinite(V) & (gain > 1e-9) & ~frz[:, None]
                  & (cost + dc <= budget + _ALLOC_EPS)
                  & (cnt + dn <= contacts + _ALLOC_EPS))
            if not ok.any():
                break
            if by_ratio:
                use = (np.maximum(dc, 0.0) / max(budget - cost, 1e-9)
                       + np.maximum(dn, 0.0) / max(contacts - cnt, 1e-9))
                free = ok & (use <= 1e-12)
                if free.any():  # moves using no extra resource first, largest gain wins
                    g = np.where(free, gain, -np.inf)
                else:
                    g = np.where(ok, gain / np.maximum(use, 1e-12), -np.inf)
            else:
                g = np.where(ok, gain, -np.inf)
            flat = int(np.argmax(g))
            i, j = divmod(flat, V.shape[1])
            ch[i] = j
        return ch

    def _alloc_refill_best(self, ch: np.ndarray, V, C, N, budget: float, contacts: float,
                           tabu: Iterable[int] = ()) -> np.ndarray:
        """Better of the two greedy fill variants started from `ch`.

        Rows in `tabu` are frozen during a first fill pass and released for a second one
        (so a dropped sub cannot immediately take back the resources it freed).
        """
        tabu = list(tabu)
        res = []
        for by_ratio in (False, True):
            x = ch
            if tabu:
                frozen = np.zeros(ch.size, dtype=bool)
                frozen[tabu] = True
                x = self._alloc_fill(x, V, C, N, budget, contacts, by_ratio=by_ratio, frozen=frozen)
            res.append(self._alloc_fill(x, V, C, N, budget, contacts, by_ratio=by_ratio))
        a, b = res
        return b if self._alloc_totals(b, V, C, N)[0] > self._alloc_totals(a, V, C, N)[0] + 1e-9 else a

    def _alloc_local(self, ch: np.ndarray, V, C, N, budget: float, contacts: float,
                     max_rounds: int = 4) -> np.ndarray:
        """Drop-and-refill local search (drop one, or a pair when few are chosen); keeps strict improvements."""
        best = ch.copy()
        best_v = self._alloc_totals(best, V, C, N)[0]
        if best.size * V.shape[1] > 6000:
            return best
        for _ in range(max_rounds):
            improved = False
            r = np.nonzero(best >= 0)[0]
            order = [int(i) for i in r[np.argsort(V[r, best[r]], kind="stable")]]
            drops: list[tuple[int, ...]] = [(i,) for i in order]
            if len(order) <= 24:
                drops += [(order[a], order[b]) for a in range(len(order)) for b in range(a + 1, len(order))]
            for d in drops:
                if any(best[i] < 0 for i in d):
                    continue
                trial = best.copy()
                trial[list(d)] = -1
                trial = self._alloc_refill_best(trial, V, C, N, budget, contacts, tabu=d)
                tv = self._alloc_totals(trial, V, C, N)[0]
                if tv > best_v + 1e-9:
                    best, best_v, improved = trial, tv, True
            if not improved and best.size * V.shape[1] <= 800:
                improved, best, best_v = self._alloc_swap_in(best, best_v, V, C, N, budget, contacts)
            if not improved:
                break
        return best

    def _alloc_swap_in(self, best: np.ndarray, best_v: float, V, C, N, budget: float,
                       contacts: float) -> tuple[bool, np.ndarray, float]:
        """Force each unchosen feasible option in, repair the rest, refill; first strict improvement wins."""
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if (best[i] == j or not np.isfinite(V[i, j]) or C[i, j] > budget + _ALLOC_EPS
                        or N[i, j] > contacts + _ALLOC_EPS):
                    continue
                trial = best.copy()
                trial[i] = j
                trial = self._alloc_repair(trial, V, C, N, budget, contacts, keep=i)
                trial = self._alloc_refill_best(trial, V, C, N, budget, contacts)
                tv = self._alloc_totals(trial, V, C, N)[0]
                if tv > best_v + 1e-9:
                    return True, trial, tv
        return False, best, best_v

    def allocate(self, budget: float, contacts: int, exclude_subs: Iterable[SubKey] = frozenset(),
                 extra_cost: Optional[dict] = None, polish: bool = True) -> Plan:
        """Best plan (<= 1 option per sub) with total cost <= budget and contacts <= contacts.

        Objective per chosen option = net_lcb - extra_cost[sub]; `Plan.total_net_lcb` reports this objective.
        `polish=False` skips the drop-and-refill / swap-in local search (fast mode for repeated calls).
        """
        budget = float(budget)
        budget = 0.0 if math.isnan(budget) else max(0.0, budget)  # +inf = no money limit
        contacts_f = float(contacts)
        contacts_f = 0.0 if math.isnan(contacts_f) else max(0.0, contacts_f)
        keys, opts, V, C, N = self._alloc_build(exclude_subs, extra_cost)
        if not keys:
            return self._alloc_plan(keys, opts, V, np.zeros(0, dtype=np.int64), 0.0, 0.0)
        lm, lr, ch = self._alloc_lagrange(V, C, N, budget, contacts_f)
        # candidate starts: repaired Lagrangian solution, the unconstrained optimum repaired, empty
        empty = np.full(len(keys), -1, dtype=np.int64)
        starts = [self._alloc_repair(ch, V, C, N, budget, contacts_f),
                  self._alloc_repair(self._alloc_choose(V, C, N, 0.0, 0.0), V, C, N, budget, contacts_f),
                  empty]
        cands = [self._alloc_refill_best(st, V, C, N, budget, contacts_f) for st in starts]
        vals = [self._alloc_totals(c, V, C, N)[0] for c in cands]
        ch = cands[int(np.argmax(vals))]
        if polish:
            ch = self._alloc_local(ch, V, C, N, budget, contacts_f)
        ch = self._alloc_repair(ch, V, C, N, budget, contacts_f)  # safety net
        plan = self._alloc_plan(keys, opts, V, ch, lm, lr)
        if self.log is not None:
            self.log.log("allocate", n_subs=len(keys), n_options=len(plan.options), lambda_money=lm,
                         lambda_reach=lr, total_net_lcb=plan.total_net_lcb, total_cost=plan.total_cost,
                         total_contacts=plan.total_contacts)
        return plan

    @staticmethod
    def _alloc_plan(keys, opts, V, ch: np.ndarray, lm: float, lr: float) -> Plan:
        chosen: list[Option] = []
        total_v = 0.0
        for i, j in enumerate(ch.tolist()):
            if j >= 0:
                chosen.append(opts[i][j])
                total_v += float(V[i, j])
        return Plan(options=chosen, lambda_money=float(lm), lambda_reach=float(lr),
                    total_net_lcb=float(total_v), total_cost=float(sum(o.cost for o in chosen)),
                    total_contacts=int(sum(int(o.n) for o in chosen)))

    def shadow_prices(self, budget: float, contacts: int) -> tuple[float, float]:
        """(lambda_money, lambda_reach) of the current Lagrangian solution."""
        p = self.allocate(budget, contacts, polish=False)
        return p.lambda_money, p.lambda_reach
