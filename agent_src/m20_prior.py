from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PriorBuilder: слабый иерархический априор на базовый lift ratio каждого arm.
# Априор: иерархическая усадка arm -> (from, seg) -> global, Beta-неопределённость доли.
# ---------------------------------------------------------------------------

_PR_ARPU_BINS = (-np.inf, 1000.0, 5000.0, np.inf)
_PR_ARPU_LABELS = ("LOW", "MID", "HIGH")
_PR_MIN_PREV = 100.0
_PR_CHG_LO, _PR_CHG_HI = -1.0, 3.0
_PR_NOHIST_SD = 0.25
_PR_PRICE_SHIFT = 0.02  # небольшой сдвиг базового ratio / change по знаку цены, когда истории arm нет
_PR_REQUIRED = ("AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M", "tariff_plan_code_from", "tariff_plan_code_to")


def _pr_sign(x: float) -> float:
    """Знак конечного float (0 для NaN / 0)."""
    if x != x or x == 0:
        return 0.0
    return 1.0 if x > 0 else -1.0


def _pr_num(x: Any, default: float, lo: float = -math.inf, hi: float = math.inf) -> float:
    """Конечный float, ограниченный [lo, hi]; ``default`` для нечислового / неконечного входа."""
    try:
        v = float(x)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return min(max(v, lo), hi)


def _pr_beta_mean_var(a: float, b: float) -> tuple[float, float]:
    """Среднее и дисперсия Beta(a, b)."""
    s = a + b
    return a / s, a * b / (s * s * (s + 1.0))


def _pr_prepare(history: pd.DataFrame) -> pd.DataFrame:
    """Очистка строк истории: сегмент, фильтр PREV, обрезанное относительное изменение."""
    df = history.loc[:, list(_PR_REQUIRED)].copy()
    prev = pd.to_numeric(df["AVG_ARPU_PREV_3M"], errors="coerce")
    nxt = pd.to_numeric(df["AVG_ARPU_NEXT_3M"], errors="coerce")
    prev = prev.where(np.isfinite(prev.to_numpy(dtype=float)))
    nxt = nxt.where(np.isfinite(nxt.to_numpy(dtype=float)))
    frm = df["tariff_plan_code_from"].astype("string").str.strip()
    to = df["tariff_plan_code_to"].astype("string").str.strip()
    seg = pd.cut(prev, bins=list(_PR_ARPU_BINS), labels=list(_PR_ARPU_LABELS))
    out = pd.DataFrame(
        {
            "frm": frm,
            "to": to,
            "seg": seg.astype("string"),
            "chg": ((nxt - prev) / prev).clip(_PR_CHG_LO, _PR_CHG_HI),
        }
    )
    ok = (prev >= _PR_MIN_PREV) & nxt.notna() & frm.notna() & to.notna() & out["seg"].notna()
    ok = ok & (frm != "") & (to != "")
    out = out.loc[ok.fillna(False).astype(bool)]
    out = out.loc[np.isfinite(out["chg"].to_numpy(dtype=float))]
    out = out.astype({"frm": object, "to": object, "seg": object, "chg": float})
    # канонический порядок строк -> агрегаты не зависят от порядка входных строк (побитовый детерминизм)
    return out.sort_values(["frm", "seg", "to", "chg"], kind="mergesort").reset_index(drop=True)


class PriorBuilder:
    """Строит Prior для каждого arm (cell x target != current).

    mu/sd заданы для базового lift ratio (change * share) при множителе канала 1.0.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._share0 = _pr_num(getattr(cfg, "conv_prior_default", 0.3), 0.3, 1e-6, 1.0)
        self._tau = _pr_num(getattr(cfg, "tau", 0.2), 0.2, 0.0)
        self._k = _pr_num(getattr(cfg, "shrink_k", 20.0), 20.0, 0.0, 1e9)

    # -- публичные ------------------------------------------------------------

    def build(self, history: pd.DataFrame | None, dv: Any) -> dict[ArmKey, Prior]:
        """Один Prior на каждую пару из dv.cells x dv.targets_for(cell). Никогда не бросает исключений."""
        try:
            arms = self._arms(dv)
        except Exception:
            return {}
        stats = None
        if history is not None:
            try:
                stats = self._stats(history)
            except Exception:
                stats = None
        out: dict[ArmKey, Prior] = {}
        prices = self._prices(dv)
        for arm in arms:
            try:
                if stats is None:
                    out[arm] = self._no_history(arm, prices)
                else:
                    out[arm] = self._from_history(arm, stats, prices)
            except Exception:
                out[arm] = Prior(0.0, _PR_NOHIST_SD, self._share0, 0, "none")
        return out

    # -- вспомогательные -----------------------------------------------------------

    @staticmethod
    def _arms(dv: Any) -> list[ArmKey]:
        """Все arm (from, seg, to), отсортированные."""
        arms: list[ArmKey] = []
        for cell in sorted(dv.cells):
            for tgt in dv.targets_for(cell):
                if tgt != cell[0]:
                    arms.append((str(cell[0]), str(cell[1]), str(tgt)))
        return sorted(set(arms))

    @staticmethod
    def _prices(dv: Any) -> dict[str, float]:
        """Цены тарифов (пусто при ошибке)."""
        try:
            return {str(k): float(v) for k, v in dict(dv.tariff_price).items()}
        except Exception:
            return {}

    @staticmethod
    def _price_sign(arm: ArmKey, prices: dict[str, float]) -> float | None:
        """sign(price_to - price_from) или None, если цена неизвестна."""
        pf, pt = prices.get(arm[0]), prices.get(arm[2])
        if pf is None or pt is None or not (math.isfinite(pf) and math.isfinite(pt)):
            return None
        return _pr_sign(pt - pf)

    def _no_history(self, arm: ArmKey, prices: dict[str, float]) -> Prior:
        """Слабый априор: mu = небольшой сдвиг по знаку цены, sd = 0.25."""
        share = self._share0
        sgn = self._price_sign(arm, prices)
        if sgn is None:
            return Prior(0.0, _PR_NOHIST_SD, share, 0, "none")
        return Prior(_PR_PRICE_SHIFT * sgn, _PR_NOHIST_SD, share, 0, "price")

    def _stats(self, history: pd.DataFrame) -> dict | None:
        """Агрегирует статистики arm / cell / global; None, если непригодны."""
        if not isinstance(history, pd.DataFrame) or any(c not in history.columns for c in _PR_REQUIRED):
            return None
        df = _pr_prepare(history)
        if len(df) == 0:
            return None
        chg = df["chg"].to_numpy(dtype=float)
        g_mean = float(chg.mean())
        g_var = float(chg.var(ddof=0)) if len(chg) > 1 else 0.0

        cell_g = df.groupby(["frm", "seg"], sort=True)["chg"].agg(["size", "mean"])
        cells = {(str(f), str(s)): (int(r["size"]), float(r["mean"])) for (f, s), r in cell_g.iterrows()}

        arm_df = df.loc[df["frm"] != df["to"]]
        arm_g = arm_df.groupby(["frm", "seg", "to"], sort=True)["chg"].agg(["size", "mean", "var"])
        arms = {
            (str(f), str(s), str(t)): (int(r["size"]), float(r["mean"]), float(r["var"]) if r["size"] > 1 else float("nan"))
            for (f, s, t), r in arm_g.iterrows()
        }
        shares = [c / cells[(a[0], a[1])][0] for a, (c, _, _) in arms.items() if cells.get((a[0], a[1]), (0,))[0] > 0]
        g_share = float(np.median(shares)) if shares else self._share0
        g_share = min(max(g_share, 1e-4), 1.0)
        return {"g_mean": g_mean, "g_var": g_var, "cells": cells, "arms": arms, "g_share": g_share}

    def _from_history(self, arm: ArmKey, st: dict, prices: dict[str, float]) -> Prior:
        """Иерархический shrinkage-априор для одного arm.

        Наблюдаемый arm (c > 0): share ~ Beta(c+1, tot-c+1). Ненаблюдаемый arm (c == 0): скорер
        переходит на правило с медианной конверсией arm, поэтому share = глобальная медианная share
        (слабо, pseudo-count 2), и добавляется сдвиг по знаку цены в единицах base ratio.
        """
        k, tau = self._k, self._tau
        g_mean, g_var = st["g_mean"], st["g_var"]
        tot, m_cell = st["cells"].get((arm[0], arm[1]), (0, g_mean))
        c, m_arm, v_arm = st["arms"].get(arm, (0, g_mean, float("nan")))

        # change: arm -> (from, seg) -> global
        m_cell_s = (tot * m_cell + k * g_mean) / (tot + k) if tot + k > 0 else g_mean
        m_s = (c * m_arm + k * m_cell_s) / (c + k) if c + k > 0 else m_cell_s
        dof = max(c - 1, 0)
        v_arm = v_arm if v_arm == v_arm else 0.0
        s2 = (dof * v_arm + k * g_var) / (dof + k) if dof + k > 0 else g_var
        var_m = s2 / (c + k) if c + k > 0 else s2

        if c > 0:
            share, var_share = _pr_beta_mean_var(c + 1.0, tot - c + 1.0)
            source = "arm"
        else:
            share = st["g_share"]
            var_share = share * (1.0 - share) / 3.0  # слабый: Beta с pseudo-count 2
            source = "cell" if tot > 0 else "global"
        share = min(max(float(share), 1e-6), 1.0)

        mu = m_s * share
        if c == 0:
            sgn = self._price_sign(arm, prices)
            if sgn is not None:
                mu += _PR_PRICE_SHIFT * sgn
        var = var_m * share * share + m_s * m_s * var_share + tau * tau
        sd = math.sqrt(var) if math.isfinite(var) and var > 1e-12 else 1e-6
        if not math.isfinite(mu):
            return Prior(0.0, _PR_NOHIST_SD, self._share0, 0, "none")
        return Prior(float(mu), float(sd), share, int(c), source)
