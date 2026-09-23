from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip
"""ScoreSimulator: vectorised replica of the organizer scoring mechanics.

Воспроизводит механику scoring_core.score_campaigns: фильтры, сортировка по ID,
лимит кампании 5000 -> общий охват -> денежный бюджет (push бесплатен),
дедупликация абонентов по максимальному лифту, net = gross - cost.
"""

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from agent_src.contract import SimResult  # bundle:strip

_SIM_MAX_PER_CAMPAIGN = 5000
_SIM_MAX_CAMPAIGNS = 10
_SIM_SEG_FILTERS = (
    ("filter_arpu_segment", "arpu_segment"),
    ("filter_data_segment", "data_segment"),
    ("filter_call_segment", "call_segment"),
)


def _sim_is_set(value: Any) -> bool:
    """Копия ``pd.notna(value)`` из scoring_core.apply_filters (только скаляры)."""
    try:
        res = pd.notna(value)
    except Exception:  # pragma: no cover - защитная ветка
        return False
    if isinstance(res, (bool, np.bool_)):
        return bool(res)
    # списочное значение: scoring_core упал бы на `if array`; считаем "заданным" (ничего не совпадает).
    return True


def _sim_explicit(campaign: dict) -> list | None:
    """Явный список id (пилотные кампании) или None."""
    explicit = campaign.get("explicit_ids")
    if isinstance(explicit, (list, tuple, set, np.ndarray)) and len(explicit) > 0:
        return list(explicit)
    return None


def _sim_member(value: Any, allowed: set) -> bool:
    """``value in allowed``, возвращающий False (а не TypeError) для нехешируемых значений."""
    try:
        return value in allowed
    except TypeError:
        return False


def _sim_limit_int(value: Any) -> int:
    """Лимит контактов как int: +inf -> фактически без ограничений, NaN / некорректное -> 0."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0
    if math.isnan(v):
        return 0
    if math.isinf(v):
        return 1 << 62 if v > 0 else 0
    return int(v)


class ScoreSimulator:
    """Точная векторизованная реализация ``score_campaigns`` для заданной функции ratio."""

    def __init__(self, dv: Any, profile: pd.DataFrame):
        self.dv = dv
        prof = profile.sort_values("ID_NUMBER", kind="mergesort").reset_index(drop=True)
        self._profile = prof
        self._n = len(prof)
        self._ids = prof["ID_NUMBER"].to_numpy()
        # индекс абонента по строке: дедупликация по ID_NUMBER (groupby), пустые ID не учитываются в gross
        cust, cust_uniq = pd.factorize(prof["ID_NUMBER"], use_na_sentinel=True)
        self._cust = cust.astype(np.int64)
        self._n_cust = len(cust_uniq)
        self._parr = pd.to_numeric(prof["predicted_arpu"], errors="coerce").to_numpy(dtype=float)
        self._codes: dict[str, np.ndarray] = {}
        self._lookup: dict[str, dict[Any, int]] = {}
        self._uniques: dict[str, list[Any]] = {}
        for col in ("current_tariff", "arpu_segment", "data_segment", "call_segment"):
            codes, uniques = pd.factorize(prof[col], use_na_sentinel=True)
            self._codes[col] = codes.astype(np.int64)
            uniq = list(uniques)
            self._uniques[col] = uniq
            self._lookup[col] = {u: i for i, u in enumerate(uniq)}
        n_arpu = len(self._uniques["arpu_segment"]) + 1
        # id комбинации (tariff, arpu); NaN отображается в дополнительный последний слот
        t = self._codes["current_tariff"].copy()
        a = self._codes["arpu_segment"].copy()
        t[t < 0] = len(self._uniques["current_tariff"])
        a[a < 0] = len(self._uniques["arpu_segment"])
        self._n_arpu = n_arpu
        self._combo = t * n_arpu + a
        self._tariff_codes = set(getattr(dv, "tariff_codes", []) or [])
        self._channels = set(getattr(dv, "channels", []) or [])

    # ------------------------------------------------------------------ фильтры
    def _col_mask(self, col: str, value: Any) -> np.ndarray:
        code = None
        try:
            code = self._lookup[col].get(value)
        except TypeError:  # нехешируемое
            code = None
        if code is None:
            return np.zeros(self._n, dtype=bool)
        return self._codes[col] == code

    def _indices(self, campaign: dict) -> np.ndarray:
        """Позиции строк (по возрастанию ID), отобранные фильтрами кампании."""
        explicit = _sim_explicit(campaign)
        if explicit is not None:
            try:
                hit = self._profile["ID_NUMBER"].isin(explicit).to_numpy(dtype=bool)
            except TypeError:  # нехешируемые элементы
                hit = np.zeros(self._n, dtype=bool)
            return np.flatnonzero(hit)
        mask = np.ones(self._n, dtype=bool)
        for key, col in _SIM_SEG_FILTERS:
            val = campaign.get(key)
            if _sim_is_set(val):
                mask &= self._col_mask(col, val)
        val = campaign.get("filter_current_tariff")
        if _sim_is_set(val):
            wanted = [t.strip() for t in str(val).split(";") if t.strip()]
            codes = [self._lookup["current_tariff"][t] for t in wanted if t in self._lookup["current_tariff"]]
            mask &= np.isin(self._codes["current_tariff"], np.asarray(codes, dtype=np.int64))
        return np.flatnonzero(mask)

    def segment(self, campaign: dict) -> pd.DataFrame:
        """Строки, подходящие под фильтры кампании, отсортированные по ID_NUMBER, до любых ограничений."""
        return self._profile.iloc[self._indices(campaign)]

    # ------------------------------------------------------------------ коэффициенты
    def _ratio_table(self, idx: np.ndarray, target: str, channel: str,
                     ratio_fn: Callable[[str, str, str, str], float]) -> tuple[np.ndarray, int]:
        """Коэффициент лифта по строкам ``idx`` (NaN-ключи передаются в ratio_fn как None) и число ошибок ratio_fn.

        Исключение в ratio_fn даёт ratio 0 для этой комбинации (tariff, arpu) и учитывается, а не пробрасывается.
        """
        combos = self._combo[idx]
        uniq, inv = np.unique(combos, return_inverse=True)
        tu, au = self._uniques["current_tariff"], self._uniques["arpu_segment"]
        vals = np.zeros(len(uniq), dtype=float)
        errors = 0
        for j, c in enumerate(uniq.tolist()):
            ti, ai = divmod(c, self._n_arpu)
            cur = tu[ti] if ti < len(tu) else None
            seg = au[ai] if ai < len(au) else None
            try:
                r = float(ratio_fn(cur, seg, target, channel))
            except Exception:
                r = 0.0
                errors += 1
            vals[j] = r
        return vals[inv], errors

    # ------------------------------------------------------------------ симуляция
    def simulate(self, campaigns: list[dict], ratio_fn: Callable[[str, str, str, str], float],
                 budget: float, contacts: int) -> SimResult:
        """Оценивает ``campaigns`` по порядку при стартовых лимитах ``budget`` / ``contacts``.

        Кампании с неизвестным target/channel получают нулевую запись ``dropped`` (сохраняет
        выравнивание индексов ``per_campaign`` с ``campaigns``) и далее пропускаются, как в sanitize_campaigns.
        Кампания может содержать ``explicit_ids`` (пилотный режим) — тогда фильтры игнорируются.
        Лимиты: +inf означает без ограничений, NaN / отрицательное — ничего не осталось.
        Отличие от организаторов (только для устойчивости): неконечные лифты по абоненту считаются 0
        (организаторы обнуляют только NaN); исключения ratio_fn дают ratio 0 и учитываются в
        ``per_campaign[i]["ratio_errors"]``.
        """
        best = np.full(self._n_cust, -np.inf)
        touched = np.zeros(self._n_cust, dtype=bool)
        reach_left = _sim_limit_int(contacts)
        money_left = float(budget)
        if math.isnan(money_left) or money_left == -math.inf:
            money_left = 0.0
        total_cost = 0.0
        total_contacts = 0
        per_campaign: list[dict] = []
        within = True
        n_valid = 0
        for i, camp in enumerate(campaigns or []):
            if not isinstance(camp, dict):
                continue
            target, channel = camp.get("target_tariff"), camp.get("channel")
            name = camp.get("campaign_name", f"campaign_{i}")
            if not (_sim_member(target, self._tariff_codes) and _sim_member(channel, self._channels)):
                per_campaign.append({"name": name, "n_segment": 0, "n_contacted": 0, "cost": 0.0,
                                     "gross": 0.0, "capped_campaign": False, "capped_reach": False,
                                     "capped_money": False, "dropped": True})
                continue
            n_valid += 1
            cpc = float(self.dv.cost(channel))
            idx = self._indices(camp)
            n_seg = len(idx)
            cap_c = len(idx) > _SIM_MAX_PER_CAMPAIGN
            if cap_c:
                idx = idx[:_SIM_MAX_PER_CAMPAIGN]
            cap_r = len(idx) > reach_left
            if cap_r:
                idx = idx[:max(reach_left, 0)]
            cap_m = False
            if cpc > 0 and math.isfinite(money_left):
                affordable = int(money_left // cpc)
                if len(idx) > affordable:
                    cap_m = True
                    idx = idx[:max(affordable, 0)]
            n = len(idx)
            reach_left -= n
            cost = n * cpc
            money_left -= cost
            total_cost += cost
            total_contacts += n
            if cap_r or cap_m:
                within = False
            gross_c = 0.0
            n_err = 0
            if n > 0:
                ratio, n_err = self._ratio_table(idx, target, channel, ratio_fn)
                with np.errstate(invalid="ignore", over="ignore"):
                    lift = ratio * self._parr[idx]
                lift = np.where(np.isfinite(lift), lift, 0.0)
                gross_c = float(lift.sum())
                cust = self._cust[idx]
                ok = cust >= 0
                np.maximum.at(best, cust[ok], lift[ok])
                touched[cust[ok]] = True
            entry = {"name": name, "n_segment": int(n_seg), "n_contacted": int(n),
                     "cost": float(cost), "gross": gross_c, "capped_campaign": bool(cap_c),
                     "capped_reach": bool(cap_r), "capped_money": bool(cap_m)}
            if n_err:
                entry["ratio_errors"] = int(n_err)
            per_campaign.append(entry)
        if n_valid > _SIM_MAX_CAMPAIGNS:
            within = False
        gross = float(best[touched].sum()) if touched.any() else 0.0
        return SimResult(gross=gross, cost=float(total_cost), contacts=int(total_contacts),
                         net=gross - float(total_cost), per_campaign=per_campaign, within_limits=within)
