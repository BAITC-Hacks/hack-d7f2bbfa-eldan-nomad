from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip

# m60 CampaignPacker: превращает Plan аллокатора в <= 10 непересекающихся валидных кампаний.
# Упаковка плана подячеек в ≤10 непересекающихся кампаний с пересимуляцией.

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class _PackCamp:
    """Внутренняя кампания: фильтры + оценка ценности (из опций плана)."""

    target: str
    channel: str
    arpu: str
    data: Optional[str]
    call: Optional[str]
    tariffs: tuple  # исходные тарифы, естественный порядок
    est_net: float = 0.0
    est_n: int = 0

    def key(self) -> tuple:
        """Детерминированный ключ сортировки / идентичности."""
        return (self.target, self.channel, self.arpu, self.data or "", self.call or "", self.tariffs)


_PACK_HARD_MAX_CAMPAIGNS = 10  # MAX_CAMPAIGNS организатора
_PACK_HARD_MAX_PER_CAMPAIGN = 5000  # MAX_CUSTOMERS_PER_CAMPAIGN организатора


def _pack_limit(v: Any, hard: int) -> int:
    """Лимит из конфига, ограниченный диапазоном [1, жёсткий лимит организатора]."""
    try:
        return max(1, min(int(v), hard))
    except (TypeError, ValueError):
        return hard


def _pack_short(code: str) -> str:
    """'tariff_12' -> 't12'; прочие коды без изменений."""
    m = re.fullmatch(r"tariff_(\d+)", str(code))
    return f"t{m.group(1)}" if m else str(code)


def _pack_nat_key(s: str) -> tuple:
    """Ключ естественной сортировки ('tariff_2' < 'tariff_10')."""
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(s)))


class CampaignPacker:
    """Упаковывает опции Plan в кампании (группировка, огрубление, разбиение, пересимуляция)."""

    def __init__(self, cfg: Config, dv: Any, sim: Any, log: Optional[RunLog] = None):
        self.cfg = cfg
        self.dv = dv
        self.sim = sim
        self.log = log
        self.last_sim: Optional[SimResult] = None
        self.last_report: dict = {}
        self._max_c = _pack_limit(cfg.max_campaigns, _PACK_HARD_MAX_CAMPAIGNS)
        self._max_per = _pack_limit(cfg.max_per_campaign, _PACK_HARD_MAX_PER_CAMPAIGN)
        self._size_cache: dict[tuple, int] = {}
        codes = list(getattr(dv, "tariff_codes", []) or [])
        self._order = {c: i for i, c in enumerate(codes)}
        # грубый префиксный индекс: (tariff, arpu) и (tariff, arpu, data) -> ключи подячеек dv
        self._subs_by_ta: dict[tuple, list] = {}
        self._subs_by_tad: dict[tuple, list] = {}
        for sk in sorted(getattr(dv, "subs", {}) or {}):
            self._subs_by_ta.setdefault(sk[:2], []).append(sk)
            self._subs_by_tad.setdefault(sk[:3], []).append(sk)

    # ------------------------------------------------------------------ вспомогательные

    def _tkey(self, code: str) -> tuple:
        return (0, self._order[code], "") if code in self._order else (1, 0, _pack_nat_key(code))

    def _sorted_tariffs(self, codes: Iterable[str]) -> tuple:
        return tuple(sorted(set(codes), key=self._tkey))

    def _to_dict(self, c: _PackCamp, name: Optional[str] = None) -> dict:
        return campaign_dict(
            campaign_name=name or self._name(c, 0),
            filter_arpu_segment=c.arpu,
            filter_data_segment=c.data,
            filter_call_segment=c.call,
            filter_current_tariff=";".join(c.tariffs),
            target_tariff=c.target,
            channel=c.channel,
        )

    def _name(self, c: _PackCamp, idx: int) -> str:
        shorts = [_pack_short(t) for t in c.tariffs]
        if len(shorts) > 6:
            shorts = shorts[:6] + [f"{len(c.tariffs) - 6}more"]
        return (f"c{idx:02d}_{_pack_short(c.target)}_{c.channel}_{c.arpu}_{c.data or 'ALL'}_"
                f"{c.call or 'ALL'}_{'+'.join(shorts)}")

    def _size(self, c: _PackCamp) -> int:
        """Точный размер сегмента (до лимитов) через симулятор (кэшируется; не зависит от target/channel)."""
        k = (c.arpu, c.data, c.call, c.tariffs)
        if k not in self._size_cache:
            try:
                v = int(len(self.sim.segment(self._to_dict(c))))
            except Exception:  # noqa: BLE001 - откат к счётчикам подячеек dv
                subs = getattr(self.dv, "subs", {}) or {}
                v = int(sum(s.n for sk, s in subs.items() if self._covers(c, sk)))
            self._size_cache[k] = v
        return self._size_cache[k]

    @staticmethod
    def _covers(c: _PackCamp, sk: tuple) -> bool:
        return (sk[0] in c.tariffs and sk[1] == c.arpu and (c.data is None or sk[2] == c.data)
                and (c.call is None or sk[3] == c.call))

    def _estimate(self, c: _PackCamp, assign: dict) -> _PackCamp:
        """Пересчитать est_net / est_n по назначенным опциям, покрытым c."""
        net, n = 0.0, 0
        for sk, opt in assign.items():
            if opt.target == c.target and opt.channel == c.channel and self._covers(c, sk):
                v = float(opt.net_mean)
                net += v if math.isfinite(v) else 0.0
                n += int(opt.n)
        c.est_net, c.est_n = net, n
        return c

    # ------------------------------------------------------------------ шаги

    def _group(self, assign: dict) -> list[_PackCamp]:
        groups: dict[tuple, list[str]] = {}
        for sk, opt in assign.items():
            groups.setdefault((opt.target, opt.channel, sk[1], sk[2], sk[3]), []).append(sk[0])
        out = []
        for (tg, ch, a, d, cl) in sorted(groups):
            tariffs = self._sorted_tariffs(t for t in groups[(tg, ch, a, d, cl)] if t != tg)
            if tariffs:
                out.append(self._estimate(_PackCamp(tg, ch, a, d, cl, tariffs), assign))
        return out

    def _split(self, c: _PackCamp, assign: dict) -> list[_PackCamp]:
        """Разбить кампанию, сегмент которой превышает max_per_campaign (сначала по тарифам, затем по сегментам)."""
        limit = self._max_per
        size = self._size(c)
        if size <= limit:
            return [c]
        if len(c.tariffs) > 1:
            sizes = {t: self._size(_PackCamp(c.target, c.channel, c.arpu, c.data, c.call, (t,)))
                     for t in c.tariffs}
            bins: list[list] = []  # [total, [tariffs]]
            for t in sorted(c.tariffs, key=lambda t: (-sizes[t], self._tkey(t))):
                for b in bins:
                    if b[0] + sizes[t] <= limit:
                        b[0] += sizes[t]
                        b[1].append(t)
                        break
                else:
                    bins.append([sizes[t], [t]])
            out: list[_PackCamp] = []
            for _, ts in bins:
                part = _PackCamp(c.target, c.channel, c.arpu, c.data, c.call, self._sorted_tariffs(ts))
                out.extend(self._split(self._estimate(part, assign), assign))
            return out
        if c.data is None or c.call is None:
            # уточняем самый грубый незаданный сегмент; части без назначенных подячеек пропускаются
            out = []
            if c.data is None:
                parts = [(d, None) for d in DATA_SEGMENTS]
            else:
                parts = [(c.data, cl) for cl in CALL_SEGMENTS]
            for d, cl in parts:
                part = self._estimate(_PackCamp(c.target, c.channel, c.arpu, d, cl, c.tariffs), assign)
                if part.est_n > 0:
                    out.extend(self._split(part, assign))
            return out
        return [c]  # одна полностью отфильтрованная подячейка > лимита: скоринг обрежет её до 5000

    def _merge_candidates(self, camps: list[_PackCamp], assign: dict) -> list[tuple]:
        """Все точные слияния-огрубления: (reduction, level_rank, key, new_camps)."""
        cands = []
        pairs = sorted({(c.target, c.channel, c.arpu) for c in camps})
        for tg, ch, a in pairs:
            same = [c for c in camps if (c.target, c.channel, c.arpu) == (tg, ch, a)]
            for level, data in [(0, d) for d in DATA_SEGMENTS] + [(1, None)]:
                # уровень 0: убираем разбиение по call (грубый фильтр data=d, call=None); уровень 1: убираем и разбиение по data
                finer = [c for c in same if (c.data == data if data is not None else True)
                         and not (c.data == data and c.call is None)]
                if not finer:
                    continue
                existing = [c for c in same if c.data == data and c.call is None]
                present = sorted({t for c in finer for t in c.tariffs}, key=self._tkey)
                elig = []
                for t in present:
                    subs = (self._subs_by_tad.get((t, a, data), []) if data is not None
                            else self._subs_by_ta.get((t, a), []))
                    if subs and all(sk in assign and assign[sk].target == tg and assign[sk].channel == ch
                                    for sk in subs):
                        elig.append(t)
                if not elig:
                    continue
                elig_set = set(elig)
                new_finer = []
                emptied = 0
                for c in finer:
                    rest = tuple(t for t in c.tariffs if t not in elig_set)
                    if rest == c.tariffs:
                        new_finer.append(c)
                    elif rest:
                        new_finer.append(self._estimate(_PackCamp(tg, ch, a, c.data, c.call, rest), assign))
                    else:
                        emptied += 1
                base_tariffs = existing[0].tariffs if existing else ()
                coarse = self._estimate(
                    _PackCamp(tg, ch, a, data, None, self._sorted_tariffs(base_tariffs + tuple(elig))), assign)
                reduction = emptied - (0 if existing else 1)
                if reduction <= 0:
                    continue
                size = self._size(coarse)
                # только точные: строки с NaN в data/call (вне подячеек) просочились бы после снятия фильтра
                expected = (self._size(existing[0]) if existing else 0) + sum(
                    int(self.dv.subs[sk].n) for t in elig
                    for sk in (self._subs_by_tad.get((t, a, data), []) if data is not None
                               else self._subs_by_ta.get((t, a), [])))
                if size > self._max_per or size != expected:
                    continue
                touched = {id(c) for c in finer} | {id(c) for c in existing}
                new_camps = [c for c in camps if id(c) not in touched] + new_finer + [coarse]
                cands.append((reduction, level, (tg, ch, a, data or ""), new_camps))
        return cands

    def _coarsen(self, camps: list[_PackCamp], assign: dict) -> tuple[list[_PackCamp], list[dict]]:
        steps: list[dict] = []
        max_c = self._max_c
        while len(camps) > max_c:
            cands = self._merge_candidates(camps, assign)
            if cands:
                best = min(cands, key=lambda x: (-x[0], x[1], x[2]))
                camps = best[3]
                steps.append({"step": "merge", "level": ("call", "data")[best[1]], "key": list(best[2]),
                              "reduction": best[0]})
                continue
            worst = min(camps, key=lambda c: (c.est_net, c.key()))
            camps = [c for c in camps if c is not worst]
            # удалённые подячейки выходят из назначения, чтобы последующие слияния не вернули их незаметно
            for sk in [sk for sk, o in assign.items()
                       if o.target == worst.target and o.channel == worst.channel and self._covers(worst, sk)]:
                del assign[sk]
            steps.append({"step": "drop", "campaign": self._name(worst, 0), "est_net": round(worst.est_net, 2)})
        return camps, steps

    def _ratio_fn(self, model: Any) -> Callable[[str, str, str, str], float]:
        cache: dict[tuple, float] = {}

        def fn(cur: str, arpu: str, target: str, channel: str) -> float:
            k = (cur, arpu, target, channel)
            if k not in cache:
                try:
                    v = float(model.posterior((cur, arpu, target), channel).mean)
                except Exception:  # noqa: BLE001 - неизвестный arm -> нет эффекта
                    v = 0.0
                cache[k] = v if math.isfinite(v) else 0.0
            return cache[k]

        return fn

    def _prune(self, dicts: list[dict], ratio_fn: Callable, budget: float, contacts: int) -> tuple[list[dict], list]:
        """Удалить кампании с симулированным предельным net <= 0 (сначала худшие, с пересимуляцией)."""
        dropped = []
        while dicts:
            full = self.sim.simulate(dicts, ratio_fn, budget, contacts)
            margins = []
            for i in range(len(dicts)):
                rest = dicts[:i] + dicts[i + 1:]
                net_rest = self.sim.simulate(rest, ratio_fn, budget, contacts).net if rest else 0.0
                margins.append(float(full.net) - float(net_rest))
            i_min = min(range(len(dicts)), key=lambda i: (margins[i], i))
            if margins[i_min] > 1e-9:
                self.last_sim = full
                break
            dropped.append({"campaign": dicts[i_min]["campaign_name"], "marginal_net": round(margins[i_min], 2)})
            dicts = dicts[:i_min] + dicts[i_min + 1:]
        if not dicts:
            self.last_sim = None
        return dicts, dropped

    # ------------------------------------------------------------------ публичное API

    def pack(self, plan: Plan, model: Any, budget: float, contacts: int) -> list[dict]:
        """Plan -> упорядоченный список из <= max_campaigns непересекающихся кампаний-словарей ([], если ничего не окупается)."""
        self.last_sim = None
        self._size_cache = {}
        budget = float(budget) if budget is not None and math.isfinite(float(budget)) else 0.0
        budget = max(budget, 0.0)
        try:
            contacts = max(int(contacts), 0)
        except (TypeError, ValueError, OverflowError):
            contacts = 0
        assign: dict[tuple, Option] = {}
        for opt in sorted(plan.options, key=lambda o: (tuple(o.sub), o.target, o.channel)):
            sk = tuple(opt.sub)
            if sk[0] == opt.target or sk in assign:
                continue  # никогда target == from; не более одной опции на подячейку
            assign[sk] = opt
        n_options = len(assign)
        camps = self._group(assign)
        n_groups = len(camps)
        split: list[_PackCamp] = []
        for c in camps:
            split.extend(self._split(c, assign))
        camps, steps = self._coarsen(split, assign)
        camps.sort(key=lambda c: (-(c.est_net / c.est_n if c.est_n > 0 else -math.inf), c.key()))
        dicts = [self._to_dict(c, self._name(c, i + 1)) for i, c in enumerate(camps)]
        ratio_fn = self._ratio_fn(model)
        dicts, dropped = self._prune(dicts, ratio_fn, budget, contacts) if dicts else ([], [])
        # перенумеровываем после отсечения, чтобы имена шли подряд
        for i, d in enumerate(dicts):
            d["campaign_name"] = "c%02d%s" % (i + 1, d["campaign_name"][3:])
        if self.last_sim is not None and len(self.last_sim.per_campaign) == len(dicts):
            for pc, d in zip(self.last_sim.per_campaign, dicts):
                pc["name"] = d["campaign_name"]
        self.last_report = {
            "n_options": n_options, "n_groups": n_groups, "n_after_split": len(split),
            "coarsen_steps": steps, "pruned": dropped, "n_campaigns": len(dicts),
            "sim_net": None if self.last_sim is None else round(float(self.last_sim.net), 2),
        }
        if self.log is not None:
            self.log.log("pack", **self.last_report)
        return dicts

    # ------------------------------------------------------------------ грубая упаковка (расширение)

    def _coarse_items(self, model: Any, z: float) -> dict[tuple, list[tuple]]:
        """(target, channel, arpu) -> [(from_tariff, n, sum_p, ratio_adj)] по ячейкам с известным эффектом."""
        cells = getattr(self.dv, "cells", {}) or {}
        chans = list(getattr(self.dv, "channels", []) or [])
        out: dict[tuple, list[tuple]] = {}
        for ck in sorted(cells):
            cell = cells[ck]
            n, sp = int(getattr(cell, "n", 0) or 0), float(getattr(cell, "sum_p", 0.0) or 0.0)
            if n <= 0 or not math.isfinite(sp):
                continue
            for tg in self.dv.targets_for(ck):
                if tg == ck[0]:
                    continue
                for ch in chans:
                    try:
                        post = model.posterior((ck[0], ck[1], tg), ch)
                        r = float(post.mean) - z * float(post.sd)
                    except Exception:  # noqa: BLE001 - неизвестный arm: пропуск
                        continue
                    if math.isfinite(r) and r > 0.0:
                        out.setdefault((tg, ch, ck[1]), []).append((ck[0], n, sp, r))
        return out

    def _coarse_greedy(self, items: dict, budget: float, contacts: int, lm: float, lr: float) -> tuple[float, list]:
        """Жадный отбор кампаний на уровне ячеек для теневых цен (lm — деньги, lr — охват на контакт)."""
        best: dict[tuple, float] = {}  # (from, arpu) -> лучший уже покрытый ratio
        camps: list[tuple] = []  # (target, channel, arpu, tariffs, n, cost, gain)
        b_left, c_left = float(budget), int(contacts)
        value = 0.0
        while len(camps) < self._max_c:
            pick = None
            for (tg, ch, a) in sorted(items):
                cost_c = float(self.dv.cost(ch))
                rows = []
                for (t, n, sp, r) in items[(tg, ch, a)]:
                    gain = sp * (r - best.get((t, a), 0.0))
                    adj = gain - cost_c * n * (1.0 + lm) - lr * n
                    if adj > 0.0:
                        rows.append((adj / n, t, n, gain, adj))
                if not rows:
                    continue
                rows.sort(key=lambda x: (-x[0], self._tkey(x[1])))
                size, spend, score, gsum, ts = 0, 0.0, 0.0, 0.0, []
                for _, t, n, gain, adj in rows:
                    if size + n > self._max_per or size + n > c_left or (cost_c > 0 and spend + cost_c * n > b_left):
                        continue
                    size += n
                    spend += cost_c * n
                    score += adj
                    gsum += gain
                    ts.append(t)
                if not ts:
                    continue
                cand = (score, tg, ch, a, ts, size, spend, gsum)
                if pick is None or score > pick[0] + 1e-9:
                    pick = cand
            if pick is None:
                break
            _, tg, ch, a, ts, size, spend, gsum = pick
            r_of = {t: r for (t, _n, _sp, r) in items[(tg, ch, a)]}
            for t in ts:
                best[(t, a)] = max(best.get((t, a), 0.0), r_of[t])
            camps.append((tg, ch, a, self._sorted_tariffs(ts), size, spend, gsum - spend))
            b_left -= spend
            c_left -= size
            value += gsum - spend
        return value, camps

    def pack_coarse(self, model: Any, budget: float, contacts: int, z: Optional[float] = None) -> list[dict]:
        """Жадная упаковка на уровне ячеек: кампании (target, channel, arpu, список исходных тарифов) без фильтра data/call.

        Эффекты зависят от (from, arpu, target, channel), поэтому кампании на всю ячейку покрывают целые ячейки с одинаковым
        ratio. Ratio скорректированы на риск (mean − z·sd); перебирается небольшая сетка теневых цен и сохраняется лучшее
        значение, затем отсекаются кампании с неположительным симулированным предельным net (апостериорные средние).
        """
        self.last_sim = None
        self._size_cache = {}
        try:
            budget = max(float(budget), 0.0) if math.isfinite(float(budget)) else 0.0
        except (TypeError, ValueError):
            budget = 0.0
        try:
            contacts = max(int(contacts), 0)
        except (TypeError, ValueError, OverflowError):
            contacts = 0
        zz = float(self.cfg.z_risk if z is None else z)
        items = self._coarse_items(model, zz if math.isfinite(zz) else 0.0)
        if not items:
            return []
        dens = sorted(sp * r / n for lst in items.values() for (_t, n, sp, r) in lst)
        base = dens[len(dens) // 2] if dens else 0.0
        best_val, best_camps, best_grid = -math.inf, [], None
        for lm in (0.0, 0.5, 1.0, 2.0, 4.0):
            for lr in (0.0, 0.25, 0.5, 1.0, 2.0):
                val, camps = self._coarse_greedy(items, budget, contacts, lm, lr * base)
                if val > best_val + 1e-9:
                    best_val, best_camps, best_grid = val, camps, (lm, lr)
        best_camps = sorted(best_camps, key=lambda c: (-(c[6] / c[4] if c[4] else 0.0), c[:4]))
        dicts = []
        for i, (tg, ch, a, ts, _n, _cost, _v) in enumerate(best_camps):
            pc = _PackCamp(tg, ch, a, None, None, ts)
            dicts.append(self._to_dict(pc, self._name(pc, i + 1)))
        dicts, dropped = self._prune(dicts, self._ratio_fn(model), budget, contacts) if dicts else ([], [])
        for i, d in enumerate(dicts):
            d["campaign_name"] = "c%02d%s" % (i + 1, d["campaign_name"][3:])
        self.last_report = {"mode": "coarse", "z": zz, "grid": best_grid, "value_adj": round(best_val, 2),
                            "pruned": dropped, "n_campaigns": len(dicts),
                            "sim_net": None if self.last_sim is None else round(float(self.last_sim.net), 2)}
        if self.log is not None:
            self.log.log("pack_coarse", **self.last_report)
        return dicts

    def empty_campaign(self) -> dict:
        """Валидная push-кампания, фильтры которой совпадают с 0 клиентов (пустой запасной вариант)."""
        codes = list(getattr(self.dv, "tariff_codes", []) or [])
        chans = list(getattr(self.dv, "channels", []) or [])
        channel = "push" if "push" in chans or not chans else chans[0]
        subs = getattr(self.dv, "subs", {}) or {}
        for t in codes:
            others = [c for c in codes if c != t]
            if not others:
                break
            for a in ARPU_SEGMENTS:
                for d in DATA_SEGMENTS:
                    for cl in CALL_SEGMENTS:
                        if (t, a, d, cl) not in subs:
                            return campaign_dict(
                                campaign_name="c01_empty_noop", filter_arpu_segment=a, filter_data_segment=d,
                                filter_call_segment=cl, filter_current_tariff=t, target_tariff=others[0],
                                channel=channel)
        # все комбинации заняты (или < 2 тарифов): наименьшая подячейка на push, target != from по возможности
        if subs and len(codes) >= 2:
            sk = min(subs, key=lambda k: (subs[k].n, k))
            target = next(c for c in codes if c != sk[0])
            return campaign_dict(
                campaign_name="c01_empty_noop", filter_arpu_segment=sk[1], filter_data_segment=sk[2],
                filter_call_segment=sk[3], filter_current_tariff=sk[0], target_tariff=target, channel=channel)
        code = codes[0] if codes else "tariff_1"
        return campaign_dict(
            campaign_name="c01_empty_noop", filter_arpu_segment=ARPU_SEGMENTS[0],
            filter_data_segment=DATA_SEGMENTS[0], filter_call_segment=CALL_SEGMENTS[0],
            filter_current_tariff=code, target_tariff=code, channel=channel)
