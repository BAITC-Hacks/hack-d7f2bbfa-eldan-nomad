from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip

import dataclasses
import itertools
import math
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Guardrails: финальная страховка для списка кампаний, возвращаемого агентом.
# ---------------------------------------------------------------------------

_GR_SEGMENT_ALLOWED: dict[str, tuple[str, ...]] = {
    "filter_arpu_segment": ARPU_SEGMENTS,
    "filter_data_segment": DATA_SEGMENTS,
    "filter_call_segment": CALL_SEGMENTS,
}
_GR_HARD_MAX_CAMPAIGNS = 10  # scoring_core.MAX_CAMPAIGNS
_GR_MAX_SCAN = 1000  # максимум элементов, читаемых из входного iterable (защита от бесконечных генераторов)


def _gr_is_missing(v: Any) -> bool:
    """True для None, NaN-подобных float, pandas NA и пустых строк."""
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == "" or v.strip().lower() in ("nan", "none", "null")
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _gr_str(v: Any) -> Optional[str]:
    """Обрезанная строка или None для отсутствующих / неконвертируемых значений."""
    if _gr_is_missing(v):
        return None
    if isinstance(v, (list, tuple, set, frozenset, np.ndarray)):
        return None
    if isinstance(v, (float, np.floating)) and not math.isfinite(float(v)):
        return None
    try:
        s = str(v).strip()
    except Exception:  # noqa: BLE001 - произвольные мусорные объекты
        return None
    return s or None


def _gr_zero_ratio(current_tariff: str, arpu_segment: str, target: str, channel: str) -> float:
    """Функция ratio с нулевым лифтом: симуляция измеряет только cost/contacts."""
    return 0.0


class Guardrails:
    """Приводит, очищает и проверяет лимиты итогового списка кампаний. Никогда не бросает исключений."""

    def __init__(self, cfg: Config, dv: Any, sim: Any, log: Optional[RunLog] = None):
        self.cfg = cfg
        self.dv = dv
        self.sim = sim
        self.log = log
        self.issues: list[str] = []
        codes = [str(c) for c in (getattr(dv, "tariff_codes", None) or [])]
        self._tariffs: list[str] = codes
        self._tariff_ci: dict[str, str] = {c.lower(): c for c in codes}
        chans = [str(c) for c in (getattr(dv, "channels", None) or [])]
        self._channels: list[str] = chans
        self._channel_ci: dict[str, str] = {c.lower(): c for c in chans}

    # ------------------------------------------------------------------ публичное

    def validate(self, campaigns: Any, budget: float, contacts: int) -> list[dict]:
        """Возвращает валидный, дедуплицированный, укладывающийся в лимиты список из 1..max_campaigns кампаний."""
        self.issues = []
        try:
            out = self._validate(campaigns, budget, contacts)
        except Exception as exc:  # noqa: BLE001 - guardrail никогда не должен падать
            self._issue("validate_error", error=repr(exc)[:300])
            out = []
        if not out:
            try:
                fb = self.empty_campaign()
            except Exception as exc:  # noqa: BLE001
                self._issue("fallback_error", error=repr(exc)[:300])
                fb = None
            out = [fb] if fb is not None else []
            self._issue("fallback_empty_campaign", n=len(out))
        return out

    def empty_campaign(self) -> Optional[dict]:
        """Валидная push-кампания, фильтры которой (в идеале) не охватывают ни одного абонента; None, если невозможно."""
        if not self._tariffs or not self._channels:
            return None
        channel = "push" if "push" in self._channels else self._channels[0]
        subs = getattr(self.dv, "subs", None) or {}
        present = set(subs.keys()) if isinstance(subs, dict) else set()
        best: Optional[tuple] = None
        for frm in self._tariffs:
            for a in ARPU_SEGMENTS:
                for d in DATA_SEGMENTS:
                    for c in CALL_SEGMENTS:
                        key = (frm, a, d, c)
                        if key not in present:
                            best = key
                            break
                    if best:
                        break
                if best:
                    break
            if best:
                break
        if best is None:  # все комбинации заполнены: берём наименьшую подгруппу (по возможности)
            sizes = sorted(((int(getattr(s, "n", 0)), k) for k, s in subs.items()), key=lambda t: (t[0], t[1]))
            best = sizes[0][1] if sizes else (self._tariffs[0], ARPU_SEGMENTS[0], DATA_SEGMENTS[0], CALL_SEGMENTS[0])
        frm = best[0]
        targets = [t for t in self._tariffs if t != frm]
        target = targets[0] if targets else self._tariffs[0]
        return campaign_dict(
            campaign_name="g00_noop_push",
            filter_arpu_segment=best[1],
            filter_data_segment=best[2],
            filter_call_segment=best[3],
            filter_current_tariff=frm,
            target_tariff=target,
            channel=channel,
        )

    # ---------------------------------------------------------------- внутреннее

    def _issue(self, kind: str, **fields: Any) -> None:
        msg = kind + ("" if not fields else " " + ", ".join(f"{k}={v}" for k, v in sorted(fields.items())))
        self.issues.append(msg)
        if self.log is not None:
            try:
                self.log.log("guardrail", issue=kind, **fields)
            except Exception:  # noqa: BLE001
                pass

    def _max_campaigns(self) -> int:
        try:
            m = int(self.cfg.max_campaigns)
        except Exception:  # noqa: BLE001
            m = _GR_HARD_MAX_CAMPAIGNS
        return max(1, min(m, _GR_HARD_MAX_CAMPAIGNS))

    def _validate(self, campaigns: Any, budget: float, contacts: int) -> list[dict]:
        if campaigns is None:
            return []
        if isinstance(campaigns, dict):
            campaigns = [campaigns]
        elif isinstance(campaigns, pd.DataFrame):
            campaigns = campaigns.to_dict("records")
        if isinstance(campaigns, (str, bytes)):
            self._issue("not_iterable", type=type(campaigns).__name__)
            return []
        try:
            items = list(itertools.islice(iter(campaigns), _GR_MAX_SCAN + 1))
        except TypeError:
            self._issue("not_iterable", type=type(campaigns).__name__)
            return []
        if len(items) > _GR_MAX_SCAN:
            self._issue("scan_capped", cap=_GR_MAX_SCAN)
            items = items[:_GR_MAX_SCAN]

        cleaned: list[dict] = []
        seen_filters: set[tuple] = set()
        for i, raw in enumerate(items):
            c = self._coerce(i, raw)
            if c is None:
                continue
            fkey = self._filter_key(c)
            if fkey in seen_filters:
                self._issue("dropped_duplicate_filters", index=i)
                continue
            seen_filters.add(fkey)
            cleaned.append(c)

        cap = self._max_campaigns()
        if len(cleaned) > cap:
            self._issue("truncated", n=len(cleaned), cap=cap)
            cleaned = cleaned[:cap]

        self._dedupe_names(cleaned)
        lim = self._limits(budget, contacts)
        if lim is None:
            return cleaned
        out = self._enforce_limits(cleaned, lim[0], lim[1])
        self._log_overlaps(out)
        return out

    def _limits(self, budget: Any, contacts: Any) -> Optional[tuple[float, int]]:
        """Конечные неотрицательные (budget, contacts); None = непригодные лимиты (проверка пропускается, скоринг всё равно ограничит)."""
        try:
            b, k = float(budget), float(contacts)
        except (TypeError, ValueError):
            self._issue("bad_limits", budget=repr(budget)[:40], contacts=repr(contacts)[:40])
            return None
        if math.isnan(b) or math.isnan(k):
            self._issue("bad_limits", budget=b, contacts=k)
            return None
        b = max(b, 0.0)
        k = max(k, 0.0)
        return b, int(min(k, float(1 << 62)))

    def _log_overlaps(self, cs: list[dict]) -> None:
        """Логирует (не удаляет) кампании с пересекающейся аудиторией: скоринг это допускает, но платим дважды."""
        seg_fn = getattr(self.sim, "segment", None)
        if seg_fn is None or len(cs) < 2:
            return
        try:
            claimed: dict[Any, str] = {}
            for c in cs:
                ids = seg_fn(dict(c))["ID_NUMBER"].tolist()
                hits = sorted({claimed[i] for i in ids if i in claimed})
                if hits:
                    self._issue("overlap", name=c["campaign_name"], others=";".join(hits))
                for i in ids:
                    claimed.setdefault(i, c["campaign_name"])
        except Exception as exc:  # noqa: BLE001
            self._issue("overlap_check_error", error=repr(exc)[:200])

    def _coerce(self, i: int, raw: Any) -> Optional[dict]:
        """Очищает одну кампанию; None, если её нужно отбросить."""
        if dataclasses.is_dataclass(raw) and not isinstance(raw, type):
            raw = dataclasses.asdict(raw)
        elif isinstance(raw, pd.Series):
            raw = raw.to_dict()
        if not isinstance(raw, dict):
            self._issue("dropped_not_dict", index=i, type=type(raw).__name__)
            return None

        target = self._tariff_ci.get((_gr_str(raw.get("target_tariff")) or "").lower())
        if target is None:
            self._issue("dropped_unknown_target", index=i, value=repr(raw.get("target_tariff"))[:60])
            return None
        ch_raw = _gr_str(raw.get("channel"))
        channel = self._channel_ci.get((ch_raw or "").lower())
        if channel is None:
            self._issue("dropped_unknown_channel", index=i, value=repr(raw.get("channel"))[:60])
            return None

        segs: dict[str, Optional[str]] = {}
        for col, allowed in _GR_SEGMENT_ALLOWED.items():
            s = _gr_str(raw.get(col))
            if s is None:
                segs[col] = None
                continue
            s = s.upper()
            if s not in allowed:
                self._issue("dropped_unknown_segment", index=i, column=col, value=s[:30])
                return None
            segs[col] = s

        from_list = self._from_list(i, raw.get("filter_current_tariff"), target)
        if from_list is None:
            return None

        name = _gr_str(raw.get("campaign_name")) or f"g{i + 1:02d}_{target}_{channel}"
        return campaign_dict(
            campaign_name=name,
            filter_arpu_segment=segs["filter_arpu_segment"],
            filter_data_segment=segs["filter_data_segment"],
            filter_call_segment=segs["filter_call_segment"],
            filter_current_tariff=";".join(from_list),
            target_tariff=target,
            channel=channel,
        )

    def _from_list(self, i: int, value: Any, target: str) -> Optional[list[str]]:
        """Известные исходные тарифы (без target, в естественном порядке); None = отбросить кампанию."""
        if isinstance(value, (list, tuple, set, frozenset, np.ndarray)):
            parts = [_gr_str(v) for v in list(value)]
            given = True
        else:
            s = _gr_str(value)
            given = s is not None
            parts = [p.strip() for p in s.split(";")] if s is not None else []
        if not given:
            # Без фильтра попадут и абоненты, уже сидящие на целевом тарифе: сужаем явно.
            chosen = {t for t in self._tariffs if t != target}
        else:
            chosen = set()
            for p in parts:
                if not p:
                    continue
                code = self._tariff_ci.get(p.lower())
                if code is None:
                    self._issue("removed_unknown_from_tariff", index=i, value=p[:40])
                    continue
                if code == target:
                    self._issue("removed_target_from_list", index=i, target=target)
                    continue
                chosen.add(code)
        if not chosen:
            self._issue("dropped_empty_from_list", index=i)
            return None
        order = {t: k for k, t in enumerate(self._tariffs)}
        return sorted(chosen, key=lambda t: order.get(t, 1 << 30))

    @staticmethod
    def _filter_key(c: dict) -> tuple:
        frm = tuple(sorted((c.get("filter_current_tariff") or "").split(";")))
        return (c.get("filter_arpu_segment"), c.get("filter_data_segment"), c.get("filter_call_segment"), frm)

    def _dedupe_names(self, cs: list[dict]) -> None:
        used: set[str] = set()
        for c in cs:
            base = str(c["campaign_name"])
            name, k = base, 2
            while name in used:
                name = f"{base}_{k}"
                k += 1
            if name != base:
                self._issue("renamed_duplicate", old=base, new=name)
            c["campaign_name"] = name
            used.add(name)

    def _simulate(self, cs: list[dict], budget: float, contacts: int) -> Any:
        try:
            return self.sim.simulate([dict(c) for c in cs], _gr_zero_ratio, float(budget), int(contacts))
        except Exception as exc:  # noqa: BLE001
            self._issue("simulate_error", error=repr(exc)[:200])
            return None

    def _enforce_limits(self, cs: list[dict], budget: float, contacts: int) -> list[dict]:
        """Отбрасывает последнюю кампанию, упёршуюся в охват/бюджет, пока план не уложится (оставляет >= 1 кампании)."""
        if self.sim is None or not cs:
            return cs
        cs = list(cs)
        while True:
            res = self._simulate(cs, budget, contacts)
            if res is None or bool(getattr(res, "within_limits", True)):
                return cs
            if len(cs) <= 1:
                self._issue("over_limits_single_campaign")
                return cs
            per = list(getattr(res, "per_campaign", None) or [])
            drop = len(cs) - 1
            for j in range(min(len(per), len(cs)) - 1, -1, -1):
                pc = per[j] if isinstance(per[j], dict) else {}
                if pc.get("capped_reach") or pc.get("capped_money"):
                    drop = j
                    break
            self._issue("trimmed_over_limits", name=cs[drop]["campaign_name"])
            cs.pop(drop)


# ---------------------------------------------------------------------------
# Reporter: читаемый markdown-отчёт о запуске (по возможности).
# ---------------------------------------------------------------------------

_GR_Z95 = 1.96


def _gr_rep_fmt(v: Any, nd: int = 4) -> str:
    """Компактный, безопасный для markdown текст ячейки."""
    if v is None:
        return "—"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", " ")
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return "—"
        if abs(f) >= 1000:
            return f"{f:,.0f}".replace(",", " ")
        return f"{f:.{nd}f}".rstrip("0").rstrip(".") if f != 0 else "0"
    if isinstance(v, (list, tuple, set, frozenset)):
        return ", ".join(_gr_rep_fmt(x, nd) for x in (sorted(v, key=str) if isinstance(v, (set, frozenset)) else v)) or "—"
    if isinstance(v, dict):
        try:
            txt = canonical_json(v)
        except Exception:  # noqa: BLE001 - несериализуемые значения
            txt = str(v)
        return txt[:200].replace("|", "/").replace("\n", " ")
    s = str(v).replace("|", "/").replace("\n", " ").strip()
    return s[:200] if s else "—"


def _gr_rep_num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _gr_rep_first(d: dict, keys: tuple[str, ...]) -> Optional[float]:
    """Первое конечное числовое значение среди ``keys`` (пропускает отсутствующие / None / NaN)."""
    for k in keys:
        f = _gr_rep_num(d.get(k))
        if f is not None:
            return f
    return None


def _gr_rep_table(rows: list[dict], cols: list[str], headers: Optional[list[str]] = None) -> list[str]:
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_gr_rep_fmt(r.get(c)) for c in cols) + " |")
    return out


class Reporter:
    """Пишет agent_report.md: пилоты с CI, итоговый план, решения LLM, тайминги. Никогда не бросает исключений."""

    _PILOT_COLS = ("pilot_index", "index", "arm", "arm_id", "target_tariff", "channel", "sub", "filters",
                   "n_req", "n", "n_customers", "cost", "y", "observed_lift_ratio", "score", "reason", "error")
    _EXTRA_RENDERED = frozenset({"per_campaign", "expected_net", "net_sd", "p_net_pos", "expected_gross",
                                 "expected_cost", "expected_contacts", "remaining_budget", "remaining_contacts",
                                 "lambda_money", "lambda_reach", "timings", "notes", "llm_mode"})

    def __init__(self, cfg: Config, path: str):
        self.cfg = cfg
        self.path = path

    def write(self, log: RunLog, campaigns: list[dict], extra: dict) -> None:
        """Рендерит и записывает markdown; OSError (и ошибки рендеринга) подавляются."""
        try:
            text = self.render(log, campaigns, extra)
        except Exception as exc:  # noqa: BLE001
            text = f"# Agent report\n\nReport rendering failed: {_gr_rep_fmt(repr(exc))}\n"
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except (OSError, TypeError, ValueError):
            return

    # ------------------------------------------------------------------ рендеринг

    def render(self, log: Optional[RunLog], campaigns: Any, extra: Any) -> str:
        """Собирает markdown-текст."""
        extra = extra if isinstance(extra, dict) else {}
        lines: list[str] = ["# Agent report", ""]
        mode = extra.get("llm_mode")
        lines.append(f"- LLM mode: {_gr_rep_fmt(mode)}; model: {_gr_rep_fmt(getattr(self.cfg, 'llm_model', None))}")
        if log is not None:
            lines.append(f"- Elapsed: {_gr_rep_fmt(round(time.monotonic() - getattr(log, 't0', time.monotonic()), 2))} s;"
                         f" events: {len(getattr(log, 'events', []) or [])}")
        lines.append("")
        lines += self._pilots_section(getattr(log, "pilots", None) or [])
        lines += self._plan_section(campaigns, extra)
        lines += self._llm_section(getattr(log, "llm", None) or [])
        lines += self._timings_section(log, extra)
        lines += self._details_section(extra)
        lines += self._notes_section(extra)
        return "\n".join(lines).rstrip() + "\n"

    def _pilot_ci(self, p: dict) -> tuple[Optional[float], Optional[float]]:
        lo, hi = _gr_rep_num(p.get("ci_lo")), _gr_rep_num(p.get("ci_hi"))
        if lo is not None and hi is not None:
            return lo, hi
        y = _gr_rep_first(p, ("y", "observed_lift_ratio"))
        n = _gr_rep_first(p, ("n_customers", "n"))
        sd = _gr_rep_num(getattr(self.cfg, "noise_sd", 0.804))
        if y is None or n is None or n <= 0 or sd is None or sd < 0:
            return None, None
        half = _GR_Z95 * sd / math.sqrt(n)
        return y - half, y + half

    def _pilots_section(self, pilots: list) -> list[str]:
        out = ["## Pilots", ""]
        rows = [p for p in pilots if isinstance(p, dict)]
        if not rows:
            return out + ["No pilots were run.", ""]
        cols = [c for c in self._PILOT_COLS if any(c in r for r in rows)]
        table = []
        for k, r in enumerate(rows):
            lo, hi = self._pilot_ci(r)
            row = dict(r)
            row["#"] = k + 1
            row["ci95"] = f"[{_gr_rep_fmt(lo)}, {_gr_rep_fmt(hi)}]" if lo is not None else None
            table.append(row)
        out += _gr_rep_table(table, ["#"] + cols + ["ci95"])
        total_cost = sum(_gr_rep_num(r.get("cost")) or 0.0 for r in rows)
        total_n = sum(_gr_rep_first(r, ("n_customers", "n")) or 0.0 for r in rows)
        out += ["", f"Pilots: {len(rows)}, contacts: {_gr_rep_fmt(int(total_n))}, cost: {_gr_rep_fmt(total_cost)}. "
                "CI = observed lift ratio ± 1.96·noise_sd/√n.", ""]
        return out

    def _plan_section(self, campaigns: Any, extra: dict) -> list[str]:
        out = ["## Final plan", ""]
        cs = [c for c in (campaigns or []) if isinstance(c, dict)] if isinstance(campaigns, (list, tuple)) else []
        per = {}
        for pc in extra.get("per_campaign") or []:
            if isinstance(pc, dict) and pc.get("name") is not None:
                per[str(pc["name"])] = pc
        if not cs:
            out += ["No campaigns.", ""]
        else:
            rows = []
            for c in cs:
                row = dict(c)
                pc = per.get(str(c.get("campaign_name")), {})
                for k in ("n_contacted", "cost", "gross", "net", "p_pos"):
                    if k in pc:
                        row[k] = pc[k]
                if "net" not in row and _gr_rep_num(pc.get("gross")) is not None and _gr_rep_num(pc.get("cost")) is not None:
                    row["net"] = float(pc["gross"]) - float(pc["cost"])
                rows.append(row)
            cols = list(CAMPAIGN_KEYS) + [k for k in ("n_contacted", "cost", "gross", "net", "p_pos")
                                          if any(k in r for r in rows)]
            out += _gr_rep_table(rows, cols)
            out.append("")
        net = _gr_rep_num(extra.get("expected_net"))
        fs = extra.get("final_sim") if isinstance(extra.get("final_sim"), dict) else {}
        if net is None:
            net = _gr_rep_num(fs.get("net"))
        sd = _gr_rep_num(extra.get("net_sd"))
        p = _gr_rep_num(extra.get("p_net_pos"))
        if p is None and net is not None and sd is not None:
            p = norm_cdf(net / sd) if sd > 0 else (1.0 if net > 0 else 0.0)
        out.append(f"- Expected net: {_gr_rep_fmt(net)} (sd {_gr_rep_fmt(sd)}); P(net>0): {_gr_rep_fmt(p)}")
        for k in ("expected_gross", "expected_cost", "expected_contacts", "remaining_budget", "remaining_contacts",
                  "lambda_money", "lambda_reach"):
            if k in extra:
                out.append(f"- {k}: {_gr_rep_fmt(extra[k])}")
        out.append("")
        return out

    def _llm_section(self, llm: list) -> list[str]:
        out = ["## LLM decisions", ""]
        rows = [r for r in llm if isinstance(r, dict)]
        if not rows:
            return out + ["No LLM calls (mode off, unavailable, or skipped).", ""]
        pref = ("agent", "kind", "source", "applied", "latency_s", "veto", "summary", "rationale", "error")
        cols = [c for c in pref if any(c in r for r in rows)]
        rest = sorted({k for r in rows for k in r if k not in pref})[:4]
        out += _gr_rep_table(rows, cols + rest)
        out.append("")
        return out

    def _timings_section(self, log: Optional[RunLog], extra: dict) -> list[str]:
        out = ["## Timings", ""]
        tm = extra.get("timings")
        if isinstance(tm, dict) and tm:
            out += _gr_rep_table([{"stage": k, "seconds": v} for k, v in sorted(tm.items(), key=lambda kv: str(kv[0]))],
                              ["stage", "seconds"])
        else:
            evs = [e for e in (getattr(log, "events", None) or []) if isinstance(e, dict)]
            if not evs:
                return out + ["No timing data.", ""]
            out += _gr_rep_table([{"t": e.get("t"), "kind": e.get("kind"),
                                "fields": {k: v for k, v in e.items() if k not in ("t", "kind")}} for e in evs[-60:]],
                              ["t", "kind", "fields"])
        out.append("")
        return out

    def _details_section(self, extra: dict) -> list[str]:
        """Любые прочие дополнительные ключи (например final_sim, allocate, review, stages, explore), отсортированные по ключу."""
        keys = sorted((k for k in extra if str(k) not in self._EXTRA_RENDERED), key=str)
        if not keys:
            return []
        out = ["## Run details", ""]
        for k in keys:
            v = extra[k]
            if isinstance(v, dict) and v:
                out.append(f"- **{_gr_rep_fmt(str(k))}**")
                out += [f"  - {_gr_rep_fmt(str(kk))}: {_gr_rep_fmt(vv)}" for kk, vv in sorted(v.items(), key=lambda kv: str(kv[0]))]
            else:
                out.append(f"- **{_gr_rep_fmt(str(k))}**: {_gr_rep_fmt(v)}")
        out.append("")
        return out

    def _notes_section(self, extra: dict) -> list[str]:
        notes = extra.get("notes")
        if isinstance(notes, str):
            notes = [notes]
        if not notes:
            return []
        return ["## Notes", ""] + [f"- {_gr_rep_fmt(n)}" for n in notes] + [""]
