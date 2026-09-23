"""Общий контракт: типы, конфигурация и вспомогательные функции, используемые всеми модулями агента.

Общий контракт: типы, конфигурация и вспомогательные функции агента.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ключи и константы
# ---------------------------------------------------------------------------

CellKey = tuple[str, str]  # (current_tariff, arpu_segment)
SubKey = tuple[str, str, str, str]  # (current_tariff, arpu_segment, data_segment, call_segment)
ArmKey = tuple[str, str, str]  # (current_tariff, arpu_segment, target_tariff)

CHANNELS_ORDER: tuple[str, ...] = ("push", "sms", "digital_ads", "call")
ARPU_SEGMENTS: tuple[str, ...] = ("LOW", "MID", "HIGH")
DATA_SEGMENTS: tuple[str, ...] = ("NON_USER", "LITE", "HEAVY")
CALL_SEGMENTS: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
CAMPAIGN_KEYS: tuple[str, ...] = (
    "campaign_name",
    "filter_arpu_segment",
    "filter_data_segment",
    "filter_call_segment",
    "filter_current_tariff",
    "target_tariff",
    "channel",
)
LLM_MODES: tuple[str, ...] = ("off", "advise", "decide")


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Неизменяемая конфигурация запуска (лимиты, параметры риска, настройки LLM)."""

    time_budget_s: float = 420.0
    llm_budget_s: float = 90.0
    llm_call_timeout_s: float = 30.0
    noise_sd: float = 0.804
    z_risk: float = 1.0
    p_min: float = 0.8
    p_min_push: float = 0.6
    tau: float = 0.2
    kappa_up: float = 0.9
    conv_prior_default: float = 0.3
    shrink_k: float = 20.0
    explore_money_frac: float = 0.25
    explore_reach_frac: float = 0.30
    max_pilots: int = 20
    pilot_sizes: tuple[int, ...] = (30, 60, 100, 150, 200)
    min_pilot: int = 10
    max_pilot: int = 200
    max_campaigns: int = 10
    max_per_campaign: int = 5000
    top_arms: int = 40
    top_targets_per_cell: int = 3
    max_pilots_per_arm: int = 3
    gh_nodes: int = 5
    seed: int = 42
    schema_version: str = "v1"
    llm_model: str = "gpt-4.1-mini"
    report_path: str = "agent_report.md"
    cache_file: str = "llm_cache.json"

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        """Собирает конфигурацию с учётом переопределений из env (OPENAI_MODEL) и явных kwargs.

        AGENT_LLM_MODE здесь не хранится; он определяется в resolve_llm_mode().
        """
        kw: dict[str, Any] = {}
        model = os.getenv("OPENAI_MODEL")
        if model and model.strip():
            kw["llm_model"] = model.strip()
        kw.update(overrides)
        return cls(**kw)

    def replace(self, **changes: Any) -> "Config":
        """Возвращает копию с изменёнными полями."""
        return dataclasses.replace(self, **changes)


def resolve_llm_mode() -> str:
    """Возвращает 'off'|'advise'|'decide' по AGENT_LLM_MODE.

    Не задано/некорректно -> 'off', независимо от учётных данных машины.
    Явный 'advise'/'decide' без ключа понижается до 'off'.
    """
    has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    raw = os.getenv("AGENT_LLM_MODE", "").strip().lower()
    if raw == "off":
        return "off"
    if raw in ("advise", "decide"):
        return raw if has_key else "off"
    return "off"


# ---------------------------------------------------------------------------
# Классы данных
# ---------------------------------------------------------------------------


@dataclass
class SubCell:
    """Единица таргетинга: cell x data_segment x call_segment."""

    key: SubKey
    n: int
    sum_p: float
    mean_p: float
    ids: np.ndarray  # отсортированные ID_NUMBER
    p: np.ndarray  # predicted_arpu, выровненные по ids


@dataclass
class Cell:
    """Единица эффекта: (current_tariff, arpu_segment)."""

    key: CellKey
    n: int
    sum_p: float
    mean_p: float
    subs: list[SubKey]


@dataclass
class Prior:
    """Априорное распределение базового lift ratio при множителе канала 1.0 (= change * share)."""

    mu: float
    sd: float
    share: float  # оценка конверсии
    n_hist: int
    source: str  # например, 'arm' | 'cell' | 'global' | 'price' | 'none'


@dataclass
class Observation:
    """Результат одного пилота, отнесённый к arm."""

    arm: ArmKey
    channel: str
    y: float  # observed_lift_ratio
    n: int  # n_customers, фактически получивших контакт
    cost: float
    sub: Optional[SubKey]
    pilot_index: int


@dataclass
class Posterior:
    """Апостериорное распределение lift ratio (доля predicted_arpu) для arm И канала."""

    mean: float
    sd: float


@dataclass
class Option:
    """Кандидатное действие для одной sub-cell."""

    sub: SubKey
    target: str
    channel: str
    n: int
    cost: float
    net_mean: float
    net_sd: float
    net_lcb: float
    p_pos: float


@dataclass
class Plan:
    """Результат аллокатора."""

    options: list[Option]
    lambda_money: float
    lambda_reach: float
    total_net_lcb: float
    total_cost: float
    total_contacts: int


@dataclass
class PilotSpec:
    """Пилот для запуска. filters = kwargs фильтров env.run_pilot (None = без фильтра)."""

    arm: ArmKey
    channel: str
    n: int
    sub: Optional[SubKey]
    filters: dict
    score: float
    reason: str

    def run_kwargs(self) -> dict:
        """Полный набор kwargs для env.run_pilot."""
        kw = {
            "target_tariff": self.arm[2],
            "channel": self.channel,
            "n_customers": int(self.n),
        }
        for k in ("filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff"):
            kw[k] = self.filters.get(k)
        return kw


@dataclass
class ExploreState:
    """Снимок ресурсов на исследование."""

    remaining_budget: float
    remaining_contacts: int
    pilots_left: int
    explore_money_spent: float
    explore_reach_spent: int
    pilots_per_arm: dict = field(default_factory=dict)  # dict[ArmKey, int]
    used_subs: dict = field(default_factory=dict)  # dict[SubKey, int] контакты, израсходованные пилотами
    deadline: float = math.inf  # на основе time.monotonic()

    def time_left(self) -> float:
        """Секунды до дедлайна (могут быть отрицательными)."""
        return self.deadline - time.monotonic()


@dataclass
class SimResult:
    """Результат ScoreSimulator.simulate."""

    gross: float
    cost: float
    contacts: int
    net: float
    per_campaign: list[dict]
    within_limits: bool


@dataclass
class ReviewOutcome:
    """Результат LLM-проверки рисков."""

    veto: list[str]
    rationale: str
    summary: str
    applied: bool
    source: str  # 'llm' | 'cache' | 'fallback' | 'off'


@dataclass
class RunLog:
    """Структурированный журнал запуска в памяти (без print)."""

    events: list[dict] = field(default_factory=list)
    pilots: list[dict] = field(default_factory=list)
    llm: list[dict] = field(default_factory=list)
    t0: float = field(default_factory=time.monotonic)

    def log(self, kind: str, **fields: Any) -> dict:
        """Добавляет событие {'kind': kind, 't': elapsed_s, **fields} и возвращает его."""
        ev = {"kind": kind, "t": round(time.monotonic() - self.t0, 3), **fields}
        self.events.append(ev)
        return ev


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    """CDF стандартного нормального распределения через math.erf (обрабатывает +-inf)."""
    if x != x:  # NaN
        return 0.5
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def campaign_dict(**kw: Any) -> dict:
    """Словарь кампании со всеми CAMPAIGN_KEYS (отсутствующие = None); неизвестные ключи отклоняются."""
    unknown = set(kw) - set(CAMPAIGN_KEYS)
    if unknown:
        raise KeyError(f"unknown campaign keys: {sorted(unknown)}")
    return {k: kw.get(k) for k in CAMPAIGN_KEYS}


def _canon(obj: Any) -> Any:
    """Преобразует в JSON-безопасную каноническую структуру (float округляются до 4 знаков)."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not math.isfinite(f):
            return None
        r = round(f, 4)
        return 0.0 if r == 0 else r
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else "|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _canon(v)
                for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_canon(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_canon(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _canon({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)})
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Детерминированный JSON: ключи отсортированы, компактно, float округлены до 4 знаков, tuple->list, ключи-tuple склеены через '|'."""
    return json.dumps(_canon(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(s: str) -> str:
    """Hex sha256 от строки UTF-8."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def arm_id(arm: ArmKey) -> str:
    """Стабильный строковый id для arm: 'from|seg|to'."""
    return "|".join(arm)


def parse_arm_id(s: str) -> ArmKey:
    """Обратная к arm_id; при некорректном вводе бросает ValueError."""
    parts = s.split("|")
    if len(parts) != 3:
        raise ValueError(f"bad arm id {s!r}")
    return (parts[0], parts[1], parts[2])
