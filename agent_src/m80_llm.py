from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip
"""LLM layer: two Pydantic AI agents (HypothesisAnalyst, RiskReviewer) with guardrails.

LLM-слой: аналитик гипотез и ревьюер риска; всё опционально, любой сбой -> нейтральный fallback.
"""

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Constants and plain validation rules (shared by output_validator and cache path)
# ---------------------------------------------------------------------------

_LLM_PLAUS_WEIGHT: dict[str, float] = {"low": 0.8, "mid": 1.0, "high": 1.2}
_LLM_VETO_P_MAX = 0.9  # only campaigns with P(net>0) below this may be vetoed
_LLM_REQUEST_LIMIT = 4
_LLM_TOOL_CALLS_LIMIT = 64  # local read-only tools; models batch one call per arm in a single request
_LLM_OUTPUT_RETRIES = 1
_LLM_REASON_MAX = 200

_LLM_ANALYST_INSTRUCTIONS = (
    "You are a telecom tariff-migration analyst. Each arm 'from|arpu_segment|to' is a hypothesis: "
    "offering tariff `to` to customers on tariff `from` in that ARPU segment raises their monthly revenue. "
    "Judge plausibility of each arm (low/mid/high) from tariff prices and package contents, the statistical "
    "prior (prior_mu = expected lift ratio, prior_sd = uncertainty, prior_share = conversion estimate, "
    "n_hist = historical evidence) and cell size (cell_n customers, cell_sum_p = sum of predicted ARPU). "
    "Upsell to a moderately pricier tariff with more package is plausible; big downgrades are not. "
    "The batch already contains every arm's prior and cell data; use tools only for a few specific checks. "
    "Return EXACTLY one item per arm_id of the batch, no extras, no duplicates, "
    "reason <= 200 chars."
)
_LLM_REVIEWER_INSTRUCTIONS = (
    "You are a risk reviewer of a marketing campaign plan. Each campaign has evidence: p_pos = P(net profit > 0), "
    "expected net and its sd. You may veto (remove) risky campaigns, but ONLY campaigns with p_pos < 0.9, "
    "and at least one campaign must remain. Use simulate_without to check the total effect of a veto. "
    "Veto only when evidence is weak AND the what-if shows no meaningful loss. Empty veto is fine. "
    "Give a short rationale and a 1-3 sentence analyst_summary."
)


def _llm_item_get(item: Any, key: str, default: Any = None) -> Any:
    """Read a field from a pydantic model or a dict."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _llm_validate_assessment(items: list, batch_ids: list[str]) -> list[str]:
    """Plain rules for HypothesisAssessment.items; returns error messages (empty = valid)."""
    errors: list[str] = []
    allowed = set(batch_ids)
    seen: set[str] = set()
    dupes: set[str] = set()
    unknown: set[str] = set()
    for it in items:
        aid = _llm_item_get(it, "arm_id")
        pl = _llm_item_get(it, "plausibility")
        if not isinstance(aid, str) or aid not in allowed:
            unknown.add(str(aid))
            continue
        if aid in seen:
            dupes.add(aid)
        seen.add(aid)
        if not isinstance(pl, str) or pl not in _LLM_PLAUS_WEIGHT:
            errors.append(f"bad plausibility {pl!r} for {aid}")
        reason = _llm_item_get(it, "reason", "")
        if not isinstance(reason, str) or len(reason) > _LLM_REASON_MAX:
            errors.append(f"reason too long or invalid for {aid}")
    missing = allowed - seen
    if unknown:
        errors.append(f"arm_ids not in batch: {sorted(unknown)}")
    if dupes:
        errors.append(f"duplicate arm_ids: {sorted(dupes)}")
    if missing:
        errors.append(f"missing arm_ids: {sorted(missing)}")
    return errors


def _llm_validate_review(veto: list, campaign_names: list[str], p_pos: dict[str, float]) -> list[str]:
    """Plain rules for PlanReview.veto; returns error messages (empty = valid)."""
    errors: list[str] = []
    names = set(campaign_names)
    if not isinstance(veto, list):
        return ["veto must be a list"]
    if len(set(map(str, veto))) != len(veto):
        errors.append("duplicate names in veto")
    for v in veto:
        if not isinstance(v, str) or v not in names:
            errors.append(f"unknown campaign {v!r}")
        elif float(p_pos.get(v, 1.0)) >= _LLM_VETO_P_MAX:
            errors.append(f"campaign {v!r} has p_pos >= {_LLM_VETO_P_MAX}; veto not allowed")
    if names and len(names - {v for v in veto if isinstance(v, str)}) < 1:
        errors.append("at least one campaign must remain")
    return errors


def _llm_evidence_name(ev: dict) -> Optional[str]:
    """Campaign name of an evidence row ('name' or 'campaign_name')."""
    n = ev.get("name", ev.get("campaign_name"))
    return None if n is None else str(n)


def _llm_evidence_ppos(ev: dict) -> float:
    """p_pos of an evidence row; missing/invalid -> 1.0 (not vetoable)."""
    try:
        v = float(ev.get("p_pos", 1.0))
    except (TypeError, ValueError):
        return 1.0
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else 1.0


# ---------------------------------------------------------------------------
# Deps (read-only snapshots for tools)
# ---------------------------------------------------------------------------


@dataclass
class AnalystDeps:
    """Read-only snapshot for HypothesisAnalyst tools."""

    batch_ids: list[str]
    items: dict[str, dict]  # arm_id -> item
    tariffs: dict[str, dict]
    cells: dict[str, dict] = field(default_factory=dict)  # "from|seg" -> aggregates


@dataclass
class ReviewDeps:
    """Read-only snapshot for RiskReviewer tools."""

    campaign_names: list[str]
    campaigns: dict[str, dict]
    evidence: dict[str, dict]
    p_pos: dict[str, float]
    what_if: Callable[[list[str]], dict]


def _llm_cells_from_items(items: list[dict]) -> dict[str, dict]:
    """Aggregate batch items per cell 'from|seg'."""
    cells: dict[str, dict] = {}
    for it in items:
        try:
            a = parse_arm_id(str(it.get("arm_id")))
        except ValueError:
            continue
        cid = f"{a[0]}|{a[1]}"
        c = cells.setdefault(cid, {"cell_id": cid, "targets": []})
        c["targets"].append(a[2])
        for k in ("n", "sum_p", "cell_n", "cell_sum_p", "mean_p"):
            if k in it and k not in c:
                c[k] = it[k]
    for c in cells.values():
        c["targets"] = sorted(set(c["targets"]))
    return cells


def _llm_tariff_price(t: dict) -> Optional[float]:
    """Monthly price from a tariff row, if present."""
    for k in ("price_tariff", "price", "monthly_fee"):
        if k in t:
            try:
                return float(t[k])
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Lazy Pydantic AI kit
# ---------------------------------------------------------------------------


def _llm_annotate(fn: Callable, ann: dict) -> Callable:
    """Attach real (non-string) annotations so pydantic-ai can resolve them lazily."""
    fn.__annotations__ = dict(ann)
    return fn


def _llm_build_kit() -> dict:
    """Import pydantic / pydantic_ai and build output models + agent factories. Raises ImportError."""
    from typing import Literal

    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")  # no console output from the library
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, ModelRetry, RunContext
    from pydantic_ai.exceptions import (  # noqa: F401  (documented failure types)
        AgentRunError,
        ModelHTTPError,
        UnexpectedModelBehavior,
        UsageLimitExceeded,
    )
    from pydantic_ai.usage import UsageLimits

    PlausLit = Literal["low", "mid", "high"]

    class ArmPlausibility(BaseModel):
        """Plausibility judgement for one arm."""

        arm_id: str
        plausibility: PlausLit  # type: ignore[valid-type]
        reason: str = Field(default="", max_length=_LLM_REASON_MAX)

    class HypothesisAssessment(BaseModel):
        """Analyst output for one batch."""

        items: list[ArmPlausibility]

    class PlanReview(BaseModel):
        """Risk reviewer output."""

        veto: list[str] = Field(default_factory=list)
        rationale: str = ""
        analyst_summary: str = ""


    ACtx = RunContext[AnalystDeps]
    RCtx = RunContext[ReviewDeps]

    def make_analyst(model: Any, settings: dict) -> Any:
        agent = Agent(model, output_type=HypothesisAssessment, deps_type=AnalystDeps, name="HypothesisAnalyst",
                      instructions=_LLM_ANALYST_INSTRUCTIONS, model_settings=settings,
                      retries={"output": _LLM_OUTPUT_RETRIES, "tools": 1})

        def get_cell(ctx, cell_id):
            """Cell aggregates (n customers, sum of predicted ARPU, candidate targets) for 'from|arpu_segment'."""
            return json.loads(canonical_json(ctx.deps.cells.get(cell_id, {"error": "unknown cell"})))

        def get_tariff(ctx, code):
            """Tariff dictionary row (price, package contents, description) for a tariff code."""
            return json.loads(canonical_json(ctx.deps.tariffs.get(code, {"error": "unknown tariff"})))

        def compare_tariffs(ctx, a, b):
            """Compare two tariffs: both rows and price difference b - a."""
            ta, tb = ctx.deps.tariffs.get(a), ctx.deps.tariffs.get(b)
            if ta is None or tb is None:
                return {"error": "unknown tariff"}
            pa, pb = _llm_tariff_price(ta), _llm_tariff_price(tb)
            out = {"a": ta, "b": tb, "price_diff": None if pa is None or pb is None else pb - pa}
            return json.loads(canonical_json(out))

        def get_history_arm(ctx, arm_id):
            """Statistical prior / history for an arm id 'from|seg|to' (mu, sd, share, n_hist, source)."""
            return json.loads(canonical_json(ctx.deps.items.get(arm_id, {"error": "unknown arm"})))

        agent.tool(_llm_annotate(get_cell, {"ctx": ACtx, "cell_id": str, "return": dict}))
        agent.tool(_llm_annotate(get_tariff, {"ctx": ACtx, "code": str, "return": dict}))
        agent.tool(_llm_annotate(compare_tariffs, {"ctx": ACtx, "a": str, "b": str, "return": dict}))
        agent.tool(_llm_annotate(get_history_arm, {"ctx": ACtx, "arm_id": str, "return": dict}))

        def check(ctx, out):
            errs = _llm_validate_assessment(list(out.items), ctx.deps.batch_ids)
            if errs:
                raise ModelRetry("; ".join(errs)[:800])
            return out

        agent.output_validator(_llm_annotate(check, {"ctx": ACtx, "out": HypothesisAssessment,
                                                     "return": HypothesisAssessment}))
        return agent

    def make_reviewer(model: Any, settings: dict) -> Any:
        agent = Agent(model, output_type=PlanReview, deps_type=ReviewDeps, name="RiskReviewer",
                      instructions=_LLM_REVIEWER_INSTRUCTIONS, model_settings=settings,
                      retries={"output": _LLM_OUTPUT_RETRIES, "tools": 1})

        def get_campaign_evidence(ctx, campaign_id):
            """Campaign definition and evidence (p_pos, net mean/sd, pilots) by campaign name."""
            if campaign_id not in ctx.deps.campaigns:
                return {"error": "unknown campaign", "known": ctx.deps.campaign_names}
            out = {"campaign": ctx.deps.campaigns[campaign_id], "evidence": ctx.deps.evidence.get(campaign_id, {})}
            return json.loads(canonical_json(out))

        def simulate_without(ctx, campaign_ids):
            """What-if: plan totals (net, gross, cost, contacts) if the listed campaigns are removed."""
            try:
                names = [c for c in campaign_ids if c in ctx.deps.campaigns]
                return json.loads(canonical_json({"without": names, **dict(ctx.deps.what_if(names))}))
            except Exception as exc:  # what-if must never crash the run
                return {"error": type(exc).__name__}

        agent.tool(_llm_annotate(get_campaign_evidence, {"ctx": RCtx, "campaign_id": str, "return": dict}))
        agent.tool(_llm_annotate(simulate_without, {"ctx": RCtx, "campaign_ids": list[str], "return": dict}))

        def check(ctx, out):
            errs = _llm_validate_review(list(out.veto), ctx.deps.campaign_names, ctx.deps.p_pos)
            if errs:
                raise ModelRetry("; ".join(errs)[:800])
            return out

        agent.output_validator(_llm_annotate(check, {"ctx": RCtx, "out": PlanReview, "return": PlanReview}))
        return agent

    return {
        "make_analyst": make_analyst,
        "make_reviewer": make_reviewer,
        "UsageLimits": UsageLimits,
        "HypothesisAssessment": HypothesisAssessment,
        "PlanReview": PlanReview,
    }


# ---------------------------------------------------------------------------
# LLMLayer
# ---------------------------------------------------------------------------


class LLMLayer:
    """Optional LLM advisors. Modes: off | advise (log only) | decide (apply). Never raises on LLM failure."""

    def __init__(self, cfg: Config, mode: str, cache_path: str, log: RunLog, model: Any = None):
        """`model` (optional extension): a pydantic-ai Model instance to use instead of OpenAI (tests)."""
        self.cfg = cfg
        self.mode = mode if mode in LLM_MODES else "off"
        self.cache_path = cache_path
        self.log = log
        self._model_override = model
        self._kit: Optional[dict] = None
        self._kit_failed = False
        self._model: Any = None
        self._cache: Optional[dict] = None
        self._spent = 0.0  # seconds of LLM wall time consumed

    # ---- infrastructure -------------------------------------------------

    def _remaining(self) -> float:
        return float(self.cfg.llm_budget_s) - self._spent

    def _get_kit(self) -> Optional[dict]:
        """Lazily build the pydantic-ai kit; ImportError (or any build failure) -> None (off behaviour)."""
        if self._kit is None and not self._kit_failed:
            try:
                self._kit = _llm_build_kit()
            except ImportError as exc:
                self._kit_failed = True
                self.log.log("llm_unavailable", error=f"ImportError: {exc}"[:200])
            except Exception as exc:
                self._kit_failed = True
                self.log.log("llm_unavailable", error=type(exc).__name__)
        return self._kit

    def _get_model(self) -> Any:
        """OpenAI Responses model (key from env only) or the injected test model."""
        if self._model_override is not None:
            return self._model_override
        if self._model is None:
            from pydantic_ai.models.openai import OpenAIResponsesModel
            from pydantic_ai.providers.openai import OpenAIProvider

            provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL"))
            self._model = OpenAIResponsesModel(self.cfg.llm_model, provider=provider)
        return self._model

    def _settings(self) -> dict:
        return {"temperature": 0.0, "timeout": float(self.cfg.llm_call_timeout_s)}

    def _load_cache(self) -> dict:
        if self._cache is None:
            self._cache = {}
            try:
                with open(self.cache_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    self._cache = data
            except (OSError, ValueError, TypeError):
                pass
        return self._cache

    def _save_cache(self) -> None:
        if self._cache is None:
            return
        try:
            tmp = f"{self.cache_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(self._cache, sort_keys=True, ensure_ascii=False, indent=1))
            os.replace(tmp, self.cache_path)
        except OSError:
            pass

    def cache_key(self, agent_name: str, payload: Any) -> str:
        """sha256(schema_version + agent + model + canonical_json(input))."""
        return sha256_text(f"{self.cfg.schema_version}{agent_name}{self.cfg.llm_model}{canonical_json(payload)}")

    def _record(self, sink: Optional[list] = None, **fields: Any) -> None:
        """Append an LLM record to the run log, or to `sink` (flushed later in deterministic order)."""
        rec = json.loads(canonical_json(fields))
        if sink is not None:
            sink.append(rec)
            return
        self.log.llm.append(rec)
        self.log.log("llm", **{k: v for k, v in rec.items() if k in ("agent", "batch", "source", "ok", "error",
                                                                      "latency_s", "mode")})

    async def _run_agent(self, agent: Any, prompt: str, deps: Any) -> tuple[Any, Optional[str], float]:
        """Run one agent call with per-call timeout; returns (output|None, error|None, latency)."""
        t0 = time.monotonic()
        timeout = min(float(self.cfg.llm_call_timeout_s), self._remaining())
        if timeout <= 0:
            return None, "llm_budget_exhausted", 0.0
        kit = self._kit or {}
        limits = kit["UsageLimits"](request_limit=_LLM_REQUEST_LIMIT, tool_calls_limit=_LLM_TOOL_CALLS_LIMIT)
        try:
            res = await asyncio.wait_for(agent.run(prompt, deps=deps, usage_limits=limits), timeout=timeout)
            return res.output, None, time.monotonic() - t0
        except TimeoutError:
            return None, "TimeoutError", time.monotonic() - t0
        except Exception as exc:  # AgentRunError / ModelHTTPError / UnexpectedModelBehavior / UsageLimitExceeded / ...
            return None, f"{type(exc).__name__}: {str(exc)[:160]}", time.monotonic() - t0

    # ---- HypothesisAnalyst ------------------------------------------------

    async def assess_hypotheses(self, batches: dict[str, list[dict]], tariffs: dict[str, dict]) -> dict[ArmKey, float]:
        """Plausibility weights {0.8,1.0,1.2} per arm; neutral 1.0 on off/advise/error/timeout."""
        neutral: dict[ArmKey, float] = {}
        for key in sorted(batches):
            for it in batches[key]:
                try:
                    neutral[parse_arm_id(str(it.get("arm_id")))] = 1.0
                except ValueError:
                    continue
        if self.mode == "off" or not neutral:
            self._record(agent="HypothesisAnalyst", source="off", mode=self.mode, ok=False, n_arms=len(neutral))
            return neutral

        t0 = time.monotonic()
        keys = sorted(k for k in batches if batches[k])
        sinks: dict[str, list] = {k: [] for k in keys}
        try:
            results = await asyncio.gather(*(self._assess_batch(k, batches[k], tariffs, sinks[k]) for k in keys))
        finally:  # account time and flush records in sorted batch order even on outer cancellation
            self._spent += time.monotonic() - t0
            for k in keys:
                for rec in sinks[k]:
                    self._record(**rec)

        weights = dict(neutral)
        for mapping in results:
            for aid, pl in mapping.items():
                try:
                    arm = parse_arm_id(aid)
                except ValueError:
                    continue
                if arm in weights:
                    weights[arm] = _LLM_PLAUS_WEIGHT.get(pl, 1.0)
        if self.mode != "decide":
            self.log.log("llm_advise", agent="HypothesisAnalyst",
                         would_apply={arm_id(a): w for a, w in sorted(weights.items()) if w != 1.0})
            return neutral
        return weights

    async def _assess_batch(self, batch_key: str, items: list[dict], tariffs: dict[str, dict],
                            sink: Optional[list] = None) -> dict[str, str]:
        """One batch -> {arm_id: plausibility}; {} on any failure. Records go to `sink` when given."""
        items_by_id: dict[str, dict] = {}
        for it in items:
            aid = str(it.get("arm_id"))
            items_by_id.setdefault(aid, dict(it))
        batch_ids = sorted(items_by_id)
        codes: set[str] = set()
        for aid in batch_ids:
            try:
                a = parse_arm_id(aid)
                codes.update((a[0], a[2]))
            except ValueError:
                pass
        rel_tariffs = {c: dict(tariffs[c]) for c in sorted(codes) if c in tariffs}
        payload = {"batch": batch_key, "items": [items_by_id[a] for a in batch_ids], "tariffs": rel_tariffs}
        prompt = (
            f"ARPU segment batch: {batch_key}. Assess every arm below.\n"
            f"arms={canonical_json(payload['items'])}\n"
            f"tariffs={canonical_json(rel_tariffs)}"
        )
        key = self.cache_key("HypothesisAnalyst", {"input": payload, "prompt": prompt})
        base = {"agent": "HypothesisAnalyst", "batch": batch_key, "mode": self.mode, "n_arms": len(batch_ids)}

        cached = self._load_cache().get(key)
        if isinstance(cached, dict) and isinstance(cached.get("output"), dict):
            out_items = cached["output"].get("items", [])
            if isinstance(out_items, list) and not _llm_validate_assessment(out_items, batch_ids):
                mapping = {str(i["arm_id"]): str(i["plausibility"]) for i in out_items}
                self._record(sink, **base, source="cache", ok=True, output=mapping)
                return mapping

        kit = self._get_kit()
        if kit is None:
            self._record(sink, **base, source="off", ok=False, error="pydantic_ai unavailable")
            return {}
        try:
            agent = kit["make_analyst"](self._get_model(), self._settings())
        except Exception as exc:
            self._record(sink, **base, source="fallback", ok=False, error=type(exc).__name__)
            return {}
        deps = AnalystDeps(batch_ids=batch_ids, items=items_by_id, tariffs=rel_tariffs,
                           cells=_llm_cells_from_items(list(items_by_id.values())))
        out, err, lat = await self._run_agent(agent, prompt, deps)
        if out is None:
            self._record(sink, **base, source="fallback", ok=False, error=err, latency_s=lat)
            return {}
        out_items = [i.model_dump() for i in out.items]
        errs = _llm_validate_assessment(out_items, batch_ids)
        if errs:
            self._record(sink, **base, source="fallback", ok=False, error="; ".join(errs)[:200], latency_s=lat)
            return {}
        mapping = {str(i["arm_id"]): str(i["plausibility"]) for i in out_items}
        self._load_cache()[key] = {"agent": "HypothesisAnalyst", "output": {"items": out_items}}
        self._save_cache()
        self._record(sink, **base, source="llm", ok=True, latency_s=lat, output=mapping,
                     reasons={str(i["arm_id"]): i.get("reason", "") for i in out_items})
        return mapping

    # ---- RiskReviewer ---------------------------------------------------

    async def review_plan(self, campaigns: list[dict], evidence: list[dict],
                          what_if: Callable[[list[str]], dict]) -> ReviewOutcome:
        """Optional veto of weak campaigns. applied=True only in decide mode with a valid non-empty veto."""
        names = [str(c.get("campaign_name")) for c in campaigns if c.get("campaign_name") is not None]
        if self.mode == "off" or not names:
            self._record(agent="RiskReviewer", source="off", mode=self.mode, ok=False)
            return ReviewOutcome(veto=[], rationale="", summary="", applied=False, source="off")

        ev_by: dict[str, dict] = {}
        for ev in evidence:
            n = _llm_evidence_name(ev)
            if n is not None and n not in ev_by:
                ev_by[n] = dict(ev)
        p_pos = {n: _llm_evidence_ppos(ev_by.get(n, {})) for n in names}
        camp_by = {str(c.get("campaign_name")): dict(c) for c in campaigns if c.get("campaign_name") is not None}
        payload = {"campaigns": [camp_by[n] for n in names], "evidence": [ev_by.get(n, {"name": n}) for n in names]}
        prompt = (
            "Review this campaign plan and decide vetoes.\n"
            f"campaigns={canonical_json(payload['campaigns'])}\n"
            f"evidence={canonical_json(payload['evidence'])}"
        )
        key = self.cache_key("RiskReviewer", {"input": payload, "prompt": prompt})
        base = {"agent": "RiskReviewer", "mode": self.mode, "n_campaigns": len(names)}
        fallback = ReviewOutcome(veto=[], rationale="", summary="", applied=False, source="fallback")

        out_d: Optional[dict] = None
        source = "llm"
        cached = self._load_cache().get(key)
        if isinstance(cached, dict) and isinstance(cached.get("output"), dict):
            c_out = cached["output"]
            if isinstance(c_out.get("veto"), list) and not _llm_validate_review(c_out["veto"], names, p_pos):
                out_d, source = c_out, "cache"

        if out_d is None:
            kit = self._get_kit()
            if kit is None:
                self._record(**base, source="off", ok=False, error="pydantic_ai unavailable")
                return ReviewOutcome(veto=[], rationale="", summary="", applied=False, source="off")
            try:
                agent = kit["make_reviewer"](self._get_model(), self._settings())
            except Exception as exc:
                self._record(**base, source="fallback", ok=False, error=type(exc).__name__)
                return fallback
            deps = ReviewDeps(campaign_names=names, campaigns=camp_by, evidence=ev_by, p_pos=p_pos, what_if=what_if)
            t0 = time.monotonic()
            try:
                out, err, lat = await self._run_agent(agent, prompt, deps)
            finally:
                self._spent += time.monotonic() - t0
            if out is None:
                self._record(**base, source="fallback", ok=False, error=err, latency_s=lat)
                return fallback
            out_d = out.model_dump()
            errs = _llm_validate_review(list(out_d.get("veto", [])), names, p_pos)
            if errs:
                self._record(**base, source="fallback", ok=False, error="; ".join(errs)[:200], latency_s=lat)
                return fallback
            self._load_cache()[key] = {"agent": "RiskReviewer", "output": out_d}
            self._save_cache()

        veto = [str(v) for v in out_d.get("veto", [])]
        rationale = str(out_d.get("rationale", ""))[:1000]
        summary = str(out_d.get("analyst_summary", ""))[:1000]
        applied = self.mode == "decide" and bool(veto)
        self._record(**base, source=source, ok=True, veto=veto, applied=applied, rationale=rationale[:300])
        return ReviewOutcome(veto=veto, rationale=rationale, summary=summary, applied=applied, source=source)
