"""Tests for agent_src.m80_llm (LLMLayer) using pydantic-ai TestModel / FunctionModel; no network."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from agent_src.contract import Config, RunLog, parse_arm_id
from agent_src.m80_llm import (
    LLMLayer,
    _llm_validate_assessment,
    _llm_validate_review,
)

pydantic_ai = pytest.importorskip("pydantic_ai")
from pydantic_ai import models  # noqa: E402
from pydantic_ai.messages import ModelResponse, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.models.test import TestModel  # noqa: E402

models.ALLOW_MODEL_REQUESTS = False

def _item(arm: str, mu: float, sd: float, cell_n: int, cell_sum_p: float) -> dict:
    """Batch item in the orchestrator's production schema."""
    f, seg, t = arm.split("|")
    return {"arm_id": arm, "from_tariff": f, "arpu_segment": seg, "target_tariff": t, "prior_mu": mu,
            "prior_sd": sd, "prior_share": 0.3, "n_hist": 5, "prior_source": "arm", "cell_n": cell_n,
            "cell_sum_p": cell_sum_p}


BATCHES = {
    "HIGH": [
        _item("tariff_8|HIGH|tariff_10", 0.05, 0.02, 40, 20000.0),
        _item("tariff_8|HIGH|tariff_1", -0.2, 0.05, 40, 20000.0),
    ],
    "LOW": [
        _item("tariff_2|LOW|tariff_3", 0.01, 0.03, 90, 9000.0),
    ],
}
TARIFFS = {
    "tariff_1": {"price_tariff": 0.0, "description": "free"},
    "tariff_2": {"price_tariff": 300.0, "description": "cheap"},
    "tariff_3": {"price_tariff": 500.0, "description": "mid"},
    "tariff_8": {"price_tariff": 2000.0, "description": "rich"},
    "tariff_10": {"price_tariff": 2500.0, "description": "richer"},
}
PLAUS = {"tariff_8|HIGH|tariff_10": "high", "tariff_8|HIGH|tariff_1": "low", "tariff_2|LOW|tariff_3": "mid"}

CAMPAIGNS = [
    {"campaign_name": "c01", "target_tariff": "tariff_10", "channel": "sms"},
    {"campaign_name": "c02", "target_tariff": "tariff_3", "channel": "push"},
]
EVIDENCE = [{"name": "c01", "p_pos": 0.99, "net": 1000.0}, {"name": "c02", "p_pos": 0.7, "net": 50.0}]


def _what_if(names: list[str]) -> dict:
    cost = 1000.0 - 50.0 * len(names)
    return {"net": 2000.0 - 100.0 * len(names) - cost, "gross": 2000.0 - 100.0 * len(names), "cost": cost,
            "contacts": 100 - 10 * len(names)}


def _batch_ids(messages) -> list[str]:
    """Extract arm ids of this batch from the user prompt."""
    text = ""
    for m in messages:
        for p in getattr(m, "parts", []):
            if getattr(p, "part_kind", "") == "user-prompt":
                text = str(p.content)
    arms_line = [ln for ln in text.splitlines() if ln.startswith("arms=")][0]
    return [it["arm_id"] for it in json.loads(arms_line[len("arms="):])]


def _analyst_ok(messages, info: AgentInfo) -> ModelResponse:
    items = [{"arm_id": a, "plausibility": PLAUS[a], "reason": "r"} for a in _batch_ids(messages)]
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"items": items})])


def _layer(tmp_path: Path, mode: str = "decide", model=None, **cfg_kw) -> LLMLayer:
    cfg = Config().replace(**cfg_kw) if cfg_kw else Config()
    return LLMLayer(cfg, mode, str(tmp_path / "llm_cache.json"), RunLog(), model=model)


class _Boom:
    """Sentinel model that must never be used."""


def _raising_model() -> FunctionModel:
    def fn(messages, info):
        raise RuntimeError("model exploded")

    return FunctionModel(fn)


# ---------------------------------------------------------------------------
# Plain validators
# ---------------------------------------------------------------------------


def test_validate_assessment_rules() -> None:
    ids = ["a|L|b", "a|L|c"]
    ok = [{"arm_id": "a|L|b", "plausibility": "low", "reason": ""},
          {"arm_id": "a|L|c", "plausibility": "high", "reason": "x"}]
    assert _llm_validate_assessment(ok, ids) == []
    assert any("missing" in e for e in _llm_validate_assessment(ok[:1], ids))
    assert any("duplicate" in e for e in _llm_validate_assessment(ok + ok[:1], ids))
    assert any("not in batch" in e for e in _llm_validate_assessment(ok + [{"arm_id": "z|L|q",
                                                                           "plausibility": "low"}], ids))
    bad = [dict(ok[0], plausibility="huge"), ok[1]]
    assert any("plausibility" in e for e in _llm_validate_assessment(bad, ids))
    long_reason = [dict(ok[0], reason="x" * 201), ok[1]]
    assert _llm_validate_assessment(long_reason, ids)


def test_validate_review_rules() -> None:
    names = ["c01", "c02", "c03"]
    p = {"c01": 0.99, "c02": 0.7, "c03": 0.5}
    assert _llm_validate_review([], names, p) == []
    assert _llm_validate_review(["c02", "c03"], names, p) == []
    assert _llm_validate_review(["c01"], names, p)  # p_pos >= 0.9
    assert _llm_validate_review(["zzz"], names, p)
    assert _llm_validate_review(["c02", "c02"], names, p)
    assert _llm_validate_review(["c02"], ["c02"], {"c02": 0.1})  # nothing would remain
    assert _llm_validate_review(["c02"], names, {})  # missing p_pos -> treated as 1.0


# ---------------------------------------------------------------------------
# HypothesisAnalyst
# ---------------------------------------------------------------------------


def test_off_mode_neutral_without_model(tmp_path: Path) -> None:
    layer = _layer(tmp_path, mode="off", model=_Boom())
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert set(w) == {parse_arm_id(a) for a in PLAUS} and set(w.values()) == {1.0}
    rev = asyncio.run(layer.review_plan(CAMPAIGNS, EVIDENCE, _what_if))
    assert rev.source == "off" and rev.veto == [] and not rev.applied
    assert not (tmp_path / "llm_cache.json").exists()


def test_decide_applies_weights_and_writes_cache(tmp_path: Path) -> None:
    layer = _layer(tmp_path, model=FunctionModel(_analyst_ok))
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert w[("tariff_8", "HIGH", "tariff_10")] == 1.2
    assert w[("tariff_8", "HIGH", "tariff_1")] == 0.8
    assert w[("tariff_2", "LOW", "tariff_3")] == 1.0
    cache = json.loads((tmp_path / "llm_cache.json").read_text())
    assert len(cache) == 2  # one entry per batch
    assert {r["source"] for r in layer.log.llm} == {"llm"}


def test_cache_hit_skips_model(tmp_path: Path) -> None:
    asyncio.run(_layer(tmp_path, model=FunctionModel(_analyst_ok)).assess_hypotheses(BATCHES, TARIFFS))
    layer2 = _layer(tmp_path, model=_raising_model())
    w = asyncio.run(layer2.assess_hypotheses(BATCHES, TARIFFS))
    assert w[("tariff_8", "HIGH", "tariff_10")] == 1.2
    assert {r["source"] for r in layer2.log.llm} == {"cache"}


def test_invalid_cache_entry_is_revalidated(tmp_path: Path) -> None:
    asyncio.run(_layer(tmp_path, model=FunctionModel(_analyst_ok)).assess_hypotheses(BATCHES, TARIFFS))
    path = tmp_path / "llm_cache.json"
    cache = json.loads(path.read_text())
    for v in cache.values():
        v["output"]["items"] = v["output"]["items"][:0]  # drop all items -> invalid
    path.write_text(json.dumps(cache))
    layer = _layer(tmp_path, model=_raising_model())
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert set(w.values()) == {1.0}
    assert {r["source"] for r in layer.log.llm} == {"fallback"}


def test_corrupt_cache_file_ignored(tmp_path: Path) -> None:
    (tmp_path / "llm_cache.json").write_text("{not json")
    w = asyncio.run(_layer(tmp_path, model=FunctionModel(_analyst_ok)).assess_hypotheses(BATCHES, TARIFFS))
    assert w[("tariff_8", "HIGH", "tariff_1")] == 0.8


def test_output_validator_retry_then_success(tmp_path: Path) -> None:
    calls = {"n": 0}

    def fn(messages, info):
        calls["n"] += 1
        ids = _batch_ids(messages)
        retry_seen = any(getattr(p, "part_kind", "") == "retry-prompt" for m in messages for p in m.parts)
        chosen = ids if retry_seen else ids[:-1] + ["bogus|X|y"]
        items = [{"arm_id": a, "plausibility": PLAUS.get(a, "mid"), "reason": ""} for a in chosen]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"items": items})])

    layer = _layer(tmp_path, model=FunctionModel(fn))
    w = asyncio.run(layer.assess_hypotheses({"HIGH": BATCHES["HIGH"]}, TARIFFS))
    assert calls["n"] == 2
    assert w[("tariff_8", "HIGH", "tariff_10")] == 1.2


def test_output_validator_retry_exhausted_falls_back(tmp_path: Path) -> None:
    def fn(messages, info):
        items = [{"arm_id": "bogus|X|y", "plausibility": "high", "reason": ""}]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"items": items})])

    layer = _layer(tmp_path, model=FunctionModel(fn))
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert set(w.values()) == {1.0}
    assert all(r["source"] == "fallback" for r in layer.log.llm)
    assert not (tmp_path / "llm_cache.json").exists()


def test_model_exception_falls_back(tmp_path: Path) -> None:
    layer = _layer(tmp_path, model=_raising_model())
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert set(w.values()) == {1.0} and len(w) == 3
    assert any("RuntimeError" in (r.get("error") or "") for r in layer.log.llm)


def test_timeout_falls_back_quickly(tmp_path: Path) -> None:
    async def slow(messages, info):
        await asyncio.sleep(5)
        return _analyst_ok(messages, info)

    layer = _layer(tmp_path, model=FunctionModel(slow), llm_call_timeout_s=0.1)
    import time

    start = time.monotonic()
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert time.monotonic() - start < 2.0
    assert set(w.values()) == {1.0}
    assert all(r.get("error") == "TimeoutError" for r in layer.log.llm)


def test_budget_exhausted_no_call(tmp_path: Path) -> None:
    layer = _layer(tmp_path, model=_raising_model(), llm_budget_s=0.0)
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert set(w.values()) == {1.0}
    assert all(r.get("error") == "llm_budget_exhausted" for r in layer.log.llm)


def test_import_error_behaves_as_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == "pydantic_ai" or name.startswith("pydantic_ai."):
            monkeypatch.setitem(sys.modules, name, None)
    layer = _layer(tmp_path, model=_Boom())
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert set(w.values()) == {1.0}
    rev = asyncio.run(layer.review_plan(CAMPAIGNS, EVIDENCE, _what_if))
    assert rev.source == "off" and not rev.applied and rev.veto == []
    assert any(e["kind"] == "llm_unavailable" for e in layer.log.events)


def test_advise_logs_but_returns_neutral(tmp_path: Path) -> None:
    layer = _layer(tmp_path, mode="advise", model=FunctionModel(_analyst_ok))
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert set(w.values()) == {1.0}
    adv = [e for e in layer.log.events if e["kind"] == "llm_advise"]
    assert adv and adv[0]["would_apply"]["tariff_8|HIGH|tariff_10"] == 1.2


def test_testmodel_tools_do_not_crash(tmp_path: Path) -> None:
    """TestModel calls every tool with generated args, then returns generated (invalid) output -> fallback."""
    layer = _layer(tmp_path, model=TestModel())
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert set(w.values()) == {1.0}


def test_testmodel_custom_valid_output(tmp_path: Path) -> None:
    items = [{"arm_id": "tariff_2|LOW|tariff_3", "plausibility": "high", "reason": "ok"}]
    layer = _layer(tmp_path, model=TestModel(call_tools=[], custom_output_args={"items": items}))
    w = asyncio.run(layer.assess_hypotheses({"LOW": BATCHES["LOW"]}, TARIFFS))
    assert w == {("tariff_2", "LOW", "tariff_3"): 1.2}


def test_cache_save_oserror_swallowed(tmp_path: Path) -> None:
    cfg = Config()
    layer = LLMLayer(cfg, "decide", str(tmp_path / "no_such_dir" / "c.json"), RunLog(), model=FunctionModel(_analyst_ok))
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert w[("tariff_8", "HIGH", "tariff_10")] == 1.2


def test_cache_key_deterministic(tmp_path: Path) -> None:
    layer = _layer(tmp_path)
    k1 = layer.cache_key("A", {"b": 1.000001, "a": [1, 2]})
    k2 = layer.cache_key("A", {"a": [1, 2], "b": 1.0})
    assert k1 == k2 and len(k1) == 64
    assert layer.cache_key("B", {"a": [1, 2], "b": 1.0}) != k1
    other = LLMLayer(Config().replace(llm_model="x"), "decide", str(tmp_path / "c.json"), RunLog())
    assert other.cache_key("A", {"a": [1, 2], "b": 1.0}) != k1


# ---------------------------------------------------------------------------
# RiskReviewer
# ---------------------------------------------------------------------------


def _reviewer_fn(veto: list[str], record: dict):
    def fn(messages, info):
        n_req = sum(1 for m in messages if getattr(m, "kind", "") == "request")
        if n_req == 1:
            return ModelResponse(parts=[ToolCallPart("simulate_without", {"campaign_ids": veto}),
                                        ToolCallPart("get_campaign_evidence", {"campaign_id": "c02"})])
        for m in messages:
            for p in m.parts:
                if getattr(p, "part_kind", "") == "tool-return":
                    record[p.tool_name] = p.content
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                 {"veto": veto, "rationale": "weak", "analyst_summary": "s"})])

    return fn


def test_review_decide_applies_valid_veto_and_uses_tools(tmp_path: Path) -> None:
    rec: dict = {}
    layer = _layer(tmp_path, model=FunctionModel(_reviewer_fn(["c02"], rec)))
    out = asyncio.run(layer.review_plan(CAMPAIGNS, EVIDENCE, _what_if))
    assert out.veto == ["c02"] and out.applied and out.source == "llm"
    assert rec["simulate_without"]["net"] == 950.0
    assert rec["simulate_without"]["net"] == rec["simulate_without"]["gross"] - rec["simulate_without"]["cost"]
    assert rec["get_campaign_evidence"]["evidence"]["p_pos"] == 0.7
    # cache hit on second layer
    layer2 = _layer(tmp_path, model=_raising_model())
    out2 = asyncio.run(layer2.review_plan(CAMPAIGNS, EVIDENCE, _what_if))
    assert out2.source == "cache" and out2.veto == ["c02"] and out2.applied


def test_review_invalid_veto_falls_back(tmp_path: Path) -> None:
    layer = _layer(tmp_path, model=FunctionModel(_reviewer_fn(["c01"], {})))  # c01 p_pos 0.99
    out = asyncio.run(layer.review_plan(CAMPAIGNS, EVIDENCE, _what_if))
    assert out.source == "fallback" and out.veto == [] and not out.applied


def test_review_advise_not_applied(tmp_path: Path) -> None:
    layer = _layer(tmp_path, mode="advise", model=FunctionModel(_reviewer_fn(["c02"], {})))
    out = asyncio.run(layer.review_plan(CAMPAIGNS, EVIDENCE, _what_if))
    assert out.veto == ["c02"] and not out.applied


def test_review_what_if_error_is_contained(tmp_path: Path) -> None:
    def bad_what_if(names):
        raise ValueError("sim failed")

    rec: dict = {}
    layer = _layer(tmp_path, model=FunctionModel(_reviewer_fn([], rec)))
    out = asyncio.run(layer.review_plan(CAMPAIGNS, EVIDENCE, bad_what_if))
    assert rec["simulate_without"] == {"error": "ValueError"}
    assert out.source == "llm" and out.veto == [] and not out.applied


def test_review_exception_fallback(tmp_path: Path) -> None:
    layer = _layer(tmp_path, model=_raising_model())
    out = asyncio.run(layer.review_plan(CAMPAIGNS, EVIDENCE, _what_if))
    assert out.source == "fallback" and out.veto == []


def test_validators_reject_unhashable_values() -> None:
    assert _llm_validate_review([{"x": 1}], ["c01", "c02"], {"c01": 0.1})
    assert _llm_validate_assessment([{"arm_id": "a|L|b", "plausibility": ["high"]}], ["a|L|b"])


def test_malformed_cache_values_do_not_crash(tmp_path: Path) -> None:
    asyncio.run(_layer(tmp_path, model=FunctionModel(_analyst_ok)).assess_hypotheses(BATCHES, TARIFFS))
    asyncio.run(_layer(tmp_path, model=FunctionModel(_reviewer_fn(["c02"], {}))).review_plan(
        CAMPAIGNS, EVIDENCE, _what_if))
    path = tmp_path / "llm_cache.json"
    cache = json.loads(path.read_text())
    for v in cache.values():
        if "items" in v["output"]:
            for it in v["output"]["items"]:
                it["plausibility"] = ["high"]
        else:
            v["output"]["veto"] = [{"bad": 1}]
    path.write_text(json.dumps(cache))
    layer = _layer(tmp_path, model=_raising_model())
    w = asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert set(w.values()) == {1.0}
    rev = asyncio.run(layer.review_plan(CAMPAIGNS, EVIDENCE, _what_if))
    assert rev.source == "fallback" and rev.veto == [] and not rev.applied


def test_records_in_sorted_batch_order_despite_completion_order(tmp_path: Path) -> None:
    async def fn(messages, info):
        ids = _batch_ids(messages)
        await asyncio.sleep(0.2 if "HIGH" in ids[0] else 0.0)  # HIGH finishes last
        return _analyst_ok(messages, info)

    layer = _layer(tmp_path, model=FunctionModel(fn))
    asyncio.run(layer.assess_hypotheses(BATCHES, TARIFFS))
    assert [r["batch"] for r in layer.log.llm] == ["HIGH", "LOW"]
    assert [e["batch"] for e in layer.log.events if e["kind"] == "llm"] == ["HIGH", "LOW"]


def test_outer_cancellation_still_accounts_budget(tmp_path: Path) -> None:
    async def slow(messages, info):
        await asyncio.sleep(5)
        return _analyst_ok(messages, info)

    layer = _layer(tmp_path, model=FunctionModel(slow), llm_budget_s=10.0)

    async def main() -> None:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(layer.assess_hypotheses(BATCHES, TARIFFS), timeout=0.2)

    asyncio.run(main())
    assert layer._spent >= 0.15
    assert layer._remaining() < 10.0


def test_non_probability_p_pos_not_vetoable(tmp_path: Path) -> None:
    ev = [{"name": "c01", "p_pos": 0.99}, {"name": "c02", "p_pos": float("-inf")}]
    layer = _layer(tmp_path, model=FunctionModel(_reviewer_fn(["c02"], {})))
    out = asyncio.run(layer.review_plan(CAMPAIGNS, ev, _what_if))
    assert out.source == "fallback" and out.veto == [] and not out.applied
