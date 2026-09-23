"""Tests for agent_src.contract and tools/build_agent.py."""
from __future__ import annotations

import dataclasses
import importlib.util
import math

import numpy as np
import pytest

from agent_src.contract import (
    CAMPAIGN_KEYS,
    CHANNELS_ORDER,
    Config,
    ExploreState,
    Option,
    PilotSpec,
    RunLog,
    arm_id,
    campaign_dict,
    canonical_json,
    norm_cdf,
    parse_arm_id,
    resolve_llm_mode,
    sha256_text,
)


def test_config_defaults_and_frozen():
    cfg = Config()
    assert cfg.noise_sd == pytest.approx(0.804)
    assert cfg.pilot_sizes == (30, 60, 100, 150, 200)
    assert cfg.max_campaigns == 10 and cfg.max_per_campaign == 5000
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1  # type: ignore[misc]
    assert cfg.replace(seed=1).seed == 1


def test_config_from_env(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert Config.from_env().llm_model == "gpt-4.1-mini"
    monkeypatch.setenv("OPENAI_MODEL", "gpt-x")
    assert Config.from_env().llm_model == "gpt-x"
    assert Config.from_env(z_risk=2.0).z_risk == 2.0


@pytest.mark.parametrize(
    "mode,key,expected",
    [
        (None, None, "off"),
        (None, "sk-test", "off"),
        ("off", "sk-test", "off"),
        ("advise", "sk-test", "advise"),
        ("DECIDE", "sk-test", "decide"),
        ("decide", None, "off"),
        ("garbage", "sk-test", "off"),
    ],
)
def test_resolve_llm_mode(monkeypatch, mode, key, expected):
    for var, val in (("AGENT_LLM_MODE", mode), ("OPENAI_API_KEY", key)):
        if val is None:
            monkeypatch.delenv(var, raising=False)
        else:
            monkeypatch.setenv(var, val)
    assert resolve_llm_mode() == expected


def test_norm_cdf():
    assert norm_cdf(0.0) == pytest.approx(0.5)
    assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert norm_cdf(-math.inf) == 0.0 and norm_cdf(math.inf) == 1.0
    assert norm_cdf(float("nan")) == 0.5


def test_campaign_dict():
    c = campaign_dict(campaign_name="a", target_tariff="tariff_2", channel="sms")
    assert tuple(c) == CAMPAIGN_KEYS
    assert c["filter_arpu_segment"] is None
    with pytest.raises(KeyError):
        campaign_dict(bogus=1)


def test_canonical_json_deterministic():
    a = {"b": 1.234567, "a": (1, 2), ("x", "y"): np.float64(0.00001), "arr": np.array([1.5, 2.0])}
    b = {"arr": [1.5, 2.0], "x|y": 0.0, "a": [1, 2], "b": 1.2346}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json({"v": float("nan")}) == '{"v":null}'
    opt = Option(("t", "LOW", "LITE", "LOW"), "t2", "sms", 5, 20.0, 1.0, 0.5, 0.5, 0.9)
    assert '"channel":"sms"' in canonical_json(opt)
    assert sha256_text("abc") == sha256_text("abc") and len(sha256_text("abc")) == 64


def test_arm_id_roundtrip():
    arm = ("tariff_1", "HIGH", "tariff_8")
    assert parse_arm_id(arm_id(arm)) == arm
    with pytest.raises(ValueError):
        parse_arm_id("a|b")


def test_runlog_and_state():
    log = RunLog()
    ev = log.log("pilot", n=30)
    assert ev["kind"] == "pilot" and ev["n"] == 30 and log.events == [ev]
    st = ExploreState(100.0, 10, 20, 0.0, 0)
    assert st.time_left() == math.inf and st.pilots_per_arm == {}


def test_pilot_spec_run_kwargs():
    spec = PilotSpec(("tariff_1", "MID", "tariff_2"), "sms", 60, None,
                     {"filter_arpu_segment": "MID", "filter_current_tariff": "tariff_1"}, 1.0, "r")
    kw = spec.run_kwargs()
    assert kw["target_tariff"] == "tariff_2" and kw["n_customers"] == 60
    assert kw["filter_data_segment"] is None and kw["filter_current_tariff"] == "tariff_1"


def test_channels_order():
    assert CHANNELS_ORDER == ("push", "sms", "digital_ads", "call")


def test_mock_env_fixture(mock_env_factory):
    env = mock_env_factory(0)
    assert set(CHANNELS_ORDER) <= set(env.channels)
    assert env.pilots_left == 20


def test_build_agent(tmp_path, repo_root):
    spec = importlib.util.spec_from_file_location("build_agent", repo_root / "tools" / "build_agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    src = tmp_path / "src"
    src.mkdir()
    (src / "contract.py").write_text(
        "from __future__ import annotations\nX = 1\n", encoding="utf-8")
    (src / "m10_a.py").write_text(
        "from __future__ import annotations  # bundle:strip\n"
        "from agent_src.contract import *  # noqa: F401,F403  bundle:strip\n"
        "from agent_src.contract import X  # bundle:strip\n"
        "Y = X + 1\n", encoding="utf-8")
    out = tmp_path / "agent.py"
    mod.build(out, src)
    text = out.read_text(encoding="utf-8")
    assert text.count("from __future__") == 1
    assert "bundle:strip" not in text
    ns: dict = {}
    exec(compile(text, str(out), "exec"), ns)
    assert ns["Y"] == 2

    (src / "m20_bad.py").write_text("import gc\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden"):
        mod.build(out, src)
    (src / "m20_bad.py").write_text("def f(:\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="compile"):
        mod.build(out, src)
