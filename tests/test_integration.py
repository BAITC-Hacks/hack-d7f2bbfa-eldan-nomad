"""End-to-end checks of the bundled agent.py on the organizer mock env (LLM off)."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
FILTER_COLS = ["filter_arpu_segment", "filter_data_segment", "filter_call_segment", "filter_current_tariff",
               "explicit_ids"]


def _load_agent_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bundled(tmp_path_factory):
    """Fresh build of agent.py, imported under a private module name."""
    import subprocess

    subprocess.run([sys.executable, str(REPO / "tools" / "build_agent.py")], check=True, cwd=str(REPO),
                   capture_output=True)
    return _load_agent_module(REPO / "agent.py", "_bundled_agent_it")


@pytest.fixture(autouse=True)
def _llm_off(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_MODE", "off")


def _mock(seed: int):
    old = os.getcwd()
    os.chdir(REPO)
    try:
        from mock_environment import make_mock_env

        return make_mock_env(seed=seed)
    finally:
        os.chdir(old)


def _score(env, internals, campaigns):
    from mock_environment import _mock_fallback, _mock_impact_model
    from scoring_core import score_campaigns

    frame = pd.DataFrame(internals.executed_pilot_campaigns() + list(campaigns))
    for col in FILTER_COLS:
        if col not in frame.columns:
            frame[col] = None
    impact = _mock_impact_model(pd.read_csv(REPO / "data" / "change_tariff.csv"))
    prof = env.customer_profile
    return score_campaigns(frame, prof, impact, env.tariffs, float(prof["predicted_arpu"].sum()), _mock_fallback)


@pytest.mark.parametrize("seed", [0, 42])
def test_bundled_agent_on_mock_env(bundled, seed):
    from scoring_core import MAX_TOTAL_CONTACTS, TOTAL_BUDGET, sanitize_campaigns

    env, internals = _mock(seed)
    out = bundled.Agent().act(env)
    assert isinstance(out, list) and 1 <= len(out) <= 10
    clean = sanitize_campaigns(out, env.tariffs)
    assert len(clean) == len(out)  # nothing dropped by the organizer sanitizer
    assert len({c["campaign_name"] for c in out}) == len(out)
    assert len(env.pilot_history) > 0
    assert env.remaining_budget >= 0 and env.remaining_contacts >= 0
    res = _score(env, internals, clean)
    assert res["total_contacts"] <= MAX_TOTAL_CONTACTS
    assert res["total_cost"] <= TOTAL_BUDGET
    finals = res["campaigns_detail"][len(env.pilot_history):]
    assert not any(d["capped_at_reach_budget"] or d["capped_at_money_budget"] or d["capped_at_campaign_limit"]
                   for d in finals)
    assert res["net_arpu_gain"] > 0


def test_build_submission_is_deterministic(bundled):
    old = os.getcwd()
    os.chdir(REPO)
    try:
        from make_submission import build_submission

        a = build_submission(bundled.Agent())
        b = build_submission(bundled.Agent())
    finally:
        os.chdir(old)
    assert len(a) >= 1
    pd.testing.assert_frame_equal(a, b)


def test_agent_runs_from_foreign_cwd(bundled, tmp_path, monkeypatch):
    env, _ = _mock(1)
    monkeypatch.chdir(tmp_path)
    out = bundled.Agent().act(env)
    assert 1 <= len(out) <= 10
    assert len(env.pilot_history) > 0


def test_agent_copy_without_data_dir(tmp_path, monkeypatch):
    """agent.py alone in a directory without data/: no history prior, still valid output."""
    from scoring_core import sanitize_campaigns

    shutil.copy(REPO / "agent.py", tmp_path / "agent.py")
    env, _ = _mock(2)
    monkeypatch.chdir(tmp_path)
    mod = _load_agent_module(tmp_path / "agent.py", "_isolated_agent_it")
    out = mod.Agent().act(env)
    assert 1 <= len(out) <= 10
    assert len(sanitize_campaigns(out, env.tariffs)) == len(out)
    assert not (tmp_path / "data").exists()
