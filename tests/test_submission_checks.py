"""Проверки корректности сабмита на неизменённых скриптах организаторов."""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest

from tools.eval_checks import (
    build_checked_submission, campaign_errors, evaluate_agent, resource_errors,
)


class BrokenAgent:
    def act(self, env):
        return [{"target_tariff": "not-a-tariff", "channel": "push"}]


def test_invalid_campaigns_are_reported_by_extra_checks(repo_root):
    tariffs = pd.read_csv(repo_root / "data" / "dict_tariff.csv")
    assert campaign_errors(BrokenAgent().act(None), tariffs)


def test_raw_campaign_count_and_explicit_ids_are_checked():
    tariffs = pd.DataFrame({"tariff_plan_code": ["a", "b"]})
    camp = {"target_tariff": "b", "channel": "push", "filter_current_tariff": "a"}
    assert not campaign_errors([camp], tariffs)
    assert campaign_errors([camp] * 11, tariffs)
    assert campaign_errors([dict(camp, explicit_ids=[1])], tariffs)
    assert campaign_errors([dict(camp, filter_arpu_segment="TYPO")], tariffs)
    assert campaign_errors([dict(camp, filter_arpu_segment=["HIGH"])], tariffs)


def test_evaluation_does_not_hide_invalid_campaigns_or_missing_pilots():
    result = evaluate_agent(BrokenAgent(), seed=42, verbose=False)
    assert not result["requirements_pass"]
    assert result["n_final_campaigns"] == 0
    assert result["n_pilots"] == 0
    assert len(result["requirement_errors"]) >= 2


def test_submission_rejects_invalid_agent():
    with pytest.raises(ValueError, match="Invalid submission"):
        build_checked_submission(BrokenAgent())


def test_pilots_and_finals_share_resource_limits_and_duplicate_contacts_count():
    profile = pd.DataFrame({"ID_NUMBER": range(100)})
    pilot = {"target_tariff": "a", "channel": "sms", "explicit_ids": list(range(30))}
    final = {"target_tariff": "a", "channel": "sms"}
    # Each campaign fits separately. Repeat customers still consume resources.
    errors = resource_errors([pilot, final], profile, total_budget=500, max_total_contacts=120)
    assert len(errors) == 2
    assert "130 contacts" in errors[0]
    assert "520" in errors[1]


def test_preflight_reports_campaign_limit_before_scoring():
    profile = pd.DataFrame({"ID_NUMBER": range(5001)})
    errors = resource_errors([{"target_tariff": "a", "channel": "push"}], profile)
    assert len(errors) == 1
    assert "per-campaign limit 5000" in errors[0]


def test_submission_check_leaves_mismatched_file_intact(tmp_path, monkeypatch):
    from tools import preflight
    from make_submission import CAMPAIGN_COLUMNS

    df = pd.DataFrame([{"campaign_name": "valid"}]).reindex(columns=CAMPAIGN_COLUMNS)
    monkeypatch.setattr(preflight, "bundle_errors", lambda: [])
    monkeypatch.setattr(preflight, "build_checked_submission", lambda *args, **kwargs: df)
    output = tmp_path / "submission.csv"
    output.write_text("campaign_name\nstale\n")
    assert preflight.main(["--check", "--output", str(output)]) == 1
    assert output.read_text() == "campaign_name\nstale\n"


def test_submission_check_rejects_different_replays(tmp_path, monkeypatch):
    from tools import preflight

    output = tmp_path / "submission.csv"
    output.write_text("campaign_name\nfirst\n")
    results = iter([pd.DataFrame({"campaign_name": ["first"]}),
                    pd.DataFrame({"campaign_name": ["second"]})])
    monkeypatch.setattr(preflight, "build_checked_submission", lambda *args, **kwargs: next(results))
    with pytest.raises(ValueError, match="replays differ"):
        preflight.check_submission(object, output)
    assert output.read_text() == "campaign_name\nfirst\n"


def test_submission_check_accepts_matching_replays_without_rewriting(tmp_path, monkeypatch):
    from tools import preflight

    result = pd.DataFrame({"campaign_name": ["valid"]})
    output = tmp_path / "submission.csv"
    output.write_text(result.to_csv(index=False))
    previous_mtime = output.stat().st_mtime_ns
    monkeypatch.setattr(preflight, "build_checked_submission", lambda *args, **kwargs: result)
    pd.testing.assert_frame_equal(preflight.check_submission(object, output), result)
    assert output.stat().st_mtime_ns == previous_mtime


def test_stale_bundle_fails_before_running_agent(tmp_path, monkeypatch):
    from tools import preflight

    (tmp_path / "agent.py").write_text("# stale\n")
    monkeypatch.setattr(preflight, "module_files", lambda _: [])
    assert any("Stale agent.py" in error for error in preflight.bundle_errors(tmp_path))
    monkeypatch.setattr(preflight, "bundle_errors", lambda: ["Stale agent.py bundle"])

    def unexpected_replay(*args, **kwargs):
        pytest.fail("A stale bundle must fail before agent execution")

    monkeypatch.setattr(preflight, "check_submission", unexpected_replay)
    assert preflight.main(["--check"]) == 1


def test_bundled_submission_identical_across_processes_and_credentials(repo_root):
    """Собранный agent.py даёт одинаковый сабмит при разных PYTHONHASHSEED и наличии ключа."""
    code = (
        "from agent import Agent, Config; from make_submission import build_submission; "
        "import os; os.environ.pop('AGENT_LLM_MODE', None); "
        "print(build_submission(Agent(cfg=Config(report_path=''))).to_csv(index=False), end='')"
    )
    outputs = []
    for hash_seed, key in [("0", ""), ("73", "sk-test-not-a-real-key")]:
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, OPENAI_API_KEY=key)
        env.pop("AGENT_LLM_MODE", None)
        # Локальный .env не должен менять режим по умолчанию в этом тесте.
        script = "import agent; agent._orc_load_dotenv = lambda: None; " + code
        run = subprocess.run([sys.executable, "-c", script], cwd=repo_root, env=env,
                             text=True, capture_output=True, timeout=120, check=True)
        outputs.append(run.stdout)
    assert outputs[0] == outputs[1]
    assert "campaign_name" in outputs[0]


def test_generated_agent_matches_source(repo_root):
    from tools.build_agent import bundle, module_files

    assert (repo_root / "agent.py").read_text() == bundle(module_files())
