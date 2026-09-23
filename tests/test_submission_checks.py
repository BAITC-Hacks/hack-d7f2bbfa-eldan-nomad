"""Submission correctness must fail visibly, even when the scorer can sanitize output."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from local_eval import campaign_errors, evaluate_agent
from make_submission import build_submission


class BrokenAgent:
    def act(self, env):
        return [{"target_tariff": "not-a-tariff", "channel": "push"}]


def test_evaluation_does_not_hide_invalid_campaigns_or_missing_pilots():
    result = evaluate_agent(BrokenAgent(), seed=42, verbose=False)
    assert not result["requirements_pass"]
    assert result["n_final_campaigns"] == 0
    assert result["n_pilots"] == 0
    assert len(result["requirement_errors"]) >= 2


def test_submission_rejects_invalid_agent():
    with pytest.raises(ValueError, match="Invalid submission"):
        build_submission(BrokenAgent())


def test_raw_campaign_count_and_explicit_ids_are_checked():
    tariffs = pd.DataFrame({"tariff_plan_code": ["a", "b"]})
    camp = {"target_tariff": "b", "channel": "push", "filter_current_tariff": "a"}
    assert not campaign_errors([camp], tariffs)
    assert campaign_errors([camp] * 11, tariffs)
    assert campaign_errors([dict(camp, explicit_ids=[1])], tariffs)


def test_submission_check_leaves_mismatched_file_intact(tmp_path, monkeypatch):
    import agent
    import make_submission as submission

    df = pd.DataFrame([{"campaign_name": "valid"}]).reindex(columns=submission.CAMPAIGN_COLUMNS)
    monkeypatch.setattr(agent, "Agent", lambda: object())
    monkeypatch.setattr(submission, "build_submission", lambda _: df)
    output = tmp_path / "submission.csv"
    output.write_text("campaign_name\nstale\n")
    with pytest.raises(SystemExit) as exc:
        submission.main(["--check", "--output", str(output)])
    assert exc.value.code == 1
    assert output.read_text() == "campaign_name\nstale\n"


def test_bundled_submission_identical_across_processes_and_credentials(repo_root):
    """Use the shipped single-file Agent, not imports from agent_src.

    API credentials and Python's randomized hashing must not alter default decisions.
    No requests can be made: the default LLM mode is explicitly tested to be off.
    """
    code = (
        "from agent import Agent, Config; from make_submission import build_submission; "
        "import os; os.environ.pop('AGENT_LLM_MODE', None); "
        "print(build_submission(Agent(cfg=Config(report_path=''))).to_csv(index=False), end='')"
    )
    outputs = []
    for hash_seed, key in [("0", ""), ("73", "sk-test-not-a-real-key")]:
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, OPENAI_API_KEY=key)
        env.pop("AGENT_LLM_MODE", None)
        # Prevent an optional local .env from changing this test's explicit default mode.
        script = "import agent; agent._orc_load_dotenv = lambda: None; " + code
        run = subprocess.run([sys.executable, "-c", script], cwd=repo_root, env=env,
                             text=True, capture_output=True, timeout=120, check=True)
        outputs.append(run.stdout)
    assert outputs[0] == outputs[1]
    assert "campaign_name" in outputs[0]


def test_generated_agent_matches_source(repo_root):
    from tools.build_agent import bundle, module_files

    assert (repo_root / "agent.py").read_text() == bundle(module_files())
