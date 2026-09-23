"""Check the source bundle and replay an existing submission without overwriting it.

Usage: python tools/preflight.py --check [--output submission.csv]
The organizer's generation command remains: python make_submission.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_submission import SUBMISSION_SEED
from tools.build_agent import bundle, module_files, check_forbidden
from tools.eval_checks import build_checked_submission


def bundle_errors(root=ROOT):
    """Detect a forgotten rebuild before spending time running the agent."""
    path = root / "agent.py"
    if not path.is_file():
        return [f"Missing agent bundle: {path}"]
    source = path.read_text(encoding="utf-8")
    errors = []
    if source != bundle(module_files(root / "agent_src")):
        errors.append("Stale agent.py bundle; run python tools/build_agent.py")
    errors.extend(f"Forbidden bundle token: {hit}" for hit in check_forbidden(source))
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        errors.append(f"agent.py cannot compile: {exc}")
    return errors


def check_submission(agent_factory, output, seed=SUBMISSION_SEED):
    """Two fresh seed-identical runs must agree with each other and the saved CSV."""
    output = Path(output)
    if not output.is_file():
        raise ValueError(f"Missing submission: {output}; run python make_submission.py")
    first = build_checked_submission(agent_factory(), seed=seed)
    second = build_checked_submission(agent_factory(), seed=seed)
    expected = first.to_csv(index=False)
    if second.to_csv(index=False) != expected:
        raise ValueError("Seed-identical replays differ; Agent output is not reproducible")
    if output.read_text(encoding="utf-8") != expected:
        raise ValueError(f"Submission mismatch: {output}; regenerate with python make_submission.py")
    return first


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check only (the default; never writes CSV)")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--seed", type=int, default=SUBMISSION_SEED)
    args = parser.parse_args(argv)
    try:
        errors = bundle_errors()
        if errors:
            raise ValueError("; ".join(errors))
        from agent import Agent, Config

        result = check_submission(lambda: Agent(cfg=Config(report_path="")), args.output, args.seed)
    except (ValueError, OSError) as exc:
        print(f"[preflight] FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"[preflight] PASS: current bundle; {len(result)} valid campaigns; "
          f"pilots/resources valid; two seed-{args.seed} replays match {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
