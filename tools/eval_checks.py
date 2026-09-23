"""Extra submission checks kept outside the organizer's scripts.

The organizer's local_eval.py / make_submission.py / scoring_core.py stay unmodified;
these helpers audit the raw agent output before sanitize_campaigns can hide problems.
"""
from __future__ import annotations

import pandas as pd

from scoring_core import MAX_CAMPAIGNS, validate_strategy


def campaign_errors(campaigns, tariffs):
    """Return human-readable problems with a raw Agent.act() result (empty list = OK)."""
    if not isinstance(campaigns, list) or not all(isinstance(c, dict) for c in campaigns):
        return ["Agent.act must return a list of campaign dictionaries"]
    errors = []
    if not 1 <= len(campaigns) <= MAX_CAMPAIGNS:
        errors.append(f"Expected 1..{MAX_CAMPAIGNS} final campaigns, got {len(campaigns)}")
    if any("explicit_ids" in c for c in campaigns):
        errors.append("Final campaigns must use segment filters, not explicit_ids")
    try:
        validate_strategy(pd.DataFrame(campaigns), tariffs)
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    return errors
