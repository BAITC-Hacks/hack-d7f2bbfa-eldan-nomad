"""Extra submission checks kept outside the organizer's scripts.

The organizer's local_eval.py / make_submission.py / scoring_core.py stay unmodified;
these helpers audit the raw agent output before sanitize_campaigns can hide problems.
"""
from __future__ import annotations

import time

import pandas as pd

from environment import MAX_PILOTS, MAX_PILOT_CUSTOMERS, MIN_PILOT_CUSTOMERS
from make_submission import CAMPAIGN_COLUMNS, SUBMISSION_SEED
from mock_environment import make_mock_env, _mock_fallback, _mock_impact_model
from scoring_core import (
    CHANNELS, MAX_CAMPAIGNS, MAX_CUSTOMERS_PER_CAMPAIGN,
    MAX_TOTAL_CONTACTS, TOTAL_BUDGET, apply_filters, score_campaigns,
    validate_strategy,
)


def campaign_errors(campaigns, tariffs):
    """Return human-readable problems with a raw Agent.act() result (empty list = OK)."""
    if not isinstance(campaigns, list) or not all(isinstance(c, dict) for c in campaigns):
        return ["Agent.act must return a list of campaign dictionaries"]
    errors = []
    if not 1 <= len(campaigns) <= MAX_CAMPAIGNS:
        errors.append(f"Expected 1..{MAX_CAMPAIGNS} final campaigns, got {len(campaigns)}")
    if any("explicit_ids" in c for c in campaigns):
        errors.append("Final campaigns must use segment filters, not explicit_ids")
    for index, campaign in enumerate(campaigns, 1):
        unknown = set(campaign) - set(CAMPAIGN_COLUMNS) - {"explicit_ids"}
        if unknown:
            errors.append(f"Campaign {index}: unsupported fields {sorted(unknown, key=str)}")
        for key in CAMPAIGN_COLUMNS:
            value = campaign.get(key)
            if not pd.api.types.is_scalar(value):
                errors.append(f"Campaign {index}: {key} must be a scalar value")
    if errors:
        return errors
    try:
        validate_strategy(pd.DataFrame(campaigns), tariffs)
    except (ValueError, TypeError, KeyError) as exc:
        errors.append(str(exc))
    return errors


def resource_errors(campaigns, profile, total_budget=TOTAL_BUDGET,
                    max_total_contacts=MAX_TOTAL_CONTACTS):
    """Audit all pilots + finals before the scorer silently clips their demand.

    Repeat contacts consume resources even if their ARPU effect is deduplicated.
    Input campaigns must already have valid target/channel/filter values.
    """
    errors = []
    contacts, cost = 0, 0
    for index, campaign in enumerate(campaigns, 1):
        size = len(apply_filters(profile, pd.Series(campaign)))
        if size > MAX_CUSTOMERS_PER_CAMPAIGN:
            errors.append(
                f"Campaign {index} would be truncated: {size} customers exceeds "
                f"the per-campaign limit {MAX_CUSTOMERS_PER_CAMPAIGN}"
            )
        size = min(size, MAX_CUSTOMERS_PER_CAMPAIGN)
        contacts += size
        cost += size * CHANNELS[campaign["channel"]]["cost_per_contact"]
    if contacts > max_total_contacts:
        errors.append(f"Pilots + final campaigns require {contacts} contacts; limit is {max_total_contacts}")
    if cost > total_budget:
        errors.append(f"Pilots + final campaigns require budget {cost:,.0f}; limit is {total_budget:,.0f}")
    return errors


def evaluate_agent(agent, seed=SUBMISSION_SEED, verbose=False, **env_kwargs):
    """Score once and expose requirement failures in the unsanitized agent output.

    This evaluator may use the public organizer scoring hooks. None of these
    checks, mock effects, or organizer internals are part of the shipped agent.
    """
    env, internals = make_mock_env(seed=seed, **env_kwargs)
    errors = []
    started = time.monotonic()
    try:
        campaigns = agent.act(env)
    except Exception as exc:
        errors.append(f"Agent.act failed: {type(exc).__name__}: {exc}")
        campaigns = []
    elapsed = time.monotonic() - started
    invalid = campaign_errors(campaigns, env.tariffs)
    errors.extend(invalid)
    finals = [] if invalid else campaigns
    pilots = internals.executed_pilot_campaigns()
    if not 1 <= len(pilots) <= MAX_PILOTS:
        errors.append(f"Expected 1..{MAX_PILOTS} pilots, got {len(pilots)}")
    for index, pilot in enumerate(pilots, 1):
        size = len(pilot.get("explicit_ids", []))
        if not MIN_PILOT_CUSTOMERS <= size <= MAX_PILOT_CUSTOMERS:
            errors.append(f"Pilot {index}: {size} contacts, expected {MIN_PILOT_CUSTOMERS}..{MAX_PILOT_CUSTOMERS}")
    if env.remaining_budget < 0 or env.remaining_contacts < 0 or env.pilots_left < 0:
        errors.append("Pilot resource limits exceeded")
    if elapsed > 600:
        errors.append(f"Agent runtime {elapsed:.1f}s exceeds 600s")
    all_campaigns = pilots + finals
    errors.extend(resource_errors(all_campaigns, env.customer_profile))
    strategy = pd.DataFrame(all_campaigns).reindex(columns=[*CAMPAIGN_COLUMNS, "explicit_ids"])
    impact = _mock_impact_model(pd.read_csv(f"{env_kwargs.get('data_dir', 'data')}/change_tariff.csv"))
    result = score_campaigns(
        strategy, env.customer_profile, impact, env.tariffs,
        float(env.customer_profile["predicted_arpu"].sum()), _mock_fallback,
        team_id="preflight",
    )
    result.update(
        requirements_pass=not errors, requirement_errors=errors,
        n_pilots=len(pilots), n_final_campaigns=len(finals),
        final_campaigns=finals, elapsed_seconds=elapsed,
    )
    if verbose:
        print(f"seed {seed}: {'PASS' if not errors else 'FAIL'} | "
              f"net={result['net_arpu_gain']:,.0f} | pilots={len(pilots)} | "
              f"finals={len(finals)} | contacts={result['total_contacts']:,} | "
              f"cost={result['total_cost']:,.0f} | {elapsed:.1f}s")
        for error in errors:
            print(f"[!] {error}")
    return result


def build_checked_submission(agent, seed=SUBMISSION_SEED, **env_kwargs):
    """Build the same seven CSV columns as the organizer, rejecting invalid runs."""
    result = evaluate_agent(agent, seed=seed, **env_kwargs)
    if not result["requirements_pass"]:
        raise ValueError("Invalid submission: " + "; ".join(result["requirement_errors"]))
    return pd.DataFrame(result["final_campaigns"]).reindex(columns=CAMPAIGN_COLUMNS)
