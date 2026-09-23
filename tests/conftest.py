"""Shared pytest fixtures: repo on sys.path, mock env factory, small synthetic data."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SYNTH_CHANNELS = {
    "push": {"cost_per_contact": 0, "conversion_multiplier": 0.50},
    "sms": {"cost_per_contact": 4, "conversion_multiplier": 0.65},
    "digital_ads": {"cost_per_contact": 22, "conversion_multiplier": 0.85},
    "call": {"cost_per_contact": 160, "conversion_multiplier": 1.20},
}


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def mock_env_factory(repo_root: Path) -> Callable[..., object]:
    """Factory seed -> fresh mock env (organizer mock_environment.make_mock_env).

    Chdirs to the repo root for the session because make_mock_env uses relative paths.
    Returns only env (internals are dropped: agent code must never see them).
    """
    old = os.getcwd()
    os.chdir(repo_root)
    from mock_environment import make_mock_env

    def _make(seed: int | None = 0):
        env, _ = make_mock_env(seed=seed)
        return env

    yield _make
    os.chdir(old)


@pytest.fixture(scope="session")
def synth_channels() -> dict:
    """Channel table identical in shape to env.channels."""
    return {k: dict(v) for k, v in SYNTH_CHANNELS.items()}


@pytest.fixture()
def synth_tariffs() -> pd.DataFrame:
    """Four tariffs with dict_tariff columns."""
    return pd.DataFrame(
        {
            "Data_in_PKG": [0, 2048, 10240, 30720],
            "Min_another_operator_in_PKG": [0, 0, 100, 300],
            "Min_another_operator_and_city_in_PKG": [0, 0, 50, 100],
            "price_tariff": [0.0, 3140.0, 5990.0, 9990.0],
            "tariff_plan_code": ["tariff_1", "tariff_2", "tariff_3", "tariff_4"],
        }
    )


@pytest.fixture()
def synth_profile() -> pd.DataFrame:
    """Deterministic ~120-row profile over tariffs 1..3 with a few NaN rows.

    Columns: ID_NUMBER, current_tariff, arpu_segment, data_segment, call_segment, predicted_arpu.
    IDs are shuffled (not sorted) to exercise sort-by-ID semantics.
    """
    rng = np.random.default_rng(7)
    n = 120
    tariffs = np.array(["tariff_1", "tariff_2", "tariff_3"])
    arpu = np.array(["LOW", "MID", "HIGH"])
    data = np.array(["NON_USER", "LITE", "HEAVY"])
    call = np.array(["LOW", "MEDIUM", "HIGH"])
    df = pd.DataFrame(
        {
            "ID_NUMBER": rng.permutation(np.arange(1000, 1000 + n)),
            "current_tariff": tariffs[rng.integers(0, 3, n)],
            "arpu_segment": arpu[rng.integers(0, 3, n)],
            "data_segment": data[rng.integers(0, 3, n)],
            "call_segment": call[rng.integers(0, 3, n)],
            "predicted_arpu": np.round(rng.uniform(500, 12000, n), 2),
        }
    )
    df = df.astype({"current_tariff": object, "arpu_segment": object, "data_segment": object, "call_segment": object})
    df.loc[0, "current_tariff"] = np.nan
    df.loc[1, "arpu_segment"] = np.nan
    df.loc[2, "data_segment"] = np.nan
    return df
