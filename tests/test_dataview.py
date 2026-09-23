"""Tests for agent_src.m10_dataview."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent_src.contract import Cell, SubCell
from agent_src.m10_dataview import DataView, _dv_natural_key, load_history, load_tariff_descriptions

SEG = ["current_tariff", "arpu_segment", "data_segment", "call_segment"]


@pytest.fixture(scope="module")
def real_profile(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(repo_root / "customer_profile.csv")


@pytest.fixture(scope="module")
def real_tariffs(repo_root: Path) -> pd.DataFrame:
    return pd.read_csv(repo_root / "tariff_dictionary.csv").drop(columns=["description"])


@pytest.fixture(scope="module")
def real_dv(real_profile, real_tariffs, synth_channels) -> DataView:
    return DataView(real_profile, real_tariffs, synth_channels)


# ---------------------------------------------------------------- real data


def test_real_counts(real_dv: DataView, real_profile: pd.DataFrame) -> None:
    assert len(real_dv.cells) == 63
    assert real_dv.n_nan == 110
    n_sub_rows = sum(s.n for s in real_dv.subs.values())
    assert n_sub_rows == len(real_profile) - 110
    expected_subs = real_profile.dropna(subset=SEG).groupby(SEG).ngroups
    assert len(real_dv.subs) == expected_subs


def test_real_sums_consistent(real_dv: DataView, real_profile: pd.DataFrame) -> None:
    valid = real_profile.dropna(subset=SEG)
    total_sub = sum(s.sum_p for s in real_dv.subs.values())
    assert total_sub == pytest.approx(valid["predicted_arpu"].sum(), rel=1e-9)
    cell_valid = real_profile.dropna(subset=SEG[:2])
    assert sum(c.n for c in real_dv.cells.values()) == len(cell_valid)
    assert sum(c.sum_p for c in real_dv.cells.values()) == pytest.approx(cell_valid["predicted_arpu"].sum(), rel=1e-9)
    # Cell totals >= sum of its subs (rows with NaN data/call belong to no sub).
    for ck, c in real_dv.cells.items():
        assert isinstance(c, Cell)
        subs = real_dv.subs_of(ck)
        assert sum(s.n for s in subs) <= c.n
        assert sum(s.sum_p for s in subs) <= c.sum_p + 1e-6
        assert c.mean_p == pytest.approx(c.sum_p / c.n)


def test_real_subcell_arrays(real_dv: DataView, real_profile: pd.DataFrame) -> None:
    by_id = real_profile.set_index("ID_NUMBER")["predicted_arpu"]
    for key, s in real_dv.subs.items():
        assert isinstance(s, SubCell) and s.key == key
        assert s.ids.dtype == np.int64 and s.p.dtype == np.float64
        assert s.ids.size == s.n == s.p.size
        assert np.all(np.diff(s.ids) > 0)
        assert s.sum_p == pytest.approx(float(s.p.sum()))
        assert all(isinstance(x, str) for x in key)
    # spot check alignment on one sub
    s = next(iter(real_dv.subs.values()))
    np.testing.assert_allclose(s.p, by_id.loc[s.ids].to_numpy())


def test_real_keys_sorted_and_plain_str(real_dv: DataView) -> None:
    assert list(real_dv.subs) == sorted(real_dv.subs)
    assert list(real_dv.cells) == sorted(real_dv.cells)
    for c in real_dv.cells.values():
        assert c.subs == sorted(c.subs)
        assert all(real_dv.cell_of(k) == c.key for k in c.subs)
        assert type(c.key[0]) is str


def test_real_tariffs(real_dv: DataView, real_tariffs: pd.DataFrame) -> None:
    assert len(real_dv.tariff_codes) == len(real_tariffs)
    assert real_dv.tariff_codes == sorted(real_dv.tariff_codes, key=_dv_natural_key)
    assert real_dv.tariff_codes.index("tariff_2") < real_dv.tariff_codes.index("tariff_10")
    assert real_dv.tariff_price["tariff_1"] == 0.0
    info = real_dv.tariff_info("tariff_2")
    assert info["Data_in_PKG"] == 2048 and type(info["Data_in_PKG"]) is int
    assert info["price_tariff"] == pytest.approx(3140.0)
    assert set(info) == set(real_tariffs.columns)
    targets = real_dv.targets_for(("tariff_2", "LOW"))
    assert "tariff_2" not in targets and len(targets) == len(real_dv.tariff_codes) - 1


def test_mock_env(mock_env_factory) -> None:
    env = mock_env_factory(0)
    dv = DataView(env.customer_profile, env.tariffs, env.channels)
    assert len(dv.cells) == 63 and dv.n_nan == 110
    assert dv.channels == ["push", "sms", "digital_ads", "call"]
    for ch in dv.channels:
        assert dv.cost(ch) == float(env.channels[ch]["cost_per_contact"])
        assert dv.mult(ch) == float(env.channels[ch]["conversion_multiplier"])


# ---------------------------------------------------------------- synthetic data


def test_synth_nan_handling(synth_profile, synth_tariffs, synth_channels) -> None:
    dv = DataView(synth_profile, synth_tariffs, synth_channels)
    assert dv.n_nan == 3
    # row 2 has NaN data_segment only: counted in a cell, not in any sub
    assert sum(c.n for c in dv.cells.values()) == 118
    assert sum(s.n for s in dv.subs.values()) == 117
    row2 = synth_profile.loc[2]
    cell = dv.cells[(row2["current_tariff"], row2["arpu_segment"])]
    assert sum(s.n for s in dv.subs_of(cell.key)) == cell.n - 1
    # IDs shuffled in input, sorted within subs
    for s in dv.subs.values():
        assert np.all(np.diff(s.ids) > 0)


def test_synth_categorical_and_string_dtypes(synth_profile, synth_tariffs, synth_channels) -> None:
    base = DataView(synth_profile, synth_tariffs, synth_channels)
    cat = synth_profile.copy()
    for c in SEG:
        cat[c] = cat[c].astype("category")
    strd = synth_profile.copy()
    for c in SEG:
        strd[c] = strd[c].astype("string")
    for df in (cat, strd):
        dv = DataView(df, synth_tariffs, synth_channels)
        assert dv.n_nan == base.n_nan
        assert list(dv.cells) == list(base.cells)
        assert list(dv.subs) == list(base.subs)
        for k, s in dv.subs.items():
            np.testing.assert_array_equal(s.ids, base.subs[k].ids)
            assert all(type(x) is str for x in k)


def test_synth_unobserved_categories_dropped(synth_profile, synth_tariffs, synth_channels) -> None:
    df = synth_profile.copy()
    df["current_tariff"] = pd.Categorical(df["current_tariff"], categories=["tariff_1", "tariff_2", "tariff_3", "tariff_9"])
    dv = DataView(df, synth_tariffs, synth_channels)
    assert all(k[0] != "tariff_9" for k in dv.cells)
    assert all(c.n > 0 for c in dv.cells.values())


def test_channels_order_and_extra(synth_profile, synth_tariffs) -> None:
    ch = {
        "zeta": {"cost_per_contact": 1, "conversion_multiplier": 0.1},
        "call": {"cost_per_contact": 160, "conversion_multiplier": 1.2},
        "alpha": {"cost_per_contact": 2, "conversion_multiplier": 0.2},
        "push": {"cost_per_contact": 0, "conversion_multiplier": 0.5},
    }
    dv = DataView(synth_profile, synth_tariffs, ch)
    assert dv.channels == ["push", "call"]  # organizer scorer rejects unknown channels
    assert dv.cost("call") == 160.0 and dv.mult("push") == 0.5
    dv2 = DataView(synth_profile, synth_tariffs, ch, allow_extra_channels=True)
    assert dv2.channels == ["push", "call", "alpha", "zeta"]


def test_invalid_and_duplicate_ids(synth_channels) -> None:
    t = pd.DataFrame({"tariff_plan_code": ["tariff_1", "tariff_2"], "price_tariff": [1.0, 2.0]})
    p = pd.DataFrame(
        {
            "ID_NUMBER": [5, np.nan, 3, 5, 2.5],
            "current_tariff": ["tariff_1"] * 5,
            "arpu_segment": ["LOW"] * 5,
            "data_segment": ["LITE"] * 5,
            "call_segment": ["LOW"] * 5,
            "predicted_arpu": [1.0, 2.0, np.nan, 4.0, 8.0],
        }
    )
    dv = DataView(p, t, synth_channels)
    s = dv.subs[("tariff_1", "LOW", "LITE", "LOW")]
    assert s.ids.tolist() == [3, 5] and s.p.tolist() == [0.0, 1.0]
    assert dv.n_bad_id == 2
    assert dv.cells[("tariff_1", "LOW")].n == 5


def test_empty_profile(synth_profile, synth_tariffs, synth_channels) -> None:
    dv = DataView(synth_profile.iloc[:0], synth_tariffs, synth_channels)
    assert dv.cells == {} and dv.subs == {} and dv.n_nan == 0


def test_subs_match_organizer_filters(synth_profile, synth_tariffs, synth_channels) -> None:
    from scoring_core import apply_filters

    dv = DataView(synth_profile, synth_tariffs, synth_channels)
    for (tar, arpu, data, call), s in dv.subs.items():
        camp = {"filter_current_tariff": tar, "filter_arpu_segment": arpu,
                "filter_data_segment": data, "filter_call_segment": call}
        seg = apply_filters(synth_profile, camp).sort_values("ID_NUMBER")
        np.testing.assert_array_equal(s.ids, seg["ID_NUMBER"].to_numpy())
        assert s.sum_p == pytest.approx(float(seg["predicted_arpu"].fillna(0.0).sum()))


def test_natural_sort(synth_profile, synth_channels) -> None:
    t = pd.DataFrame({"tariff_plan_code": ["tariff_10", "tariff_2", "tariff_1"], "price_tariff": [3.0, 2.0, np.nan]})
    dv = DataView(synth_profile, t, synth_channels)
    assert dv.tariff_codes == ["tariff_1", "tariff_2", "tariff_10"]
    for codes in (["tariff_2", "tariff_02"], ["tariff_02", "tariff_2"]):
        t2 = pd.DataFrame({"tariff_plan_code": codes, "price_tariff": [1.0, 1.0]})
        assert DataView(synth_profile, t2, synth_channels).tariff_codes == ["tariff_02", "tariff_2"]
    assert "tariff_1" not in dv.tariff_price  # NaN price skipped
    assert dv.tariff_info("tariff_1")["price_tariff"] is None
    assert dv.targets_for(("tariff_3", "LOW")) == ["tariff_1", "tariff_2", "tariff_10"]


def test_subs_of_unknown_cell(synth_profile, synth_tariffs, synth_channels) -> None:
    dv = DataView(synth_profile, synth_tariffs, synth_channels)
    assert dv.subs_of(("nope", "LOW")) == []
    assert dv.cell_of(("a", "b", "c", "d")) == ("a", "b")


# ---------------------------------------------------------------- loaders


def test_load_history_real(repo_root: Path) -> None:
    df = load_history([str(repo_root)])
    assert df is not None and len(df) > 0
    assert {"tariff_plan_code_from", "tariff_plan_code_to"} <= set(df.columns)
    # searching a data dir directly also works
    df2 = load_history([str(repo_root / "data")])
    assert df2 is not None and len(df2) == len(df)


def test_load_history_missing(tmp_path: Path, repo_root: Path) -> None:
    assert load_history([str(tmp_path), "", "/nonexistent/dir"]) is None
    assert load_history([]) is None
    (tmp_path / "change_tariff.csv").write_bytes(b"\xff\xfe\x00garbage")
    assert load_history([str(tmp_path)]) is None  # never raises, corrupt file rejected
    (tmp_path / "change_tariff.csv").write_text("a,b\n1,2\n")
    assert load_history([str(tmp_path)]) is None  # wrong schema rejected
    df = load_history([str(tmp_path), str(repo_root)])  # falls back to next dir
    assert df is not None and len(df) > 0


def test_load_tariff_descriptions(repo_root: Path, tmp_path: Path) -> None:
    d = load_tariff_descriptions([str(tmp_path), str(repo_root)])
    assert "tariff_1" in d and isinstance(d["tariff_1"], str) and d["tariff_1"]
    assert load_tariff_descriptions([str(tmp_path)]) == {}
    (tmp_path / "tariff_dictionary.csv").write_text("a,b\n1,2\n")
    assert load_tariff_descriptions([str(tmp_path)]) == {}
