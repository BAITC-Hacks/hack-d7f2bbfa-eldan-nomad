from __future__ import annotations  # bundle:strip
from agent_src.contract import *  # noqa: F401,F403  bundle:strip
# m10 DataView: cells / sub-cells of the audience, tariff and channel tables.
# DataView: ячейки и подячейки аудитории, справочник тарифов и каналов.
import os
import re
from typing import Any

import numpy as np
import pandas as pd

_DV_SEG_COLS: tuple[str, ...] = ("current_tariff", "arpu_segment", "data_segment", "call_segment")
_DV_HISTORY_COLS: tuple[str, ...] = ("AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M", "tariff_plan_code_from", "tariff_plan_code_to")


def _dv_natural_key(code: str) -> tuple:
    """Natural sort key: 'tariff_2' < 'tariff_10'."""
    parts = re.split(r"(\d+)", str(code))
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in parts)


def _dv_py(v: Any) -> Any:
    """Convert numpy / pandas scalars to plain Python values (NaN -> None)."""
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    try:
        if v is pd.NA or v is pd.NaT:
            return None
    except Exception:  # pragma: no cover - defensive
        pass
    return v


class DataView:
    """Read-only aggregated view of env.customer_profile / env.tariffs / env.channels."""

    def __init__(
        self, profile: pd.DataFrame, tariffs: pd.DataFrame, channels: dict, allow_extra_channels: bool = False
    ):
        """allow_extra_channels=False keeps only CHANNELS_ORDER (the organizer scorer drops other channels)."""
        self._channels_raw: dict = {str(k): dict(v) for k, v in dict(channels).items()}
        self._build_tariffs(tariffs)
        self._build_channels(allow_extra_channels)
        self._build_cells(profile)

    # ------------------------------------------------------------------ tariffs / channels
    def _build_tariffs(self, tariffs: pd.DataFrame) -> None:
        t = tariffs.copy()
        t = t[t["tariff_plan_code"].notna()]
        # Keep raw codes (no strip): the organizer validates target_tariff against the raw values.
        t["tariff_plan_code"] = t["tariff_plan_code"].astype(str)
        t = t[t["tariff_plan_code"].str.strip() != ""]
        t = t.drop_duplicates("tariff_plan_code", keep="first")
        self.tariff_codes: list[str] = sorted(t["tariff_plan_code"].tolist(), key=lambda c: (_dv_natural_key(c), c))
        self._tariff_rows: dict[str, dict] = {}
        for rec in t.to_dict(orient="records"):
            code = str(rec["tariff_plan_code"])
            self._tariff_rows[code] = {str(k): _dv_py(v) for k, v in rec.items()}
        self.tariff_price: dict[str, float] = {}
        if "price_tariff" in t.columns:
            prices = pd.to_numeric(t["price_tariff"], errors="coerce").to_numpy(dtype=float)
            for code, pr in zip(t["tariff_plan_code"].tolist(), prices):
                if np.isfinite(pr):
                    self.tariff_price[str(code)] = float(pr)

    def _build_channels(self, allow_extra: bool) -> None:
        present = set(self._channels_raw)
        ordered = [c for c in CHANNELS_ORDER if c in present]
        extra = sorted(present - set(CHANNELS_ORDER)) if allow_extra else []
        self.channels: list[str] = ordered + extra

    # ------------------------------------------------------------------ cells
    def _build_cells(self, profile: pd.DataFrame) -> None:
        cols = list(_DV_SEG_COLS)
        df = pd.DataFrame(
            {
                "ID_NUMBER": pd.to_numeric(profile["ID_NUMBER"], errors="coerce"),
                "predicted_arpu": pd.to_numeric(profile["predicted_arpu"], errors="coerce").fillna(0.0),
            }
        )
        for c in cols:
            # Plain object dtype with NaN for missing: robust across str/category/object dtypes.
            s = profile[c]
            mask = s.isna()
            df[c] = s.astype(object).where(~mask, np.nan).map(lambda v: v if v != v else str(v))
        seg_na = df[cols].isna()
        self.n_nan: int = int(seg_na.any(axis=1).sum())

        # Cells: rows with valid current_tariff + arpu_segment.
        cell_df = df[~seg_na[cols[0]] & ~seg_na[cols[1]]]
        cagg = cell_df.groupby(cols[:2], sort=True, observed=True, dropna=True)["predicted_arpu"].agg(["size", "sum"])

        # Sub-cells: rows with all four segments valid and a finite integral ID, sorted by keys then ID.
        # Duplicate IDs keep the first row (organizer gross lift is deduplicated per ID_NUMBER).
        idv = df["ID_NUMBER"].to_numpy(dtype=float)
        id_ok = np.isfinite(idv) & (np.floor(idv) == idv) & (np.abs(idv) < 2.0**62)
        self.n_bad_id: int = int((~id_ok).sum())
        sub_df = df[~seg_na.any(axis=1) & id_ok]
        sub_df = sub_df[~sub_df["ID_NUMBER"].duplicated(keep="first")]
        sub_df = sub_df.sort_values(cols + ["ID_NUMBER"], kind="mergesort")
        ids_all = sub_df["ID_NUMBER"].to_numpy(dtype=np.int64)
        p_all = sub_df["predicted_arpu"].to_numpy(dtype=float)
        groups = sub_df.groupby(cols, sort=True, observed=True, dropna=True).indices

        self.subs: dict[SubKey, SubCell] = {}
        for key in sorted(groups):
            idx = np.asarray(groups[key])
            skey: SubKey = tuple(str(x) for x in key)  # type: ignore[assignment]
            ids = ids_all[idx]
            p = p_all[idx]
            order = np.argsort(ids, kind="mergesort")  # already sorted; cheap safety
            ids, p = ids[order], p[order]
            n = int(ids.size)
            s = float(p.sum())
            self.subs[skey] = SubCell(key=skey, n=n, sum_p=s, mean_p=s / n if n else 0.0, ids=ids, p=p)

        subs_by_cell: dict[CellKey, list[SubKey]] = {}
        for skey in self.subs:
            subs_by_cell.setdefault((skey[0], skey[1]), []).append(skey)

        self.cells: dict[CellKey, Cell] = {}
        for key, row in cagg.sort_index().iterrows():
            ckey: CellKey = (str(key[0]), str(key[1]))
            n = int(row["size"])
            s = float(row["sum"])
            self.cells[ckey] = Cell(
                key=ckey, n=n, sum_p=s, mean_p=s / n if n else 0.0, subs=sorted(subs_by_cell.get(ckey, []))
            )

    # ------------------------------------------------------------------ accessors
    def tariff_info(self, code: str) -> dict:
        """All dict_tariff columns of the tariff row (KeyError if unknown)."""
        return dict(self._tariff_rows[code])

    def cost(self, ch: str) -> float:
        """Cost per contact of a channel."""
        return float(self._channels_raw[ch]["cost_per_contact"])

    def mult(self, ch: str) -> float:
        """Conversion multiplier of a channel."""
        return float(self._channels_raw[ch]["conversion_multiplier"])

    def cell_of(self, sub: SubKey) -> CellKey:
        """Cell key of a sub-cell key."""
        return (sub[0], sub[1])

    def subs_of(self, cell: CellKey) -> list[SubCell]:
        """Sub-cells of a cell, sorted by key."""
        c = self.cells.get(cell)
        if c is None:
            return []
        return [self.subs[k] for k in c.subs]

    def targets_for(self, cell: CellKey) -> list[str]:
        """Candidate target tariffs for a cell (all tariffs except the current one)."""
        return [t for t in self.tariff_codes if t != cell[0]]


def _dv_candidates(search_dirs: list[str], names: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for d in search_dirs or []:
        if not d:
            continue
        for name in names:
            out.append(os.path.join(d, name))
    return out


def load_history(search_dirs: list[str]) -> pd.DataFrame | None:
    """Load change_tariff history from <dir>/data/ or <dir>/; None if unavailable. Never raises."""
    try:
        for path in _dv_candidates(search_dirs, (os.path.join("data", "change_tariff.csv"), "change_tariff.csv")):
            try:
                if os.path.isfile(path):
                    df = pd.read_csv(path)
                    if len(df) > 0 and all(c in df.columns for c in _DV_HISTORY_COLS):
                        return df
            except Exception:
                continue
    except Exception:
        return None
    return None


def load_tariff_descriptions(search_dirs: list[str]) -> dict[str, str]:
    """Map tariff_plan_code -> description from tariff_dictionary.csv; {} on failure."""
    try:
        for path in _dv_candidates(search_dirs, ("tariff_dictionary.csv", os.path.join("data", "tariff_dictionary.csv"))):
            try:
                if not os.path.isfile(path):
                    continue
                df = pd.read_csv(path)
                if "tariff_plan_code" not in df.columns or "description" not in df.columns:
                    continue
                out: dict[str, str] = {}
                for code, desc in zip(df["tariff_plan_code"].tolist(), df["description"].tolist()):
                    if code is None or desc is None or code != code or desc != desc:
                        continue
                    out[str(code)] = str(desc)
                return out
            except Exception:
                continue
    except Exception:
        return {}
    return {}
