from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProtectedLevelParams:
    """
    Protected Low:
        dernier Swing Low confirmé disponible lors d'un BOS bullish.

    Protected High:
        dernier Swing High confirmé disponible lors d'un BOS bearish.

    break_mode:
        "close" ou "wick".
    """

    break_mode: str = "close"

    def __post_init__(self) -> None:
        if self.break_mode not in {"close", "wick"}:
            raise ValueError("break_mode doit être 'close' ou 'wick'")


def detect_protected_levels(
    df: pd.DataFrame,
    params: ProtectedLevelParams = ProtectedLevelParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
        "swing_high",
        "swing_low",
        "swing_high_price",
        "swing_low_price",
        "swing_high_origin_idx",
        "swing_low_origin_idx",
        "bos_bull",
        "bos_bear",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"detect_protected_levels(): colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["protected_low_active"] = False
    out["protected_high_active"] = False

    out["protected_low_price"] = np.nan
    out["protected_high_price"] = np.nan

    out["protected_low_broken"] = False
    out["protected_high_broken"] = False

    for side in ("low", "high"):
        out[f"protected_{side}_source_origin_idx"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

        out[f"protected_{side}_created_idx"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

        out[f"protected_{side}_age_bars"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

    latest_swing_low = None
    latest_swing_high = None

    protected_low = None
    protected_high = None

    highs = out["high"].to_numpy(dtype=np.float64)
    lows = out["low"].to_numpy(dtype=np.float64)
    closes = out["close"].to_numpy(dtype=np.float64)

    for i in range(n):

        row = out.iloc[i]
        label = out.index[i]

        # -------------------------------------------------------------
        # Swings disponibles causalement.
        # -------------------------------------------------------------

        if bool(row["swing_low"]):
            latest_swing_low = {
                "price": float(row["swing_low_price"]),
                "origin_idx": int(row["swing_low_origin_idx"]),
            }

        if bool(row["swing_high"]):
            latest_swing_high = {
                "price": float(row["swing_high_price"]),
                "origin_idx": int(row["swing_high_origin_idx"]),
            }

        # -------------------------------------------------------------
        # Création des protected levels.
        # -------------------------------------------------------------

        if bool(row["bos_bull"]) and latest_swing_low is not None:
            protected_low = {
                "price": latest_swing_low["price"],
                "origin_idx": latest_swing_low["origin_idx"],
                "created_idx": i,
                "active": True,
            }

        if bool(row["bos_bear"]) and latest_swing_high is not None:
            protected_high = {
                "price": latest_swing_high["price"],
                "origin_idx": latest_swing_high["origin_idx"],
                "created_idx": i,
                "active": True,
            }

        # -------------------------------------------------------------
        # Test de cassure.
        # -------------------------------------------------------------

        downside_price = (
            closes[i]
            if params.break_mode == "close"
            else lows[i]
        )

        upside_price = (
            closes[i]
            if params.break_mode == "close"
            else highs[i]
        )

        if protected_low is not None and protected_low["active"]:

            if downside_price < protected_low["price"]:
                out.at[label, "protected_low_broken"] = True
                protected_low["active"] = False

        if protected_high is not None and protected_high["active"]:

            if upside_price > protected_high["price"]:
                out.at[label, "protected_high_broken"] = True
                protected_high["active"] = False

        # -------------------------------------------------------------
        # État courant.
        # -------------------------------------------------------------

        if protected_low is not None:

            out.at[label, "protected_low_price"] = protected_low["price"]
            out.at[label, "protected_low_active"] = protected_low["active"]
            out.at[
                label,
                "protected_low_source_origin_idx",
            ] = protected_low["origin_idx"]

            out.at[
                label,
                "protected_low_created_idx",
            ] = protected_low["created_idx"]

            out.at[
                label,
                "protected_low_age_bars",
            ] = i - protected_low["created_idx"]

        if protected_high is not None:

            out.at[label, "protected_high_price"] = protected_high["price"]
            out.at[label, "protected_high_active"] = protected_high["active"]

            out.at[
                label,
                "protected_high_source_origin_idx",
            ] = protected_high["origin_idx"]

            out.at[
                label,
                "protected_high_created_idx",
            ] = protected_high["created_idx"]

            out.at[
                label,
                "protected_high_age_bars",
            ] = i - protected_high["created_idx"]

    return out
