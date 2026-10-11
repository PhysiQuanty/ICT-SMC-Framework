from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OBRetestEntryParams:
    """
    Order Block Retest Entry v0.1.

    Chaque Order Block confirmé devient une zone candidate.

    entry_mode:
        "touch_close"
            premier contact, prix candidat = close.

        "near"
            bull -> OB high
            bear -> OB low.

        "mid"
            midpoint de l'OB.

        "deep"
            bull -> OB low
            bear -> OB high.

    invalidation_mode:
        "close"
        "wick"

    max_age_bars:
        expiration de la zone.

    first_touch_only:
        une entrée candidate par OB.
    """

    entry_mode: str = "touch_close"

    invalidation_mode: str = "close"

    max_age_bars: int = 100
    first_touch_only: bool = True

    def __post_init__(self) -> None:

        if self.entry_mode not in {
            "touch_close",
            "near",
            "mid",
            "deep",
        }:
            raise ValueError(
                "entry_mode doit être touch_close, near, mid ou deep"
            )

        if self.invalidation_mode not in {
            "close",
            "wick",
        }:
            raise ValueError(
                "invalidation_mode doit être close ou wick"
            )

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )


def detect_ob_retest_entries(
    df: pd.DataFrame,
    params: OBRetestEntryParams = OBRetestEntryParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",

        "ob_bull",
        "ob_bear",

        "ob_bull_low",
        "ob_bull_high",
        "ob_bull_mid",

        "ob_bear_low",
        "ob_bear_high",
        "ob_bear_mid",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "detect_ob_retest_entries(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)

    out["entry_ob_bull"] = False
    out["entry_ob_bear"] = False

    out["entry_ob_bull_price"] = np.nan
    out["entry_ob_bear_price"] = np.nan

    out["entry_ob_bull_penetration_ratio"] = np.nan
    out["entry_ob_bear_penetration_ratio"] = np.nan

    for side in (
        "entry_ob_bull",
        "entry_ob_bear",
    ):
        for suffix in (
            "source_idx",
            "confirmation_idx",
            "age_bars",
        ):
            out[f"{side}_{suffix}"] = pd.Series(
                pd.array(
                    [pd.NA] * n,
                    dtype="Int64",
                ),
                index=out.index,
            )

    active_bull = []
    active_bear = []

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # BULL OB RETESTS
        # =====================================================================

        survivors = []

        for zone in active_bull:

            age = i - zone["idx"]

            if age > params.max_age_bars:
                continue

            invalidation_price = (
                low[i]
                if params.invalidation_mode == "wick"
                else close[i]
            )

            if invalidation_price < zone["low"]:
                continue

            if params.entry_mode == "mid":

                touched = (
                    low[i] <= zone["mid"]
                    and high[i] >= zone["mid"]
                )

                entry_price = zone["mid"]

            elif params.entry_mode == "deep":

                touched = (
                    low[i] <= zone["low"]
                )

                entry_price = zone["low"]

            else:

                touched = (
                    low[i] <= zone["high"]
                    and high[i] >= zone["low"]
                )

                entry_price = (
                    zone["high"]
                    if params.entry_mode == "near"
                    else close[i]
                )

            size = (
                zone["high"]
                - zone["low"]
            )

            penetration = (
                zone["high"]
                - max(
                    low[i],
                    zone["low"],
                )
            )

            ratio = (
                float(
                    np.clip(
                        penetration / size,
                        0.0,
                        1.0,
                    )
                )
                if size > 0
                else np.nan
            )

            if touched:

                out.at[
                    label,
                    "entry_ob_bull",
                ] = True

                out.at[
                    label,
                    "entry_ob_bull_price",
                ] = entry_price

                out.at[
                    label,
                    "entry_ob_bull_penetration_ratio",
                ] = ratio

                out.at[
                    label,
                    "entry_ob_bull_source_idx",
                ] = zone["idx"]

                out.at[
                    label,
                    "entry_ob_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "entry_ob_bull_age_bars",
                ] = age

                if not params.first_touch_only:
                    survivors.append(zone)

            else:
                survivors.append(zone)

        active_bull = survivors

        # =====================================================================
        # BEAR OB RETESTS
        # =====================================================================

        survivors = []

        for zone in active_bear:

            age = i - zone["idx"]

            if age > params.max_age_bars:
                continue

            invalidation_price = (
                high[i]
                if params.invalidation_mode == "wick"
                else close[i]
            )

            if invalidation_price > zone["high"]:
                continue

            if params.entry_mode == "mid":

                touched = (
                    high[i] >= zone["mid"]
                    and low[i] <= zone["mid"]
                )

                entry_price = zone["mid"]

            elif params.entry_mode == "deep":

                touched = (
                    high[i] >= zone["high"]
                )

                entry_price = zone["high"]

            else:

                touched = (
                    high[i] >= zone["low"]
                    and low[i] <= zone["high"]
                )

                entry_price = (
                    zone["low"]
                    if params.entry_mode == "near"
                    else close[i]
                )

            size = (
                zone["high"]
                - zone["low"]
            )

            penetration = (
                min(
                    high[i],
                    zone["high"],
                )
                - zone["low"]
            )

            ratio = (
                float(
                    np.clip(
                        penetration / size,
                        0.0,
                        1.0,
                    )
                )
                if size > 0
                else np.nan
            )

            if touched:

                out.at[
                    label,
                    "entry_ob_bear",
                ] = True

                out.at[
                    label,
                    "entry_ob_bear_price",
                ] = entry_price

                out.at[
                    label,
                    "entry_ob_bear_penetration_ratio",
                ] = ratio

                out.at[
                    label,
                    "entry_ob_bear_source_idx",
                ] = zone["idx"]

                out.at[
                    label,
                    "entry_ob_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "entry_ob_bear_age_bars",
                ] = age

                if not params.first_touch_only:
                    survivors.append(zone)

            else:
                survivors.append(zone)

        active_bear = survivors

        # =====================================================================
        # ADD NEW OBs ONLY AFTER CURRENT BAR RETEST CHECK
        # =====================================================================

        row = out.iloc[i]

        if bool(row["ob_bull"]):

            active_bull.append(
                {
                    "idx":
                        i,

                    "low":
                        float(
                            row["ob_bull_low"]
                        ),

                    "high":
                        float(
                            row["ob_bull_high"]
                        ),

                    "mid":
                        float(
                            row["ob_bull_mid"]
                        ),
                }
            )

        if bool(row["ob_bear"]):

            active_bear.append(
                {
                    "idx":
                        i,

                    "low":
                        float(
                            row["ob_bear_low"]
                        ),

                    "high":
                        float(
                            row["ob_bear_high"]
                        ),

                    "mid":
                        float(
                            row["ob_bear_mid"]
                        ),
                }
            )

    return out
