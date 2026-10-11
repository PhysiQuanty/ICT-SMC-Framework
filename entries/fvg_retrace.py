from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FVGRetraceEntryParams:
    """
    FVG Retrace Entry v0.1.

    Chaque FVG devient une zone candidate après sa création.

    entry_mode:
        "touch_close"
            premier overlap avec la zone,
            prix candidat = close.

        "near"
            bull -> fvg_high
            bear -> fvg_low

        "mid"
            CE / midpoint de la FVG.

        "full"
            bull -> fvg_low
            bear -> fvg_high

    max_age_bars:
        expiration d'une FVG candidate.

    first_touch_only:
        une entrée candidate par FVG.
    """

    entry_mode: str = "touch_close"

    max_age_bars: int = 100
    first_touch_only: bool = True

    def __post_init__(self) -> None:

        if self.entry_mode not in {
            "touch_close",
            "near",
            "mid",
            "full",
        }:
            raise ValueError(
                "entry_mode doit être touch_close, near, mid ou full"
            )

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )


def detect_fvg_retrace_entries(
    df: pd.DataFrame,
    params: FVGRetraceEntryParams = FVGRetraceEntryParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",

        "fvg_bull",
        "fvg_bear",

        "fvg_bull_low",
        "fvg_bull_high",
        "fvg_bull_mid",

        "fvg_bear_low",
        "fvg_bear_high",
        "fvg_bear_mid",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "detect_fvg_retrace_entries(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)

    out["entry_fvg_bull"] = False
    out["entry_fvg_bear"] = False

    out["entry_fvg_bull_price"] = np.nan
    out["entry_fvg_bear_price"] = np.nan

    out["entry_fvg_bull_fill_ratio"] = np.nan
    out["entry_fvg_bear_fill_ratio"] = np.nan

    for side in (
        "entry_fvg_bull",
        "entry_fvg_bear",
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
        # EXISTING BULL FVGs
        # =====================================================================

        survivors = []

        for zone in active_bull:

            age = i - zone["idx"]

            if age > params.max_age_bars:
                continue

            size = (
                zone["high"]
                - zone["low"]
            )

            if params.entry_mode == "mid":

                touched = (
                    low[i] <= zone["mid"]
                    and high[i] >= zone["mid"]
                )

                entry_price = zone["mid"]

            elif params.entry_mode == "full":

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

            penetration = (
                zone["high"]
                - max(
                    low[i],
                    zone["low"],
                )
            )

            fill_ratio = (
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
                    "entry_fvg_bull",
                ] = True

                out.at[
                    label,
                    "entry_fvg_bull_price",
                ] = entry_price

                out.at[
                    label,
                    "entry_fvg_bull_fill_ratio",
                ] = fill_ratio

                out.at[
                    label,
                    "entry_fvg_bull_source_idx",
                ] = zone["idx"]

                out.at[
                    label,
                    "entry_fvg_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "entry_fvg_bull_age_bars",
                ] = age

                if not params.first_touch_only:
                    survivors.append(zone)

            else:
                survivors.append(zone)

        active_bull = survivors

        # =====================================================================
        # EXISTING BEAR FVGs
        # =====================================================================

        survivors = []

        for zone in active_bear:

            age = i - zone["idx"]

            if age > params.max_age_bars:
                continue

            size = (
                zone["high"]
                - zone["low"]
            )

            if params.entry_mode == "mid":

                touched = (
                    high[i] >= zone["mid"]
                    and low[i] <= zone["mid"]
                )

                entry_price = zone["mid"]

            elif params.entry_mode == "full":

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

            penetration = (
                min(
                    high[i],
                    zone["high"],
                )
                - zone["low"]
            )

            fill_ratio = (
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
                    "entry_fvg_bear",
                ] = True

                out.at[
                    label,
                    "entry_fvg_bear_price",
                ] = entry_price

                out.at[
                    label,
                    "entry_fvg_bear_fill_ratio",
                ] = fill_ratio

                out.at[
                    label,
                    "entry_fvg_bear_source_idx",
                ] = zone["idx"]

                out.at[
                    label,
                    "entry_fvg_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "entry_fvg_bear_age_bars",
                ] = age

                if not params.first_touch_only:
                    survivors.append(zone)

            else:
                survivors.append(zone)

        active_bear = survivors

        # =====================================================================
        # NEW FVGs ADDED AFTER RETRACE CHECK
        # =====================================================================

        row = out.iloc[i]

        if bool(row["fvg_bull"]):

            active_bull.append(
                {
                    "idx":
                        i,

                    "low":
                        float(
                            row["fvg_bull_low"]
                        ),

                    "high":
                        float(
                            row["fvg_bull_high"]
                        ),

                    "mid":
                        float(
                            row["fvg_bull_mid"]
                        ),
                }
            )

        if bool(row["fvg_bear"]):

            active_bear.append(
                {
                    "idx":
                        i,

                    "low":
                        float(
                            row["fvg_bear_low"]
                        ),

                    "high":
                        float(
                            row["fvg_bear_high"]
                        ),

                    "mid":
                        float(
                            row["fvg_bear_mid"]
                        ),
                }
            )

    return out
