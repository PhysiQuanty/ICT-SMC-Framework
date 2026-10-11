from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InversionFVGParams:
    """
    Inversion FVG / IFVG v0.1.

    Bullish FVG:
        invalidée lorsque le prix traverse sous sa borne basse.
        Elle devient alors candidate IFVG bearish.

    Bearish FVG:
        invalidée lorsque le prix traverse au-dessus de sa borne haute.
        Elle devient candidate IFVG bullish.

    L'IFVG est confirmée au retest ultérieur de cette ancienne zone.

    invalidation_mode:
        "close"
        "wick"

    retest_mode:
        "touch"
        "mid"

    max_retest_bars:
        durée maximale après invalidation.
    """

    invalidation_mode: str = "close"
    retest_mode: str = "touch"

    max_retest_bars: int = 100

    def __post_init__(self) -> None:

        if self.invalidation_mode not in {
            "close",
            "wick",
        }:
            raise ValueError(
                "invalidation_mode doit être close ou wick"
            )

        if self.retest_mode not in {
            "touch",
            "mid",
        }:
            raise ValueError(
                "retest_mode doit être touch ou mid"
            )

        if self.max_retest_bars < 1:
            raise ValueError(
                "max_retest_bars doit être >= 1"
            )


def detect_inversion_fvg(
    df: pd.DataFrame,
    params: InversionFVGParams = InversionFVGParams(),
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

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_inversion_fvg(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    for side in (
        "ifvg_bull",
        "ifvg_bear",
    ):

        out[side] = False

        out[f"{side}_low"] = np.nan
        out[f"{side}_high"] = np.nan
        out[f"{side}_mid"] = np.nan

        for suffix in (
            "source_fvg_idx",
            "invalidation_idx",
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

    high = out["high"].to_numpy(
        dtype=np.float64,
    )

    low = out["low"].to_numpy(
        dtype=np.float64,
    )

    close = out["close"].to_numpy(
        dtype=np.float64,
    )

    active_bull = []
    active_bear = []

    pending_ifvg_bull = []
    pending_ifvg_bear = []

    for i in range(n):

        # =====================================================================
        # INVALIDATE EXISTING BULL FVG -> BEAR IFVG CANDIDATE
        # =====================================================================

        survivors = []

        for zone in active_bull:

            if params.invalidation_mode == "close":

                invalidated = (
                    close[i]
                    < zone["low"]
                )

            else:

                invalidated = (
                    low[i]
                    < zone["low"]
                )

            if (
                i > zone["source_idx"]
                and invalidated
            ):

                pending_ifvg_bear.append(
                    {
                        **zone,
                        "invalidation_idx":
                            i,
                    }
                )

            else:

                survivors.append(
                    zone
                )

        active_bull = survivors

        # =====================================================================
        # INVALIDATE EXISTING BEAR FVG -> BULL IFVG CANDIDATE
        # =====================================================================

        survivors = []

        for zone in active_bear:

            if params.invalidation_mode == "close":

                invalidated = (
                    close[i]
                    > zone["high"]
                )

            else:

                invalidated = (
                    high[i]
                    > zone["high"]
                )

            if (
                i > zone["source_idx"]
                and invalidated
            ):

                pending_ifvg_bull.append(
                    {
                        **zone,
                        "invalidation_idx":
                            i,
                    }
                )

            else:

                survivors.append(
                    zone
                )

        active_bear = survivors

        # =====================================================================
        # RETEST BULL IFVG
        # =====================================================================

        survivors = []

        for zone in pending_ifvg_bull:

            age = (
                i
                - zone[
                    "invalidation_idx"
                ]
            )

            if age > params.max_retest_bars:
                continue

            if i <= zone[
                "invalidation_idx"
            ]:
                survivors.append(
                    zone
                )
                continue

            if params.retest_mode == "touch":

                touched = (
                    low[i] <= zone["high"]
                    and high[i] >= zone["low"]
                )

            else:

                touched = (
                    low[i] <= zone["mid"]
                    and high[i] >= zone["mid"]
                )

            if touched:

                label = out.index[i]

                out.at[
                    label,
                    "ifvg_bull",
                ] = True

                out.at[
                    label,
                    "ifvg_bull_low",
                ] = zone["low"]

                out.at[
                    label,
                    "ifvg_bull_high",
                ] = zone["high"]

                out.at[
                    label,
                    "ifvg_bull_mid",
                ] = zone["mid"]

                out.at[
                    label,
                    "ifvg_bull_source_fvg_idx",
                ] = zone[
                    "source_idx"
                ]

                out.at[
                    label,
                    "ifvg_bull_invalidation_idx",
                ] = zone[
                    "invalidation_idx"
                ]

                out.at[
                    label,
                    "ifvg_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "ifvg_bull_age_bars",
                ] = age

            else:

                survivors.append(
                    zone
                )

        pending_ifvg_bull = survivors

        # =====================================================================
        # RETEST BEAR IFVG
        # =====================================================================

        survivors = []

        for zone in pending_ifvg_bear:

            age = (
                i
                - zone[
                    "invalidation_idx"
                ]
            )

            if age > params.max_retest_bars:
                continue

            if i <= zone[
                "invalidation_idx"
            ]:
                survivors.append(
                    zone
                )
                continue

            if params.retest_mode == "touch":

                touched = (
                    high[i] >= zone["low"]
                    and low[i] <= zone["high"]
                )

            else:

                touched = (
                    high[i] >= zone["mid"]
                    and low[i] <= zone["mid"]
                )

            if touched:

                label = out.index[i]

                out.at[
                    label,
                    "ifvg_bear",
                ] = True

                out.at[
                    label,
                    "ifvg_bear_low",
                ] = zone["low"]

                out.at[
                    label,
                    "ifvg_bear_high",
                ] = zone["high"]

                out.at[
                    label,
                    "ifvg_bear_mid",
                ] = zone["mid"]

                out.at[
                    label,
                    "ifvg_bear_source_fvg_idx",
                ] = zone[
                    "source_idx"
                ]

                out.at[
                    label,
                    "ifvg_bear_invalidation_idx",
                ] = zone[
                    "invalidation_idx"
                ]

                out.at[
                    label,
                    "ifvg_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "ifvg_bear_age_bars",
                ] = age

            else:

                survivors.append(
                    zone
                )

        pending_ifvg_bear = survivors

        # =====================================================================
        # NEW FVGs
        # =====================================================================

        row = out.iloc[i]

        if bool(
            row["fvg_bull"]
        ):

            active_bull.append(
                {
                    "source_idx":
                        i,

                    "low":
                        float(
                            row[
                                "fvg_bull_low"
                            ]
                        ),

                    "high":
                        float(
                            row[
                                "fvg_bull_high"
                            ]
                        ),

                    "mid":
                        float(
                            row[
                                "fvg_bull_mid"
                            ]
                        ),
                }
            )

        if bool(
            row["fvg_bear"]
        ):

            active_bear.append(
                {
                    "source_idx":
                        i,

                    "low":
                        float(
                            row[
                                "fvg_bear_low"
                            ]
                        ),

                    "high":
                        float(
                            row[
                                "fvg_bear_high"
                            ]
                        ),

                    "mid":
                        float(
                            row[
                                "fvg_bear_mid"
                            ]
                        ),
                }
            )

    return out
