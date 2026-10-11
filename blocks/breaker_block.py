from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BreakerBlockParams:
    """
    Breaker Block v0.1.

    Un Order Block invalidé devient une zone de polarité opposée.

    Bullish OB invalidé vers le bas:
        devient Breaker bearish.

    Bearish OB invalidé vers le haut:
        devient Breaker bullish.

    retest_required:
        si True, l'événement Breaker n'est confirmé qu'au premier
        retest de la zone après invalidation.

    max_retest_bars:
        durée maximale de vie avant retest.
    """

    retest_required: bool = True
    max_retest_bars: int = 50

    def __post_init__(self) -> None:

        if self.max_retest_bars < 1:
            raise ValueError(
                "max_retest_bars doit être >= 1"
            )


def detect_breaker_blocks(
    df: pd.DataFrame,
    params: BreakerBlockParams = BreakerBlockParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",

        "ob_bull",
        "ob_bear",

        "ob_bull_low",
        "ob_bull_high",

        "ob_bear_low",
        "ob_bear_high",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_breaker_blocks(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    for side in (
        "breaker_bull",
        "breaker_bear",
    ):

        out[side] = False

        out[f"{side}_low"] = np.nan
        out[f"{side}_high"] = np.nan
        out[f"{side}_mid"] = np.nan

        for suffix in (
            "source_ob_idx",
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

    pending_bull = []
    pending_bear = []

    active_ob_bull = []
    active_ob_bear = []

    for i in range(n):

        # =====================================================================
        # TRACK NEW OB
        # =====================================================================

        if bool(
            out.iloc[i]["ob_bull"]
        ):

            active_ob_bull.append(
                {
                    "source_idx":
                        i,

                    "low":
                        float(
                            out.iloc[i][
                                "ob_bull_low"
                            ]
                        ),

                    "high":
                        float(
                            out.iloc[i][
                                "ob_bull_high"
                            ]
                        ),
                }
            )

        if bool(
            out.iloc[i]["ob_bear"]
        ):

            active_ob_bear.append(
                {
                    "source_idx":
                        i,

                    "low":
                        float(
                            out.iloc[i][
                                "ob_bear_low"
                            ]
                        ),

                    "high":
                        float(
                            out.iloc[i][
                                "ob_bear_high"
                            ]
                        ),
                }
            )

        # =====================================================================
        # INVALIDATE BULL OB -> PENDING BEAR BREAKER
        # =====================================================================

        survivors = []

        for zone in active_ob_bull:

            if (
                i > zone["source_idx"]
                and close[i] < zone["low"]
            ):

                pending_bear.append(
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

        active_ob_bull = survivors

        # =====================================================================
        # INVALIDATE BEAR OB -> PENDING BULL BREAKER
        # =====================================================================

        survivors = []

        for zone in active_ob_bear:

            if (
                i > zone["source_idx"]
                and close[i] > zone["high"]
            ):

                pending_bull.append(
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

        active_ob_bear = survivors

        # =====================================================================
        # RETEST BULL BREAKER
        # =====================================================================

        survivors = []

        for zone in pending_bull:

            age = (
                i
                - zone[
                    "invalidation_idx"
                ]
            )

            if age > params.max_retest_bars:
                continue

            touched = (
                i > zone[
                    "invalidation_idx"
                ]
                and low[i] <= zone["high"]
                and high[i] >= zone["low"]
            )

            confirm = (
                touched
                if params.retest_required
                else (
                    i
                    == zone[
                        "invalidation_idx"
                    ]
                )
            )

            if confirm:

                label = out.index[i]

                out.at[
                    label,
                    "breaker_bull",
                ] = True

                out.at[
                    label,
                    "breaker_bull_low",
                ] = zone["low"]

                out.at[
                    label,
                    "breaker_bull_high",
                ] = zone["high"]

                out.at[
                    label,
                    "breaker_bull_mid",
                ] = (
                    zone["low"]
                    + zone["high"]
                ) / 2.0

                out.at[
                    label,
                    "breaker_bull_source_ob_idx",
                ] = zone[
                    "source_idx"
                ]

                out.at[
                    label,
                    "breaker_bull_invalidation_idx",
                ] = zone[
                    "invalidation_idx"
                ]

                out.at[
                    label,
                    "breaker_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "breaker_bull_age_bars",
                ] = age

            else:

                survivors.append(
                    zone
                )

        pending_bull = survivors

        # =====================================================================
        # RETEST BEAR BREAKER
        # =====================================================================

        survivors = []

        for zone in pending_bear:

            age = (
                i
                - zone[
                    "invalidation_idx"
                ]
            )

            if age > params.max_retest_bars:
                continue

            touched = (
                i > zone[
                    "invalidation_idx"
                ]
                and high[i] >= zone["low"]
                and low[i] <= zone["high"]
            )

            confirm = (
                touched
                if params.retest_required
                else (
                    i
                    == zone[
                        "invalidation_idx"
                    ]
                )
            )

            if confirm:

                label = out.index[i]

                out.at[
                    label,
                    "breaker_bear",
                ] = True

                out.at[
                    label,
                    "breaker_bear_low",
                ] = zone["low"]

                out.at[
                    label,
                    "breaker_bear_high",
                ] = zone["high"]

                out.at[
                    label,
                    "breaker_bear_mid",
                ] = (
                    zone["low"]
                    + zone["high"]
                ) / 2.0

                out.at[
                    label,
                    "breaker_bear_source_ob_idx",
                ] = zone[
                    "source_idx"
                ]

                out.at[
                    label,
                    "breaker_bear_invalidation_idx",
                ] = zone[
                    "invalidation_idx"
                ]

                out.at[
                    label,
                    "breaker_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "breaker_bear_age_bars",
                ] = age

            else:

                survivors.append(
                    zone
                )

        pending_bear = survivors

    return out
