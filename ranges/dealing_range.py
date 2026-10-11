from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DealingRangeParams:
    """
    Dealing Range v0.1.

    Construction strictement causale à partir des swings CONFIRMÉS.

    Bullish dealing range:
        Swing Low -> Swing High

    Bearish dealing range:
        Swing High -> Swing Low

    IMPORTANT:
        On utilise les origin_idx des swings pour déterminer
        l'ordre réel des pivots, mais le range n'existe qu'au moment
        où le second swing est confirmé.

    min_range_atr:
        taille minimale du range en ATR.

    atr_column:
        colonne ATR pour normalisation.
    """

    min_range_atr: float = 0.0
    atr_column: str = "atr"

    def __post_init__(self) -> None:
        if self.min_range_atr < 0:
            raise ValueError(
                "min_range_atr doit être >= 0"
            )


def detect_dealing_range(
    df: pd.DataFrame,
    params: DealingRangeParams = DealingRangeParams(),
) -> pd.DataFrame:

    required = {
        "swing_high",
        "swing_low",

        "swing_high_price",
        "swing_low_price",

        "swing_high_origin_idx",
        "swing_low_origin_idx",

        params.atr_column,
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "detect_dealing_range(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    # =========================================================================
    # EVENT FLAGS
    # =========================================================================

    out["dealing_range_bull"] = False
    out["dealing_range_bear"] = False

    # =========================================================================
    # ACTIVE RANGE STATE
    # =========================================================================

    out["dealing_range_active"] = False
    out["dealing_range_direction"] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out["dealing_range_low"] = np.nan
    out["dealing_range_high"] = np.nan
    out["dealing_range_mid"] = np.nan

    out["dealing_range_size"] = np.nan
    out["dealing_range_size_atr"] = np.nan

    for suffix in (
        "start_origin_idx",
        "end_origin_idx",
        "confirmation_idx",
        "age_bars",
    ):
        out[f"dealing_range_{suffix}"] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    swing_high = out[
        "swing_high"
    ].to_numpy(dtype=bool)

    swing_low = out[
        "swing_low"
    ].to_numpy(dtype=bool)

    swing_high_price = out[
        "swing_high_price"
    ].to_numpy(dtype=np.float64)

    swing_low_price = out[
        "swing_low_price"
    ].to_numpy(dtype=np.float64)

    atr = out[
        params.atr_column
    ].to_numpy(dtype=np.float64)

    latest_high = None
    latest_low = None

    current_range = None

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # NEW CONFIRMED HIGH
        # =====================================================================

        if swing_high[i]:

            origin = out.iloc[i][
                "swing_high_origin_idx"
            ]

            if pd.notna(origin):

                origin = int(origin)

                new_high = {
                    "price":
                        float(swing_high_price[i]),

                    "origin_idx":
                        origin,

                    "confirmation_idx":
                        i,
                }

                # -------------------------------------------------------------
                # Low -> High = bullish range
                # -------------------------------------------------------------

                if (
                    latest_low is not None
                    and latest_low["origin_idx"] < origin
                ):

                    low_price = (
                        latest_low["price"]
                    )

                    high_price = (
                        new_high["price"]
                    )

                    size = (
                        high_price
                        - low_price
                    )

                    size_atr = (
                        size / atr[i]
                        if (
                            np.isfinite(atr[i])
                            and atr[i] > 0
                        )
                        else np.nan
                    )

                    threshold_ok = (
                        size > 0
                        and (
                            params.min_range_atr == 0
                            or (
                                np.isfinite(size_atr)
                                and size_atr
                                >= params.min_range_atr
                            )
                        )
                    )

                    if threshold_ok:

                        current_range = {
                            "direction":
                                "bull",

                            "low":
                                low_price,

                            "high":
                                high_price,

                            "mid":
                                (
                                    low_price
                                    + high_price
                                ) / 2.0,

                            "size":
                                size,

                            "size_atr":
                                size_atr,

                            "start_origin_idx":
                                latest_low["origin_idx"],

                            "end_origin_idx":
                                origin,

                            "confirmation_idx":
                                i,
                        }

                        out.at[
                            label,
                            "dealing_range_bull",
                        ] = True

                latest_high = new_high

        # =====================================================================
        # NEW CONFIRMED LOW
        # =====================================================================

        if swing_low[i]:

            origin = out.iloc[i][
                "swing_low_origin_idx"
            ]

            if pd.notna(origin):

                origin = int(origin)

                new_low = {
                    "price":
                        float(swing_low_price[i]),

                    "origin_idx":
                        origin,

                    "confirmation_idx":
                        i,
                }

                # -------------------------------------------------------------
                # High -> Low = bearish range
                # -------------------------------------------------------------

                if (
                    latest_high is not None
                    and latest_high["origin_idx"] < origin
                ):

                    high_price = (
                        latest_high["price"]
                    )

                    low_price = (
                        new_low["price"]
                    )

                    size = (
                        high_price
                        - low_price
                    )

                    size_atr = (
                        size / atr[i]
                        if (
                            np.isfinite(atr[i])
                            and atr[i] > 0
                        )
                        else np.nan
                    )

                    threshold_ok = (
                        size > 0
                        and (
                            params.min_range_atr == 0
                            or (
                                np.isfinite(size_atr)
                                and size_atr
                                >= params.min_range_atr
                            )
                        )
                    )

                    if threshold_ok:

                        current_range = {
                            "direction":
                                "bear",

                            "low":
                                low_price,

                            "high":
                                high_price,

                            "mid":
                                (
                                    low_price
                                    + high_price
                                ) / 2.0,

                            "size":
                                size,

                            "size_atr":
                                size_atr,

                            "start_origin_idx":
                                latest_high["origin_idx"],

                            "end_origin_idx":
                                origin,

                            "confirmation_idx":
                                i,
                        }

                        out.at[
                            label,
                            "dealing_range_bear",
                        ] = True

                latest_low = new_low

        # =====================================================================
        # CURRENT STATE
        # =====================================================================

        if current_range is not None:

            out.at[
                label,
                "dealing_range_active",
            ] = True

            out.at[
                label,
                "dealing_range_direction",
            ] = current_range[
                "direction"
            ]

            out.at[
                label,
                "dealing_range_low",
            ] = current_range[
                "low"
            ]

            out.at[
                label,
                "dealing_range_high",
            ] = current_range[
                "high"
            ]

            out.at[
                label,
                "dealing_range_mid",
            ] = current_range[
                "mid"
            ]

            out.at[
                label,
                "dealing_range_size",
            ] = current_range[
                "size"
            ]

            out.at[
                label,
                "dealing_range_size_atr",
            ] = current_range[
                "size_atr"
            ]

            out.at[
                label,
                "dealing_range_start_origin_idx",
            ] = current_range[
                "start_origin_idx"
            ]

            out.at[
                label,
                "dealing_range_end_origin_idx",
            ] = current_range[
                "end_origin_idx"
            ]

            out.at[
                label,
                "dealing_range_confirmation_idx",
            ] = current_range[
                "confirmation_idx"
            ]

            out.at[
                label,
                "dealing_range_age_bars",
            ] = (
                i
                - current_range[
                    "confirmation_idx"
                ]
            )

    return out
