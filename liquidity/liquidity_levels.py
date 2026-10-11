from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiquidityLevelParams:
    """
    Construction des niveaux de liquidité structurelle.

    BSL = Buy-Side Liquidity
        dernier Swing High confirmé.

    SSL = Sell-Side Liquidity
        dernier Swing Low confirmé.

    take_mode:
        "wick"  -> niveau considéré pris dès dépassement du high/low
        "close" -> niveau considéré pris uniquement par clôture

    IMPORTANT:
        Le niveau n'existe qu'une fois le swing confirmé.
    """

    take_mode: str = "wick"

    def __post_init__(self) -> None:
        if self.take_mode not in {"wick", "close"}:
            raise ValueError(
                "take_mode doit être 'wick' ou 'close'"
            )


def detect_liquidity_levels(
    df: pd.DataFrame,
    params: LiquidityLevelParams = LiquidityLevelParams(),
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
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "detect_liquidity_levels(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    n = len(out)

    # -------------------------------------------------------------------------
    # STATE OUTPUTS
    # -------------------------------------------------------------------------

    out["bsl_active"] = False
    out["ssl_active"] = False

    out["bsl_price"] = np.nan
    out["ssl_price"] = np.nan

    out["bsl_taken"] = False
    out["ssl_taken"] = False

    for prefix in ("bsl", "ssl"):

        out[f"{prefix}_origin_idx"] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[f"{prefix}_confirmation_idx"] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[f"{prefix}_age_bars"] = pd.Series(
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

    swing_high = out["swing_high"].to_numpy(
        dtype=bool,
    )

    swing_low = out["swing_low"].to_numpy(
        dtype=bool,
    )

    swing_high_price = out[
        "swing_high_price"
    ].to_numpy(
        dtype=np.float64,
    )

    swing_low_price = out[
        "swing_low_price"
    ].to_numpy(
        dtype=np.float64,
    )

    current_bsl = None
    current_ssl = None

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # NEW CONFIRMED LIQUIDITY LEVEL
        # =====================================================================

        if swing_high[i]:

            current_bsl = {
                "price":
                    swing_high_price[i],

                "origin_idx":
                    int(
                        out.iloc[i][
                            "swing_high_origin_idx"
                        ]
                    ),

                "confirmation_idx":
                    i,

                "active":
                    True,
            }

        if swing_low[i]:

            current_ssl = {
                "price":
                    swing_low_price[i],

                "origin_idx":
                    int(
                        out.iloc[i][
                            "swing_low_origin_idx"
                        ]
                    ),

                "confirmation_idx":
                    i,

                "active":
                    True,
            }

        # =====================================================================
        # TEST LIQUIDITY TAKEN
        # =====================================================================

        upside_price = (
            high[i]
            if params.take_mode == "wick"
            else close[i]
        )

        downside_price = (
            low[i]
            if params.take_mode == "wick"
            else close[i]
        )

        if (
            current_bsl is not None
            and current_bsl["active"]
            and i > current_bsl["confirmation_idx"]
            and upside_price > current_bsl["price"]
        ):

            out.at[
                label,
                "bsl_taken",
            ] = True

            current_bsl["active"] = False

        if (
            current_ssl is not None
            and current_ssl["active"]
            and i > current_ssl["confirmation_idx"]
            and downside_price < current_ssl["price"]
        ):

            out.at[
                label,
                "ssl_taken",
            ] = True

            current_ssl["active"] = False

        # =====================================================================
        # CURRENT STATE
        # =====================================================================

        if current_bsl is not None:

            out.at[
                label,
                "bsl_price",
            ] = current_bsl["price"]

            out.at[
                label,
                "bsl_active",
            ] = current_bsl["active"]

            out.at[
                label,
                "bsl_origin_idx",
            ] = current_bsl["origin_idx"]

            out.at[
                label,
                "bsl_confirmation_idx",
            ] = current_bsl[
                "confirmation_idx"
            ]

            out.at[
                label,
                "bsl_age_bars",
            ] = (
                i
                - current_bsl[
                    "confirmation_idx"
                ]
            )

        if current_ssl is not None:

            out.at[
                label,
                "ssl_price",
            ] = current_ssl["price"]

            out.at[
                label,
                "ssl_active",
            ] = current_ssl["active"]

            out.at[
                label,
                "ssl_origin_idx",
            ] = current_ssl["origin_idx"]

            out.at[
                label,
                "ssl_confirmation_idx",
            ] = current_ssl[
                "confirmation_idx"
            ]

            out.at[
                label,
                "ssl_age_bars",
            ] = (
                i
                - current_ssl[
                    "confirmation_idx"
                ]
            )

    return out
