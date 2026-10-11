from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OldHighLowTargetParams:
    """
    Old High / Low Target v0.1.

    Utilise les swings CONFIRMÉS comme anciens extrema.

    Bull target:
        plus proche ancien swing high encore au-dessus du prix.

    Bear target:
        plus proche ancien swing low encore sous le prix.

    max_age_bars:
        âge maximal du swing cible.

    consume_taken:
        si True, un niveau déjà traversé n'est plus candidat.

    take_mode:
        "wick"
        "close"
    """

    max_age_bars: int = 500

    consume_taken: bool = True

    take_mode: str = "wick"

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )

        if self.take_mode not in {
            "wick",
            "close",
        }:
            raise ValueError(
                "take_mode doit être wick ou close"
            )


def detect_old_high_low_targets(
    df: pd.DataFrame,
    params: OldHighLowTargetParams = OldHighLowTargetParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
        params.atr_column,

        "swing_high",
        "swing_low",

        "swing_high_price",
        "swing_low_price",

        "swing_high_origin_idx",
        "swing_low_origin_idx",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_old_high_low_targets(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    high = out[
        "high"
    ].to_numpy(
        dtype=np.float64,
    )

    low = out[
        "low"
    ].to_numpy(
        dtype=np.float64,
    )

    close = out[
        "close"
    ].to_numpy(
        dtype=np.float64,
    )

    atr = out[
        params.atr_column
    ].to_numpy(
        dtype=np.float64,
    )

    swing_high = out[
        "swing_high"
    ].to_numpy(
        dtype=bool,
    )

    swing_low = out[
        "swing_low"
    ].to_numpy(
        dtype=bool,
    )

    out[
        "target_old_high"
    ] = np.nan

    out[
        "target_old_low"
    ] = np.nan

    out[
        "target_old_high_available"
    ] = False

    out[
        "target_old_low_available"
    ] = False

    out[
        "target_old_high_distance"
    ] = np.nan

    out[
        "target_old_low_distance"
    ] = np.nan

    out[
        "target_old_high_distance_atr"
    ] = np.nan

    out[
        "target_old_low_distance_atr"
    ] = np.nan

    for prefix in (
        "target_old_high",
        "target_old_low",
    ):

        out[
            f"{prefix}_source_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_origin_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_age_bars"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    old_highs = []
    old_lows = []

    for i in range(n):

        # =====================================================================
        # UPDATE TAKEN STATE BEFORE ADDING NEW CONFIRMED SWINGS
        # =====================================================================

        for level in old_highs:

            if level["taken"]:
                continue

            taken = (
                high[i] > level["price"]
                if params.take_mode == "wick"
                else close[i] > level["price"]
            )

            if (
                i > level["confirmation_idx"]
                and taken
            ):
                level["taken"] = True

        for level in old_lows:

            if level["taken"]:
                continue

            taken = (
                low[i] < level["price"]
                if params.take_mode == "wick"
                else close[i] < level["price"]
            )

            if (
                i > level["confirmation_idx"]
                and taken
            ):
                level["taken"] = True

        # =====================================================================
        # ADD NEW CONFIRMED SWINGS
        # =====================================================================

        if swing_high[i]:

            origin = out.iloc[i][
                "swing_high_origin_idx"
            ]

            price = out.iloc[i][
                "swing_high_price"
            ]

            if (
                pd.notna(origin)
                and pd.notna(price)
            ):

                old_highs.append(
                    {
                        "price":
                            float(price),

                        "origin_idx":
                            int(origin),

                        "confirmation_idx":
                            i,

                        "taken":
                            False,
                    }
                )

        if swing_low[i]:

            origin = out.iloc[i][
                "swing_low_origin_idx"
            ]

            price = out.iloc[i][
                "swing_low_price"
            ]

            if (
                pd.notna(origin)
                and pd.notna(price)
            ):

                old_lows.append(
                    {
                        "price":
                            float(price),

                        "origin_idx":
                            int(origin),

                        "confirmation_idx":
                            i,

                        "taken":
                            False,
                    }
                )

        # =====================================================================
        # DROP VERY OLD LEVELS
        # =====================================================================

        old_highs = [
            x
            for x in old_highs
            if (
                i - x["confirmation_idx"]
                <= params.max_age_bars
            )
        ]

        old_lows = [
            x
            for x in old_lows
            if (
                i - x["confirmation_idx"]
                <= params.max_age_bars
            )
        ]

        # =====================================================================
        # CANDIDATES
        # =====================================================================

        high_candidates = [
            x
            for x in old_highs
            if (
                x["price"] > close[i]
                and (
                    not params.consume_taken
                    or not x["taken"]
                )
            )
        ]

        low_candidates = [
            x
            for x in old_lows
            if (
                x["price"] < close[i]
                and (
                    not params.consume_taken
                    or not x["taken"]
                )
            )
        ]

        label = out.index[i]

        # Nearest high above.
        if high_candidates:

            selected = min(
                high_candidates,
                key=lambda x: x["price"],
            )

            distance = (
                selected["price"]
                - close[i]
            )

            out.at[
                label,
                "target_old_high",
            ] = selected["price"]

            out.at[
                label,
                "target_old_high_available",
            ] = True

            out.at[
                label,
                "target_old_high_distance",
            ] = distance

            out.at[
                label,
                "target_old_high_source_idx",
            ] = selected[
                "confirmation_idx"
            ]

            out.at[
                label,
                "target_old_high_origin_idx",
            ] = selected[
                "origin_idx"
            ]

            out.at[
                label,
                "target_old_high_age_bars",
            ] = (
                i
                - selected[
                    "confirmation_idx"
                ]
            )

            if (
                np.isfinite(atr[i])
                and atr[i] > 0
            ):

                out.at[
                    label,
                    "target_old_high_distance_atr",
                ] = (
                    distance / atr[i]
                )

        # Nearest low below.
        if low_candidates:

            selected = max(
                low_candidates,
                key=lambda x: x["price"],
            )

            distance = (
                close[i]
                - selected["price"]
            )

            out.at[
                label,
                "target_old_low",
            ] = selected["price"]

            out.at[
                label,
                "target_old_low_available",
            ] = True

            out.at[
                label,
                "target_old_low_distance",
            ] = distance

            out.at[
                label,
                "target_old_low_source_idx",
            ] = selected[
                "confirmation_idx"
            ]

            out.at[
                label,
                "target_old_low_origin_idx",
            ] = selected[
                "origin_idx"
            ]

            out.at[
                label,
                "target_old_low_age_bars",
            ] = (
                i
                - selected[
                    "confirmation_idx"
                ]
            )

            if (
                np.isfinite(atr[i])
                and atr[i] > 0
            ):

                out.at[
                    label,
                    "target_old_low_distance_atr",
                ] = (
                    distance / atr[i]
                )

    return out
