from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StackedImbalanceParams:
    """
    Stacked Imbalances v0.1.

    Détecte plusieurs FVG de même direction regroupées
    temporellement et spatialement.

    min_count:
        nombre minimal de FVG dans le stack.

    max_gap_bars:
        distance maximale entre deux FVG successives.

    spatial_mode:
        "overlap"
            les zones doivent se chevaucher.

        "distance_atr"
            elles peuvent être séparées mais leur distance doit être
            <= max_distance_atr.

    max_distance_atr:
        utilisé avec spatial_mode="distance_atr".

    ATR:
        normalisation de la distance entre zones.
    """

    min_count: int = 2

    max_gap_bars: int = 5

    spatial_mode: str = "distance_atr"

    max_distance_atr: float = 0.50

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.min_count < 2:
            raise ValueError(
                "min_count doit être >= 2"
            )

        if self.max_gap_bars < 1:
            raise ValueError(
                "max_gap_bars doit être >= 1"
            )

        if self.spatial_mode not in {
            "overlap",
            "distance_atr",
        }:
            raise ValueError(
                "spatial_mode doit être overlap ou distance_atr"
            )

        if self.max_distance_atr < 0:
            raise ValueError(
                "max_distance_atr doit être >= 0"
            )


def _zone_distance(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
) -> float:

    if (
        high_a >= low_b
        and high_b >= low_a
    ):
        return 0.0

    if high_a < low_b:
        return (
            low_b
            - high_a
        )

    return (
        low_a
        - high_b
    )


def detect_stacked_imbalances(
    df: pd.DataFrame,
    params: StackedImbalanceParams = StackedImbalanceParams(),
) -> pd.DataFrame:

    required = {
        "fvg_bull",
        "fvg_bear",

        "fvg_bull_low",
        "fvg_bull_high",

        "fvg_bear_low",
        "fvg_bear_high",

        params.atr_column,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_stacked_imbalances(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    for side in (
        "stacked_imbalance_bull",
        "stacked_imbalance_bear",
    ):

        out[side] = False

        out[f"{side}_low"] = np.nan
        out[f"{side}_high"] = np.nan
        out[f"{side}_mid"] = np.nan

        out[f"{side}_count"] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[f"{side}_span_bars"] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{side}_max_internal_distance_atr"
        ] = np.nan

        out[
            f"{side}_first_source_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{side}_confirmation_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    atr = out[
        params.atr_column
    ].to_numpy(
        dtype=np.float64,
    )

    bull_stack = []
    bear_stack = []

    def compatible(
        previous: dict,
        current: dict,
        atr_value: float,
    ) -> tuple[bool, float]:

        gap = (
            current["idx"]
            - previous["idx"]
        )

        if gap > params.max_gap_bars:
            return (
                False,
                np.nan,
            )

        distance = _zone_distance(
            previous["low"],
            previous["high"],
            current["low"],
            current["high"],
        )

        distance_atr = (
            distance / atr_value
            if (
                np.isfinite(atr_value)
                and atr_value > 0
            )
            else np.nan
        )

        if params.spatial_mode == "overlap":

            return (
                distance == 0,
                distance_atr,
            )

        return (
            np.isfinite(distance_atr)
            and (
                distance_atr
                <= params.max_distance_atr
            ),
            distance_atr,
        )

    for i in range(n):

        row = out.iloc[i]
        label = out.index[i]

        # =====================================================================
        # BULL FVG
        # =====================================================================

        if bool(
            row["fvg_bull"]
        ):

            current = {
                "idx":
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

                "distance_atr":
                    0.0,
            }

            if bull_stack:

                ok, distance_atr = compatible(
                    bull_stack[-1],
                    current,
                    atr[i],
                )

            else:

                ok = False
                distance_atr = np.nan

            if ok:

                current[
                    "distance_atr"
                ] = distance_atr

                bull_stack.append(
                    current
                )

            else:

                bull_stack = [
                    current
                ]

            if (
                len(bull_stack)
                >= params.min_count
            ):

                zone_low = min(
                    z["low"]
                    for z in bull_stack
                )

                zone_high = max(
                    z["high"]
                    for z in bull_stack
                )

                distances = [
                    z["distance_atr"]
                    for z in bull_stack[1:]
                    if np.isfinite(
                        z["distance_atr"]
                    )
                ]

                max_distance = (
                    max(distances)
                    if distances
                    else 0.0
                )

                out.at[
                    label,
                    "stacked_imbalance_bull",
                ] = True

                out.at[
                    label,
                    "stacked_imbalance_bull_low",
                ] = zone_low

                out.at[
                    label,
                    "stacked_imbalance_bull_high",
                ] = zone_high

                out.at[
                    label,
                    "stacked_imbalance_bull_mid",
                ] = (
                    zone_low
                    + zone_high
                ) / 2.0

                out.at[
                    label,
                    "stacked_imbalance_bull_count",
                ] = len(
                    bull_stack
                )

                out.at[
                    label,
                    "stacked_imbalance_bull_span_bars",
                ] = (
                    i
                    - bull_stack[0]["idx"]
                )

                out.at[
                    label,
                    "stacked_imbalance_bull_max_internal_distance_atr",
                ] = max_distance

                out.at[
                    label,
                    "stacked_imbalance_bull_first_source_idx",
                ] = bull_stack[0][
                    "idx"
                ]

                out.at[
                    label,
                    "stacked_imbalance_bull_confirmation_idx",
                ] = i

        # =====================================================================
        # BEAR FVG
        # =====================================================================

        if bool(
            row["fvg_bear"]
        ):

            current = {
                "idx":
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

                "distance_atr":
                    0.0,
            }

            if bear_stack:

                ok, distance_atr = compatible(
                    bear_stack[-1],
                    current,
                    atr[i],
                )

            else:

                ok = False
                distance_atr = np.nan

            if ok:

                current[
                    "distance_atr"
                ] = distance_atr

                bear_stack.append(
                    current
                )

            else:

                bear_stack = [
                    current
                ]

            if (
                len(bear_stack)
                >= params.min_count
            ):

                zone_low = min(
                    z["low"]
                    for z in bear_stack
                )

                zone_high = max(
                    z["high"]
                    for z in bear_stack
                )

                distances = [
                    z["distance_atr"]
                    for z in bear_stack[1:]
                    if np.isfinite(
                        z["distance_atr"]
                    )
                ]

                max_distance = (
                    max(distances)
                    if distances
                    else 0.0
                )

                out.at[
                    label,
                    "stacked_imbalance_bear",
                ] = True

                out.at[
                    label,
                    "stacked_imbalance_bear_low",
                ] = zone_low

                out.at[
                    label,
                    "stacked_imbalance_bear_high",
                ] = zone_high

                out.at[
                    label,
                    "stacked_imbalance_bear_mid",
                ] = (
                    zone_low
                    + zone_high
                ) / 2.0

                out.at[
                    label,
                    "stacked_imbalance_bear_count",
                ] = len(
                    bear_stack
                )

                out.at[
                    label,
                    "stacked_imbalance_bear_span_bars",
                ] = (
                    i
                    - bear_stack[0]["idx"]
                )

                out.at[
                    label,
                    "stacked_imbalance_bear_max_internal_distance_atr",
                ] = max_distance

                out.at[
                    label,
                    "stacked_imbalance_bear_first_source_idx",
                ] = bear_stack[0][
                    "idx"
                ]

                out.at[
                    label,
                    "stacked_imbalance_bear_confirmation_idx",
                ] = i

    return out
