from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FVGFillTargetParams:
    """
    FVG Fill Target v0.1.

    Mesure le first-passage d'une FVG créée vers différents
    niveaux de remplissage.

    target_mode:

        "near"
            premier bord de la FVG.

        "mid"
            50% / Consequent Encroachment.

        "full"
            bord opposé = remplissage complet.

    Bull FVG:
        near = high
        mid  = midpoint
        full = low

    Bear FVG:
        near = low
        mid  = midpoint
        full = high

    max_age_bars:
        horizon maximal pour suivre la FVG.

    first_hit_only:
        une émission de hit par FVG.
    """

    target_mode: str = "full"

    max_age_bars: int = 100

    first_hit_only: bool = True

    def __post_init__(self) -> None:

        if self.target_mode not in {
            "near",
            "mid",
            "full",
        }:
            raise ValueError(
                "target_mode doit être near, mid ou full"
            )

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )


def detect_fvg_fill_targets(
    df: pd.DataFrame,
    params: FVGFillTargetParams = FVGFillTargetParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",

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
            "detect_fvg_fill_targets(): "
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

    out[
        "target_fvg_fill_bull"
    ] = np.nan

    out[
        "target_fvg_fill_bear"
    ] = np.nan

    out[
        "target_fvg_fill_bull_hit"
    ] = False

    out[
        "target_fvg_fill_bear_hit"
    ] = False

    out[
        "target_fvg_fill_bull_mode"
    ] = params.target_mode

    out[
        "target_fvg_fill_bear_mode"
    ] = params.target_mode

    for prefix in (
        "target_fvg_fill_bull",
        "target_fvg_fill_bear",
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
            f"{prefix}_hit_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_time_to_hit_bars"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    active_bull = []
    active_bear = []

    def bull_target(
        zone: dict,
    ) -> float:

        if params.target_mode == "near":
            return zone["high"]

        if params.target_mode == "mid":
            return zone["mid"]

        return zone["low"]

    def bear_target(
        zone: dict,
    ) -> float:

        if params.target_mode == "near":
            return zone["low"]

        if params.target_mode == "mid":
            return zone["mid"]

        return zone["high"]

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # EXISTING BULL FVG
        # Price retraces DOWN into bullish gap.
        # =====================================================================

        survivors = []

        for zone in active_bull:

            age = (
                i - zone["idx"]
            )

            if age > params.max_age_bars:
                continue

            target = bull_target(
                zone
            )

            hit = (
                low[i] <= target
            )

            if hit:

                out.at[
                    label,
                    "target_fvg_fill_bull",
                ] = target

                out.at[
                    label,
                    "target_fvg_fill_bull_hit",
                ] = True

                out.at[
                    label,
                    "target_fvg_fill_bull_source_idx",
                ] = zone["idx"]

                out.at[
                    label,
                    "target_fvg_fill_bull_hit_idx",
                ] = i

                out.at[
                    label,
                    "target_fvg_fill_bull_time_to_hit_bars",
                ] = age

                if not params.first_hit_only:
                    survivors.append(
                        zone
                    )

            else:

                survivors.append(
                    zone
                )

        active_bull = survivors

        # =====================================================================
        # EXISTING BEAR FVG
        # Price retraces UP into bearish gap.
        # =====================================================================

        survivors = []

        for zone in active_bear:

            age = (
                i - zone["idx"]
            )

            if age > params.max_age_bars:
                continue

            target = bear_target(
                zone
            )

            hit = (
                high[i] >= target
            )

            if hit:

                out.at[
                    label,
                    "target_fvg_fill_bear",
                ] = target

                out.at[
                    label,
                    "target_fvg_fill_bear_hit",
                ] = True

                out.at[
                    label,
                    "target_fvg_fill_bear_source_idx",
                ] = zone["idx"]

                out.at[
                    label,
                    "target_fvg_fill_bear_hit_idx",
                ] = i

                out.at[
                    label,
                    "target_fvg_fill_bear_time_to_hit_bars",
                ] = age

                if not params.first_hit_only:
                    survivors.append(
                        zone
                    )

            else:

                survivors.append(
                    zone
                )

        active_bear = survivors

        # =====================================================================
        # ADD NEW FVGs AFTER HIT CHECK
        # No same-candle retroactive first-passage.
        # =====================================================================

        row = out.iloc[i]

        if bool(
            row["fvg_bull"]
        ):

            active_bull.append(
                {
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

                    "mid":
                        float(
                            row[
                                "fvg_bear_mid"
                            ]
                        ),
                }
            )

    return out
