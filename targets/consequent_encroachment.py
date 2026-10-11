from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CETargetParams:
    """
    Consequent Encroachment Target v0.1.

    CE = midpoint de la FVG.

    Cette feature est volontairement séparée de
    ranges/consequent_encroachment.py:

        ranges/...
            décrit l'événement de touch du CE.

        targets/...
            décrit le CE comme cible et mesure explicitement
            son first-passage depuis la création de la FVG.

    max_age_bars:
        horizon maximal.

    first_hit_only:
        premier passage uniquement.
    """

    max_age_bars: int = 100

    first_hit_only: bool = True

    def __post_init__(self) -> None:

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )


def detect_ce_targets(
    df: pd.DataFrame,
    params: CETargetParams = CETargetParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",

        "fvg_bull",
        "fvg_bear",

        "fvg_bull_mid",
        "fvg_bear_mid",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_ce_targets(): "
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
        "target_ce_bull"
    ] = np.nan

    out[
        "target_ce_bear"
    ] = np.nan

    out[
        "target_ce_bull_hit"
    ] = False

    out[
        "target_ce_bear_hit"
    ] = False

    for prefix in (
        "target_ce_bull",
        "target_ce_bear",
    ):

        out[
            f"{prefix}_source_fvg_idx"
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

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # BULL FVG -> CE BELOW/INSIDE GAP
        # =====================================================================

        survivors = []

        for target in active_bull:

            age = (
                i - target["idx"]
            )

            if age > params.max_age_bars:
                continue

            hit = (
                low[i] <= target["level"]
            )

            if hit:

                out.at[
                    label,
                    "target_ce_bull",
                ] = target["level"]

                out.at[
                    label,
                    "target_ce_bull_hit",
                ] = True

                out.at[
                    label,
                    "target_ce_bull_source_fvg_idx",
                ] = target["idx"]

                out.at[
                    label,
                    "target_ce_bull_hit_idx",
                ] = i

                out.at[
                    label,
                    "target_ce_bull_time_to_hit_bars",
                ] = age

                if not params.first_hit_only:
                    survivors.append(
                        target
                    )

            else:

                survivors.append(
                    target
                )

        active_bull = survivors

        # =====================================================================
        # BEAR FVG -> CE ABOVE/INSIDE GAP
        # =====================================================================

        survivors = []

        for target in active_bear:

            age = (
                i - target["idx"]
            )

            if age > params.max_age_bars:
                continue

            hit = (
                high[i] >= target["level"]
            )

            if hit:

                out.at[
                    label,
                    "target_ce_bear",
                ] = target["level"]

                out.at[
                    label,
                    "target_ce_bear_hit",
                ] = True

                out.at[
                    label,
                    "target_ce_bear_source_fvg_idx",
                ] = target["idx"]

                out.at[
                    label,
                    "target_ce_bear_hit_idx",
                ] = i

                out.at[
                    label,
                    "target_ce_bear_time_to_hit_bars",
                ] = age

                if not params.first_hit_only:
                    survivors.append(
                        target
                    )

            else:

                survivors.append(
                    target
                )

        active_bear = survivors

        # =====================================================================
        # NEW TARGETS AFTER CURRENT CANDLE
        # =====================================================================

        row = out.iloc[i]

        if bool(
            row["fvg_bull"]
        ):

            level = row[
                "fvg_bull_mid"
            ]

            if pd.notna(level):

                active_bull.append(
                    {
                        "idx":
                            i,

                        "level":
                            float(level),
                    }
                )

        if bool(
            row["fvg_bear"]
        ):

            level = row[
                "fvg_bear_mid"
            ]

            if pd.notna(level):

                active_bear.append(
                    {
                        "idx":
                            i,

                        "level":
                            float(level),
                    }
                )

    return out
