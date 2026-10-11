from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConsequentEncroachmentParams:
    """
    Consequent Encroachment (CE) v0.1.

    Ici CE = midpoint (50%) d'une FVG.

    IMPORTANT:
        Ce n'est PAS l'Equilibrium du Dealing Range.

        Equilibrium:
            midpoint du range.

        Consequent Encroachment:
            midpoint d'une imbalance / PD Array.

    max_age_bars:
        durée maximale pendant laquelle une FVG peut produire
        un CE touch.

    first_touch_only:
        émet uniquement le premier touch du CE.
    """

    max_age_bars: int = 100

    first_touch_only: bool = True

    def __post_init__(self) -> None:

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )


def detect_consequent_encroachment(
    df: pd.DataFrame,
    params: ConsequentEncroachmentParams = ConsequentEncroachmentParams(),
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
            "detect_consequent_encroachment(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["ce_bull"] = False
    out["ce_bear"] = False

    out["ce_bull_level"] = np.nan
    out["ce_bear_level"] = np.nan

    out["ce_bull_zone_low"] = np.nan
    out["ce_bull_zone_high"] = np.nan

    out["ce_bear_zone_low"] = np.nan
    out["ce_bear_zone_high"] = np.nan

    for prefix in (
        "ce_bull",
        "ce_bear",
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
            f"{prefix}_confirmation_idx"
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

    high = out["high"].to_numpy(
        dtype=np.float64,
    )

    low = out["low"].to_numpy(
        dtype=np.float64,
    )

    fvg_bull = out[
        "fvg_bull"
    ].to_numpy(dtype=bool)

    fvg_bear = out[
        "fvg_bear"
    ].to_numpy(dtype=bool)

    bull_low = out[
        "fvg_bull_low"
    ].to_numpy(dtype=np.float64)

    bull_high = out[
        "fvg_bull_high"
    ].to_numpy(dtype=np.float64)

    bull_mid = out[
        "fvg_bull_mid"
    ].to_numpy(dtype=np.float64)

    bear_low = out[
        "fvg_bear_low"
    ].to_numpy(dtype=np.float64)

    bear_high = out[
        "fvg_bear_high"
    ].to_numpy(dtype=np.float64)

    bear_mid = out[
        "fvg_bear_mid"
    ].to_numpy(dtype=np.float64)

    active_bull = []
    active_bear = []

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # EXISTING BULL FVG CE
        # =====================================================================

        surviving = []

        for zone in active_bull:

            age = (
                i
                - zone["source_idx"]
            )

            if age > params.max_age_bars:
                continue

            touched = (
                low[i] <= zone["mid"]
                and high[i] >= zone["mid"]
            )

            if touched:

                out.at[
                    label,
                    "ce_bull",
                ] = True

                out.at[
                    label,
                    "ce_bull_level",
                ] = zone["mid"]

                out.at[
                    label,
                    "ce_bull_zone_low",
                ] = zone["low"]

                out.at[
                    label,
                    "ce_bull_zone_high",
                ] = zone["high"]

                out.at[
                    label,
                    "ce_bull_source_fvg_idx",
                ] = zone[
                    "source_idx"
                ]

                out.at[
                    label,
                    "ce_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "ce_bull_age_bars",
                ] = age

                if not params.first_touch_only:

                    surviving.append(
                        zone
                    )

            else:

                surviving.append(
                    zone
                )

        active_bull = surviving

        # =====================================================================
        # EXISTING BEAR FVG CE
        # =====================================================================

        surviving = []

        for zone in active_bear:

            age = (
                i
                - zone["source_idx"]
            )

            if age > params.max_age_bars:
                continue

            touched = (
                low[i] <= zone["mid"]
                and high[i] >= zone["mid"]
            )

            if touched:

                out.at[
                    label,
                    "ce_bear",
                ] = True

                out.at[
                    label,
                    "ce_bear_level",
                ] = zone["mid"]

                out.at[
                    label,
                    "ce_bear_zone_low",
                ] = zone["low"]

                out.at[
                    label,
                    "ce_bear_zone_high",
                ] = zone["high"]

                out.at[
                    label,
                    "ce_bear_source_fvg_idx",
                ] = zone[
                    "source_idx"
                ]

                out.at[
                    label,
                    "ce_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "ce_bear_age_bars",
                ] = age

                if not params.first_touch_only:

                    surviving.append(
                        zone
                    )

            else:

                surviving.append(
                    zone
                )

        active_bear = surviving

        # =====================================================================
        # NEW FVGs
        #
        # Added AFTER processing current candle so CE cannot be touched
        # retroactively on its own creation candle.
        # =====================================================================

        if fvg_bull[i]:

            active_bull.append(
                {
                    "source_idx":
                        i,

                    "low":
                        bull_low[i],

                    "high":
                        bull_high[i],

                    "mid":
                        bull_mid[i],
                }
            )

        if fvg_bear[i]:

            active_bear.append(
                {
                    "source_idx":
                        i,

                    "low":
                        bear_low[i],

                    "high":
                        bear_high[i],

                    "mid":
                        bear_mid[i],
                }
            )

    return out
