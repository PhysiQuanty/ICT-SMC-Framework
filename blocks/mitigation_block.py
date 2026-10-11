from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MitigationBlockParams:
    """
    Mitigation Block v0.1.

    Proxy:
        premier retour causal dans un Order Block après confirmation,
        avant invalidation.

    touch_mode:
        "zone" -> premier contact avec zone
        "mid"  -> midpoint atteint

    min_age_bars:
        délai minimal après confirmation de l'OB.
    """

    touch_mode: str = "zone"
    min_age_bars: int = 1

    def __post_init__(self) -> None:

        if self.touch_mode not in {
            "zone",
            "mid",
        }:
            raise ValueError(
                "touch_mode doit être 'zone' ou 'mid'"
            )

        if self.min_age_bars < 1:
            raise ValueError(
                "min_age_bars doit être >= 1"
            )


def detect_mitigation_blocks(
    df: pd.DataFrame,
    params: MitigationBlockParams = MitigationBlockParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",

        "ob_bull",
        "ob_bear",

        "ob_bull_low",
        "ob_bull_high",
        "ob_bull_mid",

        "ob_bear_low",
        "ob_bear_high",
        "ob_bear_mid",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_mitigation_blocks(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    for side in (
        "mitigation_bull",
        "mitigation_bear",
    ):

        out[side] = False

        out[f"{side}_low"] = np.nan
        out[f"{side}_high"] = np.nan
        out[f"{side}_mid"] = np.nan

        out[f"{side}_penetration_ratio"] = np.nan

        for suffix in (
            "source_ob_idx",
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

    bull_zones = []
    bear_zones = []

    for i in range(n):

        # =====================================================================
        # EXISTING BULL OB RETESTS
        # =====================================================================

        remaining = []

        for zone in bull_zones:

            age = (
                i
                - zone["source_idx"]
            )

            if age < params.min_age_bars:
                remaining.append(
                    zone
                )
                continue

            if params.touch_mode == "zone":

                touched = (
                    low[i] <= zone["high"]
                    and high[i] >= zone["low"]
                )

            else:

                touched = (
                    low[i] <= zone["mid"]
                )

            if touched:

                size = (
                    zone["high"]
                    - zone["low"]
                )

                penetration = (
                    zone["high"]
                    - max(
                        low[i],
                        zone["low"],
                    )
                )

                ratio = (
                    np.clip(
                        penetration / size,
                        0.0,
                        1.0,
                    )
                    if size > 0
                    else np.nan
                )

                label = out.index[i]

                out.at[
                    label,
                    "mitigation_bull",
                ] = True

                out.at[
                    label,
                    "mitigation_bull_low",
                ] = zone["low"]

                out.at[
                    label,
                    "mitigation_bull_high",
                ] = zone["high"]

                out.at[
                    label,
                    "mitigation_bull_mid",
                ] = zone["mid"]

                out.at[
                    label,
                    "mitigation_bull_penetration_ratio",
                ] = ratio

                out.at[
                    label,
                    "mitigation_bull_source_ob_idx",
                ] = zone[
                    "source_idx"
                ]

                out.at[
                    label,
                    "mitigation_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "mitigation_bull_age_bars",
                ] = age

            else:

                remaining.append(
                    zone
                )

        bull_zones = remaining

        # =====================================================================
        # EXISTING BEAR OB RETESTS
        # =====================================================================

        remaining = []

        for zone in bear_zones:

            age = (
                i
                - zone["source_idx"]
            )

            if age < params.min_age_bars:
                remaining.append(
                    zone
                )
                continue

            if params.touch_mode == "zone":

                touched = (
                    high[i] >= zone["low"]
                    and low[i] <= zone["high"]
                )

            else:

                touched = (
                    high[i] >= zone["mid"]
                )

            if touched:

                size = (
                    zone["high"]
                    - zone["low"]
                )

                penetration = (
                    min(
                        high[i],
                        zone["high"],
                    )
                    - zone["low"]
                )

                ratio = (
                    np.clip(
                        penetration / size,
                        0.0,
                        1.0,
                    )
                    if size > 0
                    else np.nan
                )

                label = out.index[i]

                out.at[
                    label,
                    "mitigation_bear",
                ] = True

                out.at[
                    label,
                    "mitigation_bear_low",
                ] = zone["low"]

                out.at[
                    label,
                    "mitigation_bear_high",
                ] = zone["high"]

                out.at[
                    label,
                    "mitigation_bear_mid",
                ] = zone["mid"]

                out.at[
                    label,
                    "mitigation_bear_penetration_ratio",
                ] = ratio

                out.at[
                    label,
                    "mitigation_bear_source_ob_idx",
                ] = zone[
                    "source_idx"
                ]

                out.at[
                    label,
                    "mitigation_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "mitigation_bear_age_bars",
                ] = age

            else:

                remaining.append(
                    zone
                )

        bear_zones = remaining

        # =====================================================================
        # ADD NEW OBs AFTER PROCESSING CURRENT CANDLE
        # avoids same-candle "mitigation".
        # =====================================================================

        row = out.iloc[i]

        if bool(
            row["ob_bull"]
        ):

            bull_zones.append(
                {
                    "source_idx":
                        i,

                    "low":
                        float(
                            row[
                                "ob_bull_low"
                            ]
                        ),

                    "high":
                        float(
                            row[
                                "ob_bull_high"
                            ]
                        ),

                    "mid":
                        float(
                            row[
                                "ob_bull_mid"
                            ]
                        ),
                }
            )

        if bool(
            row["ob_bear"]
        ):

            bear_zones.append(
                {
                    "source_idx":
                        i,

                    "low":
                        float(
                            row[
                                "ob_bear_low"
                            ]
                        ),

                    "high":
                        float(
                            row[
                                "ob_bear_high"
                            ]
                        ),

                    "mid":
                        float(
                            row[
                                "ob_bear_mid"
                            ]
                        ),
                }
            )

    return out
