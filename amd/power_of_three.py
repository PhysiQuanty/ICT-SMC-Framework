from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PowerOfThreeParams:
    """
    AMD / Power of Three v0.1.

    Bullish PO3:
        accumulation
        -> manipulation LOW
        -> distribution BULL.

    Bearish PO3:
        accumulation
        -> manipulation HIGH
        -> distribution BEAR.

    Cette feature ne recalcule aucune primitive.
    Elle relie uniquement manipulation + distribution.

    max_total_bars:
        délai maximum entre confirmation accumulation
        et confirmation distribution.
    """

    max_total_bars: int = 50

    def __post_init__(self) -> None:

        if self.max_total_bars < 1:
            raise ValueError(
                "max_total_bars doit être >= 1"
            )


def detect_power_of_three(
    df: pd.DataFrame,
    params: PowerOfThreeParams = PowerOfThreeParams(),
) -> pd.DataFrame:

    required = {
        "distribution_bull",
        "distribution_bear",

        "distribution_bull_manipulation_idx",
        "distribution_bear_manipulation_idx",

        "accumulation_confirmation_idx",
        "accumulation_high",
        "accumulation_low",
        "accumulation_mid",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_power_of_three(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["po3_bull"] = False
    out["po3_bear"] = False

    out["po3_high"] = np.nan
    out["po3_low"] = np.nan
    out["po3_mid"] = np.nan

    for side in (
        "po3_bull",
        "po3_bear",
    ):

        for suffix in (
            "accumulation_idx",
            "manipulation_idx",
            "confirmation_idx",
            "total_bars",
        ):

            out[
                f"{side}_{suffix}"
            ] = pd.Series(
                pd.array(
                    [pd.NA] * n,
                    dtype="Int64",
                ),
                index=out.index,
            )

    for i in range(n):

        row = out.iloc[i]
        label = out.index[i]

        # =====================================================================
        # BULL
        # =====================================================================

        if bool(
            row[
                "distribution_bull"
            ]
        ):

            manipulation_idx = row[
                "distribution_bull_manipulation_idx"
            ]

            accumulation_idx = row[
                "accumulation_confirmation_idx"
            ]

            if (
                pd.notna(manipulation_idx)
                and pd.notna(accumulation_idx)
            ):

                manipulation_idx = int(
                    manipulation_idx
                )

                accumulation_idx = int(
                    accumulation_idx
                )

                total_bars = (
                    i
                    - accumulation_idx
                )

                if (
                    0 <= total_bars
                    <= params.max_total_bars
                ):

                    out.at[
                        label,
                        "po3_bull",
                    ] = True

                    out.at[
                        label,
                        "po3_high",
                    ] = row[
                        "accumulation_high"
                    ]

                    out.at[
                        label,
                        "po3_low",
                    ] = row[
                        "accumulation_low"
                    ]

                    out.at[
                        label,
                        "po3_mid",
                    ] = row[
                        "accumulation_mid"
                    ]

                    out.at[
                        label,
                        "po3_bull_accumulation_idx",
                    ] = accumulation_idx

                    out.at[
                        label,
                        "po3_bull_manipulation_idx",
                    ] = manipulation_idx

                    out.at[
                        label,
                        "po3_bull_confirmation_idx",
                    ] = i

                    out.at[
                        label,
                        "po3_bull_total_bars",
                    ] = total_bars

        # =====================================================================
        # BEAR
        # =====================================================================

        if bool(
            row[
                "distribution_bear"
            ]
        ):

            manipulation_idx = row[
                "distribution_bear_manipulation_idx"
            ]

            accumulation_idx = row[
                "accumulation_confirmation_idx"
            ]

            if (
                pd.notna(manipulation_idx)
                and pd.notna(accumulation_idx)
            ):

                manipulation_idx = int(
                    manipulation_idx
                )

                accumulation_idx = int(
                    accumulation_idx
                )

                total_bars = (
                    i
                    - accumulation_idx
                )

                if (
                    0 <= total_bars
                    <= params.max_total_bars
                ):

                    out.at[
                        label,
                        "po3_bear",
                    ] = True

                    out.at[
                        label,
                        "po3_high",
                    ] = row[
                        "accumulation_high"
                    ]

                    out.at[
                        label,
                        "po3_low",
                    ] = row[
                        "accumulation_low"
                    ]

                    out.at[
                        label,
                        "po3_mid",
                    ] = row[
                        "accumulation_mid"
                    ]

                    out.at[
                        label,
                        "po3_bear_accumulation_idx",
                    ] = accumulation_idx

                    out.at[
                        label,
                        "po3_bear_manipulation_idx",
                    ] = manipulation_idx

                    out.at[
                        label,
                        "po3_bear_confirmation_idx",
                    ] = i

                    out.at[
                        label,
                        "po3_bear_total_bars",
                    ] = total_bars

    return out
