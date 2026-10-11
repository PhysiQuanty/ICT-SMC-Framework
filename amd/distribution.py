from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistributionParams:
    """
    Distribution v0.1.

    Après manipulation:

        manipulation_low
            -> attente d'une expansion bullish.

        manipulation_high
            -> attente d'une expansion bearish.

    Confirmation par défaut:
        displacement de direction opposée à la manipulation.

    require_range_break:
        si True:
            bullish distribution doit clôturer au-dessus
            de l'accumulation high.

            bearish distribution doit clôturer sous
            accumulation low.

    max_lag:
        nombre maximal de bars après manipulation.
    """

    max_lag: int = 10

    require_displacement: bool = True
    require_range_break: bool = False

    def __post_init__(self) -> None:

        if self.max_lag < 0:
            raise ValueError(
                "max_lag doit être >= 0"
            )


def detect_distribution(
    df: pd.DataFrame,
    params: DistributionParams = DistributionParams(),
) -> pd.DataFrame:

    required = {
        "close",

        "manipulation_high",
        "manipulation_low",

        "accumulation_high",
        "accumulation_low",
    }

    if params.require_displacement:

        required.update(
            {
                "displacement_bull",
                "displacement_bear",
            }
        )

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_distribution(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    close = out[
        "close"
    ].to_numpy(dtype=np.float64)

    manipulation_high = out[
        "manipulation_high"
    ].to_numpy(dtype=bool)

    manipulation_low = out[
        "manipulation_low"
    ].to_numpy(dtype=bool)

    acc_high = out[
        "accumulation_high"
    ].to_numpy(dtype=np.float64)

    acc_low = out[
        "accumulation_low"
    ].to_numpy(dtype=np.float64)

    if params.require_displacement:

        disp_bull = out[
            "displacement_bull"
        ].to_numpy(dtype=bool)

        disp_bear = out[
            "displacement_bear"
        ].to_numpy(dtype=bool)

    else:

        disp_bull = np.ones(
            n,
            dtype=bool,
        )

        disp_bear = np.ones(
            n,
            dtype=bool,
        )

    out["distribution_bull"] = False
    out["distribution_bear"] = False

    out["distribution_bull_range_break"] = False
    out["distribution_bear_range_break"] = False

    for side in (
        "distribution_bull",
        "distribution_bear",
    ):

        for suffix in (
            "manipulation_idx",
            "confirmation_idx",
            "lag",
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

    pending_bull = None
    pending_bear = None

    for i in range(n):

        # =====================================================================
        # NEW MANIPULATIONS
        # =====================================================================

        if manipulation_low[i]:

            pending_bull = {
                "idx":
                    i,

                "high":
                    acc_high[i],

                "low":
                    acc_low[i],
            }

        if manipulation_high[i]:

            pending_bear = {
                "idx":
                    i,

                "high":
                    acc_high[i],

                "low":
                    acc_low[i],
            }

        # =====================================================================
        # EXPIRE
        # =====================================================================

        if (
            pending_bull is not None
            and (
                i - pending_bull["idx"]
                > params.max_lag
            )
        ):

            pending_bull = None

        if (
            pending_bear is not None
            and (
                i - pending_bear["idx"]
                > params.max_lag
            )
        ):

            pending_bear = None

        # =====================================================================
        # BULL DISTRIBUTION
        # =====================================================================

        if (
            pending_bull is not None
            and i > pending_bull["idx"]
        ):

            range_break = (
                close[i]
                > pending_bull[
                    "high"
                ]
            )

            confirm = (
                disp_bull[i]
                and (
                    not params.require_range_break
                    or range_break
                )
            )

            if confirm:

                label = out.index[i]

                out.at[
                    label,
                    "distribution_bull",
                ] = True

                out.at[
                    label,
                    "distribution_bull_range_break",
                ] = range_break

                out.at[
                    label,
                    "distribution_bull_manipulation_idx",
                ] = pending_bull[
                    "idx"
                ]

                out.at[
                    label,
                    "distribution_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "distribution_bull_lag",
                ] = (
                    i
                    - pending_bull[
                        "idx"
                    ]
                )

                pending_bull = None

        # =====================================================================
        # BEAR DISTRIBUTION
        # =====================================================================

        if (
            pending_bear is not None
            and i > pending_bear["idx"]
        ):

            range_break = (
                close[i]
                < pending_bear[
                    "low"
                ]
            )

            confirm = (
                disp_bear[i]
                and (
                    not params.require_range_break
                    or range_break
                )
            )

            if confirm:

                label = out.index[i]

                out.at[
                    label,
                    "distribution_bear",
                ] = True

                out.at[
                    label,
                    "distribution_bear_range_break",
                ] = range_break

                out.at[
                    label,
                    "distribution_bear_manipulation_idx",
                ] = pending_bear[
                    "idx"
                ]

                out.at[
                    label,
                    "distribution_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "distribution_bear_lag",
                ] = (
                    i
                    - pending_bear[
                        "idx"
                    ]
                )

                pending_bear = None

    return out
