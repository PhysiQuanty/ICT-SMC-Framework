from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smc_framework.time._utils import local_datetime


@dataclass(frozen=True)
class PreviousDayParams:
    """
    Previous Day High / Low v0.1.

    Les journées sont définies dans `timezone`.

    Le PDH/PDL du jour D n'est disponible qu'à partir
    de la première bougie observée du jour D, après que D-1
    est entièrement terminé.

    Aucun backfill.
    """

    timezone: str = "UTC"


def detect_previous_day_levels(
    df: pd.DataFrame,
    params: PreviousDayParams = PreviousDayParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "detect_previous_day_levels(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    dt = local_datetime(
        out,
        params.timezone,
    )

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

    dates = dt.dt.date.to_numpy()

    # =========================================================================
    # OUTPUTS
    # =========================================================================

    out["previous_day_high"] = np.nan
    out["previous_day_low"] = np.nan
    out["previous_day_mid"] = np.nan
    out["previous_day_range"] = np.nan

    out["previous_day_high_distance"] = np.nan
    out["previous_day_low_distance"] = np.nan

    out["above_previous_day_high"] = False
    out["below_previous_day_low"] = False

    out["previous_day_high_taken"] = False
    out["previous_day_low_taken"] = False

    out["previous_day_new_levels"] = False

    out["previous_day_source_start_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["previous_day_source_end_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    # =========================================================================
    # STATE
    # =========================================================================

    current_date = None

    current_high = None
    current_low = None

    current_start_idx = None

    pdh = None
    pdl = None

    previous_start_idx = None
    previous_end_idx = None

    pdh_taken = False
    pdl_taken = False

    for i in range(n):

        date = dates[i]

        # =====================================================================
        # NEW DAY
        # =====================================================================

        if (
            current_date is None
            or date != current_date
        ):

            # Finalize previous day.
            if current_date is not None:

                pdh = current_high
                pdl = current_low

                previous_start_idx = (
                    current_start_idx
                )

                previous_end_idx = (
                    i - 1
                )

                pdh_taken = False
                pdl_taken = False

                out.at[
                    out.index[i],
                    "previous_day_new_levels",
                ] = True

            current_date = date

            current_high = high[i]
            current_low = low[i]

            current_start_idx = i

        else:

            current_high = max(
                current_high,
                high[i],
            )

            current_low = min(
                current_low,
                low[i],
            )

        # =====================================================================
        # PREVIOUS-DAY LEVELS
        # =====================================================================

        if (
            pdh is None
            or pdl is None
        ):
            continue

        label = out.index[i]

        mid = (
            pdh + pdl
        ) / 2.0

        out.at[
            label,
            "previous_day_high",
        ] = pdh

        out.at[
            label,
            "previous_day_low",
        ] = pdl

        out.at[
            label,
            "previous_day_mid",
        ] = mid

        out.at[
            label,
            "previous_day_range",
        ] = (
            pdh - pdl
        )

        out.at[
            label,
            "previous_day_high_distance",
        ] = (
            close[i] - pdh
        )

        out.at[
            label,
            "previous_day_low_distance",
        ] = (
            close[i] - pdl
        )

        out.at[
            label,
            "above_previous_day_high",
        ] = (
            close[i] > pdh
        )

        out.at[
            label,
            "below_previous_day_low",
        ] = (
            close[i] < pdl
        )

        # First wick penetration.
        if (
            not pdh_taken
            and high[i] > pdh
        ):

            pdh_taken = True

            out.at[
                label,
                "previous_day_high_taken",
            ] = True

        if (
            not pdl_taken
            and low[i] < pdl
        ):

            pdl_taken = True

            out.at[
                label,
                "previous_day_low_taken",
            ] = True

        out.at[
            label,
            "previous_day_source_start_idx",
        ] = previous_start_idx

        out.at[
            label,
            "previous_day_source_end_idx",
        ] = previous_end_idx

    return out
