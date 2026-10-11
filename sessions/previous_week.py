from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from smc_framework.time._utils import local_datetime


@dataclass(frozen=True)
class PreviousWeekParams:
    """
    Previous Week High / Low v0.1.

    La semaine est définie par `week_start_day`:

        0 = lundi
        ...
        6 = dimanche

    Default:
        lundi, timezone UTC.

    Le PWH/PWL n'est exposé qu'une fois la semaine
    précédente entièrement terminée.
    """

    timezone: str = "UTC"
    week_start_day: int = 0

    def __post_init__(self) -> None:

        if not 0 <= self.week_start_day <= 6:
            raise ValueError(
                "week_start_day doit être entre 0 et 6"
            )


def _week_key(
    dt: pd.Series,
    week_start_day: int,
) -> np.ndarray:

    dates = dt.dt.date.to_numpy()

    weekdays = (
        dt.dt.weekday.to_numpy()
    )

    keys = []

    for date, weekday in zip(
        dates,
        weekdays,
    ):

        distance = (
            int(weekday)
            - week_start_day
        ) % 7

        keys.append(
            date
            - timedelta(
                days=distance
            )
        )

    return np.asarray(
        keys,
        dtype=object,
    )


def detect_previous_week_levels(
    df: pd.DataFrame,
    params: PreviousWeekParams = PreviousWeekParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_previous_week_levels(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    dt = local_datetime(
        out,
        params.timezone,
    )

    week_key = _week_key(
        dt,
        params.week_start_day,
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

    # =========================================================================
    # OUTPUTS
    # =========================================================================

    out["previous_week_high"] = np.nan
    out["previous_week_low"] = np.nan
    out["previous_week_mid"] = np.nan
    out["previous_week_range"] = np.nan

    out["previous_week_high_distance"] = np.nan
    out["previous_week_low_distance"] = np.nan

    out["above_previous_week_high"] = False
    out["below_previous_week_low"] = False

    out["previous_week_high_taken"] = False
    out["previous_week_low_taken"] = False

    out["previous_week_new_levels"] = False

    out["previous_week_source_start_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["previous_week_source_end_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    # =========================================================================
    # STATE
    # =========================================================================

    current_key = None

    current_high = None
    current_low = None

    current_start_idx = None

    pwh = None
    pwl = None

    previous_start_idx = None
    previous_end_idx = None

    pwh_taken = False
    pwl_taken = False

    for i in range(n):

        key = week_key[i]

        if (
            current_key is None
            or key != current_key
        ):

            if current_key is not None:

                pwh = current_high
                pwl = current_low

                previous_start_idx = (
                    current_start_idx
                )

                previous_end_idx = (
                    i - 1
                )

                pwh_taken = False
                pwl_taken = False

                out.at[
                    out.index[i],
                    "previous_week_new_levels",
                ] = True

            current_key = key

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

        if (
            pwh is None
            or pwl is None
        ):
            continue

        label = out.index[i]

        out.at[
            label,
            "previous_week_high",
        ] = pwh

        out.at[
            label,
            "previous_week_low",
        ] = pwl

        out.at[
            label,
            "previous_week_mid",
        ] = (
            pwh + pwl
        ) / 2.0

        out.at[
            label,
            "previous_week_range",
        ] = (
            pwh - pwl
        )

        out.at[
            label,
            "previous_week_high_distance",
        ] = (
            close[i] - pwh
        )

        out.at[
            label,
            "previous_week_low_distance",
        ] = (
            close[i] - pwl
        )

        out.at[
            label,
            "above_previous_week_high",
        ] = (
            close[i] > pwh
        )

        out.at[
            label,
            "below_previous_week_low",
        ] = (
            close[i] < pwl
        )

        if (
            not pwh_taken
            and high[i] > pwh
        ):

            pwh_taken = True

            out.at[
                label,
                "previous_week_high_taken",
            ] = True

        if (
            not pwl_taken
            and low[i] < pwl
        ):

            pwl_taken = True

            out.at[
                label,
                "previous_week_low_taken",
            ] = True

        out.at[
            label,
            "previous_week_source_start_idx",
        ] = previous_start_idx

        out.at[
            label,
            "previous_week_source_end_idx",
        ] = previous_end_idx

    return out
