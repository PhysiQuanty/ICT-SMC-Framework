from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils import (
    local_datetime,
    parse_hhmm,
)


@dataclass(frozen=True)
class DailyOpenParams:
    """
    Daily Open générique.

    Default:
        00:00 UTC.

    Cela le distingue du Midnight Open ICT,
    dont le default est 00:00 America/New_York.

    anchor_time et timezone sont entièrement paramétrables.
    """

    timezone: str = "UTC"
    anchor_time: str = "00:00"


def detect_daily_open(
    df: pd.DataFrame,
    params: DailyOpenParams = DailyOpenParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "close",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_daily_open(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    anchor_minute = parse_hhmm(
        params.anchor_time
    )

    out = df.copy()
    n = len(out)

    dt = local_datetime(
        out,
        params.timezone,
    )

    open_ = out[
        "open"
    ].to_numpy(
        dtype=np.float64,
    )

    close = out[
        "close"
    ].to_numpy(
        dtype=np.float64,
    )

    minutes = (
        dt.dt.hour.to_numpy()
        * 60
        + dt.dt.minute.to_numpy()
    )

    dates = (
        dt.dt.date.to_numpy()
    )

    out[
        "daily_open_event"
    ] = False

    out[
        "daily_open"
    ] = np.nan

    out[
        "daily_open_distance"
    ] = np.nan

    out[
        "daily_open_distance_pct"
    ] = np.nan

    out[
        "above_daily_open"
    ] = False

    out[
        "below_daily_open"
    ] = False

    out[
        "daily_open_idx"
    ] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out[
        "daily_open_age_bars"
    ] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    current_date = None
    current_open = None
    current_idx = None

    for i in range(n):

        if (
            current_date is None
            or dates[i] != current_date
        ):

            current_date = dates[i]
            current_open = None
            current_idx = None

        if minutes[i] == anchor_minute:

            current_open = open_[i]
            current_idx = i

            out.at[
                out.index[i],
                "daily_open_event",
            ] = True

        if current_open is None:
            continue

        label = out.index[i]

        distance = (
            close[i]
            - current_open
        )

        out.at[
            label,
            "daily_open",
        ] = current_open

        out.at[
            label,
            "daily_open_distance",
        ] = distance

        out.at[
            label,
            "daily_open_distance_pct",
        ] = (
            distance
            / current_open
            if current_open != 0
            else np.nan
        )

        out.at[
            label,
            "above_daily_open",
        ] = (
            close[i] > current_open
        )

        out.at[
            label,
            "below_daily_open",
        ] = (
            close[i] < current_open
        )

        out.at[
            label,
            "daily_open_idx",
        ] = current_idx

        out.at[
            label,
            "daily_open_age_bars",
        ] = (
            i - current_idx
        )

    return out
