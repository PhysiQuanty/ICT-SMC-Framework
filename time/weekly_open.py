from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils import (
    local_datetime,
    parse_hhmm,
)


@dataclass(frozen=True)
class WeeklyOpenParams:
    """
    Weekly Open générique.

    Default objectif pour marchés 24/7:
        lundi 00:00 UTC.

    weekday:
        lundi = 0
        ...
        dimanche = 6

    Pour FX / autres conventions, tout est configurable.
    """

    timezone: str = "UTC"

    weekday: int = 0

    anchor_time: str = "00:00"

    def __post_init__(self) -> None:

        if not 0 <= self.weekday <= 6:
            raise ValueError(
                "weekday doit être entre 0 et 6"
            )


def detect_weekly_open(
    df: pd.DataFrame,
    params: WeeklyOpenParams = WeeklyOpenParams(),
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
            "detect_weekly_open(): "
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

    weekday = (
        dt.dt.weekday.to_numpy()
    )

    minutes = (
        dt.dt.hour.to_numpy()
        * 60
        + dt.dt.minute.to_numpy()
    )

    out[
        "weekly_open_event"
    ] = False

    out[
        "weekly_open"
    ] = np.nan

    out[
        "weekly_open_distance"
    ] = np.nan

    out[
        "weekly_open_distance_pct"
    ] = np.nan

    out[
        "above_weekly_open"
    ] = False

    out[
        "below_weekly_open"
    ] = False

    out[
        "weekly_open_idx"
    ] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out[
        "weekly_open_age_bars"
    ] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    current_open = None
    current_idx = None

    for i in range(n):

        if (
            weekday[i]
            == params.weekday
            and minutes[i]
            == anchor_minute
        ):

            current_open = (
                open_[i]
            )

            current_idx = i

            out.at[
                out.index[i],
                "weekly_open_event",
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
            "weekly_open",
        ] = current_open

        out.at[
            label,
            "weekly_open_distance",
        ] = distance

        out.at[
            label,
            "weekly_open_distance_pct",
        ] = (
            distance
            / current_open
            if current_open != 0
            else np.nan
        )

        out.at[
            label,
            "above_weekly_open",
        ] = (
            close[i] > current_open
        )

        out.at[
            label,
            "below_weekly_open",
        ] = (
            close[i] < current_open
        )

        out.at[
            label,
            "weekly_open_idx",
        ] = current_idx

        out.at[
            label,
            "weekly_open_age_bars",
        ] = (
            i - current_idx
        )

    return out
