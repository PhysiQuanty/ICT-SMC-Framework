from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils import (
    local_datetime,
)


@dataclass(frozen=True)
class MidnightOpenParams:
    """
    Midnight Open v0.1.

    Default:
        minuit New York.

    Le niveau devient disponible uniquement à partir
    de la bougie dont l'heure locale est exactement 00:00.

    Aucun backfill sur les bougies précédentes.
    """

    timezone: str = "America/New_York"


def detect_midnight_open(
    df: pd.DataFrame,
    params: MidnightOpenParams = MidnightOpenParams(),
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
            "detect_midnight_open(): "
            f"colonnes manquantes: {sorted(missing)}"
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

    out[
        "midnight_open_event"
    ] = False

    out[
        "midnight_open"
    ] = np.nan

    out[
        "midnight_open_distance"
    ] = np.nan

    out[
        "midnight_open_distance_pct"
    ] = np.nan

    out[
        "above_midnight_open"
    ] = False

    out[
        "below_midnight_open"
    ] = False

    out[
        "midnight_open_idx"
    ] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out[
        "midnight_open_age_bars"
    ] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    local_date = (
        dt.dt.date.to_numpy()
    )

    hours = (
        dt.dt.hour.to_numpy()
    )

    minutes = (
        dt.dt.minute.to_numpy()
    )

    current_date = None
    current_open = None
    current_idx = None

    for i in range(n):

        date = local_date[i]

        if (
            current_date is None
            or date != current_date
        ):

            current_date = date
            current_open = None
            current_idx = None

        if (
            hours[i] == 0
            and minutes[i] == 0
        ):

            current_open = (
                open_[i]
            )

            current_idx = i

            out.at[
                out.index[i],
                "midnight_open_event",
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
            "midnight_open",
        ] = current_open

        out.at[
            label,
            "midnight_open_distance",
        ] = distance

        out.at[
            label,
            "midnight_open_distance_pct",
        ] = (
            distance
            / current_open
            if current_open != 0
            else np.nan
        )

        out.at[
            label,
            "above_midnight_open",
        ] = (
            close[i]
            > current_open
        )

        out.at[
            label,
            "below_midnight_open",
        ] = (
            close[i]
            < current_open
        )

        out.at[
            label,
            "midnight_open_idx",
        ] = current_idx

        out.at[
            label,
            "midnight_open_age_bars",
        ] = (
            i
            - current_idx
        )

    return out
