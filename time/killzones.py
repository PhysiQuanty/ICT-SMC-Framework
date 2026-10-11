from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils import (
    in_clock_window,
    local_datetime,
    minute_of_day,
)


@dataclass(frozen=True)
class KillzoneParams:
    """
    Killzones v0.1.

    Toutes les fenêtres sont définies dans timezone.

    Defaults = convention de recherche couramment utilisée
    sur l'horloge New York.

    Elles sont volontairement paramétrables pour calibration.
    """

    timezone: str = "America/New_York"

    asia_start: str = "20:00"
    asia_end: str = "00:00"

    london_start: str = "02:00"
    london_end: str = "05:00"

    new_york_start: str = "07:00"
    new_york_end: str = "10:00"

    london_close_start: str = "10:00"
    london_close_end: str = "12:00"

    weekdays_only: bool = False


def detect_killzones(
    df: pd.DataFrame,
    params: KillzoneParams = KillzoneParams(),
) -> pd.DataFrame:

    out = df.copy()

    dt = local_datetime(
        out,
        params.timezone,
    )

    minutes = minute_of_day(
        dt
    )

    asia = in_clock_window(
        minutes,
        params.asia_start,
        params.asia_end,
    )

    london = in_clock_window(
        minutes,
        params.london_start,
        params.london_end,
    )

    ny = in_clock_window(
        minutes,
        params.new_york_start,
        params.new_york_end,
    )

    london_close = in_clock_window(
        minutes,
        params.london_close_start,
        params.london_close_end,
    )

    if params.weekdays_only:

        weekday = (
            dt.dt.weekday.to_numpy()
        )

        valid_day = (
            weekday < 5
        )

        asia &= valid_day
        london &= valid_day
        ny &= valid_day
        london_close &= valid_day

    out[
        "killzone_asia"
    ] = asia

    out[
        "killzone_london"
    ] = london

    out[
        "killzone_new_york"
    ] = ny

    out[
        "killzone_london_close"
    ] = london_close

    out[
        "killzone_active"
    ] = (
        asia
        | london
        | ny
        | london_close
    )

    names = np.full(
        len(out),
        None,
        dtype=object,
    )

    # Multiple windows can overlap.
    for i in range(
        len(out)
    ):

        active = []

        if asia[i]:
            active.append(
                "asia"
            )

        if london[i]:
            active.append(
                "london"
            )

        if ny[i]:
            active.append(
                "new_york"
            )

        if london_close[i]:
            active.append(
                "london_close"
            )

        if active:

            names[i] = "|".join(
                active
            )

    out[
        "killzone_name"
    ] = names

    out[
        "time_local"
    ] = dt

    out[
        "time_local_hour"
    ] = (
        dt.dt.hour
        + dt.dt.minute / 60.0
    )

    out[
        "time_local_weekday"
    ] = dt.dt.weekday

    return out
