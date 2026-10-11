from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._utils import (
    in_clock_window,
    local_datetime,
    minute_of_day,
)


DEFAULT_MACRO_WINDOWS = (
    (
        "london_1",
        "02:33",
        "03:00",
    ),
    (
        "london_2",
        "04:03",
        "04:30",
    ),
    (
        "ny_pre_market",
        "08:50",
        "09:10",
    ),
    (
        "ny_am_1",
        "09:50",
        "10:10",
    ),
    (
        "ny_am_2",
        "10:50",
        "11:10",
    ),
    (
        "ny_lunch",
        "11:50",
        "12:10",
    ),
    (
        "ny_pm",
        "13:10",
        "13:40",
    ),
    (
        "ny_last_hour",
        "15:15",
        "15:45",
    ),
)


@dataclass(frozen=True)
class MacroParams:
    """
    Macro windows v0.1.

    La grille par défaut est une convention ICT couramment citée,
    ancrée sur l'heure New York.

    Ce ne sont PAS des vérités imposées au framework.

    windows:
        tuple de:
            (nom, HH:MM début, HH:MM fin)

    Exemple custom:
        windows=(
            ("macro_a", "09:45", "10:15"),
            ("macro_b", "13:00", "13:30"),
        )
    """

    timezone: str = "America/New_York"

    windows: tuple[
        tuple[str, str, str],
        ...
    ] = DEFAULT_MACRO_WINDOWS

    weekdays_only: bool = False


def detect_macros(
    df: pd.DataFrame,
    params: MacroParams = MacroParams(),
) -> pd.DataFrame:

    out = df.copy()

    dt = local_datetime(
        out,
        params.timezone,
    )

    minutes = minute_of_day(
        dt
    )

    n = len(out)

    active_any = np.zeros(
        n,
        dtype=bool,
    )

    count = np.zeros(
        n,
        dtype=np.int32,
    )

    names = [
        []
        for _ in range(n)
    ]

    weekday = (
        dt.dt.weekday.to_numpy()
    )

    valid_day = (
        weekday < 5
    )

    for (
        name,
        start,
        end,
    ) in params.windows:

        safe_name = (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        mask = in_clock_window(
            minutes,
            start,
            end,
        )

        if params.weekdays_only:

            mask &= valid_day

        column = (
            f"macro_{safe_name}"
        )

        out[
            column
        ] = mask

        active_any |= mask
        count += mask.astype(
            np.int32
        )

        indices = np.flatnonzero(
            mask
        )

        for i in indices:

            names[i].append(
                safe_name
            )

    out[
        "macro_active"
    ] = active_any

    out[
        "macro_count"
    ] = count

    out[
        "macro_name"
    ] = [
        "|".join(x)
        if x
        else None
        for x in names
    ]

    return out
