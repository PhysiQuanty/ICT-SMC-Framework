from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


TIMESTAMP_CANDIDATES = (
    "datetime",
    "timestamp",
    "open_time",
    "date",
    "time",
)


def get_datetime_utc(
    df: pd.DataFrame,
) -> pd.Series:

    column = None

    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in df.columns:
            column = candidate
            break

    if column is None:
        raise ValueError(
            "Aucune colonne temporelle trouvée. "
            f"Attendu: {TIMESTAMP_CANDIDATES}"
        )

    s = df[column]

    # Binance timestamp milliseconds.
    if pd.api.types.is_numeric_dtype(s):

        numeric = pd.to_numeric(
            s,
            errors="coerce",
        )

        finite = numeric.dropna()

        if len(finite) == 0:
            raise ValueError(
                f"Colonne temporelle vide: {column}"
            )

        median = float(
            finite.abs().median()
        )

        if median > 1e14:
            unit = "ns"
        elif median > 1e11:
            unit = "ms"
        elif median > 1e9:
            unit = "s"
        else:
            unit = "s"

        dt = pd.to_datetime(
            numeric,
            unit=unit,
            utc=True,
            errors="coerce",
        )

    else:

        dt = pd.to_datetime(
            s,
            utc=True,
            errors="coerce",
        )

    if dt.isna().any():

        bad = int(
            dt.isna().sum()
        )

        raise ValueError(
            f"{bad} timestamps invalides dans {column}"
        )

    return dt


def local_datetime(
    df: pd.DataFrame,
    timezone: str,
) -> pd.Series:

    # Raises explicitly on invalid TZ.
    ZoneInfo(
        timezone
    )

    dt = get_datetime_utc(
        df
    )

    return dt.dt.tz_convert(
        timezone
    )


def minute_of_day(
    dt: pd.Series,
) -> np.ndarray:

    return (
        dt.dt.hour.to_numpy(
            dtype=np.int32
        )
        * 60
        + dt.dt.minute.to_numpy(
            dtype=np.int32
        )
    )


def parse_hhmm(
    value: str,
) -> int:

    try:
        hour_str, minute_str = (
            value.split(":")
        )

        hour = int(
            hour_str
        )

        minute = int(
            minute_str
        )

    except Exception as exc:

        raise ValueError(
            f"Heure invalide: {value!r}. "
            "Format attendu HH:MM"
        ) from exc

    if not 0 <= hour <= 23:
        raise ValueError(
            f"Heure invalide: {value}"
        )

    if not 0 <= minute <= 59:
        raise ValueError(
            f"Minute invalide: {value}"
        )

    return (
        hour * 60
        + minute
    )


def in_clock_window(
    minute_values: np.ndarray,
    start: str,
    end: str,
) -> np.ndarray:
    """
    Fenêtre [start, end).

    Supporte passage à minuit.

    Ex:
        20:00 -> 00:00
    """

    start_min = parse_hhmm(
        start
    )

    end_min = parse_hhmm(
        end
    )

    if start_min == end_min:

        # 24h window.
        return np.ones(
            len(minute_values),
            dtype=bool,
        )

    if start_min < end_min:

        return (
            (minute_values >= start_min)
            & (minute_values < end_min)
        )

    return (
        (minute_values >= start_min)
        | (minute_values < end_min)
    )
