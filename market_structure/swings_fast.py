from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwingParams:
    left: int = 3
    right: int = 3
    strict: bool = True

    def __post_init__(self) -> None:
        if self.left < 1:
            raise ValueError("SwingParams.left doit être >= 1")
        if self.right < 1:
            raise ValueError("SwingParams.right doit être >= 1")


def _validate_input(df: pd.DataFrame) -> None:
    required = {"high", "low"}

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes OHLCV manquantes pour detect_swings(): {sorted(missing)}"
        )

    if len(df) == 0:
        return

    if df["high"].isna().any():
        raise ValueError("La colonne 'high' contient des NaN")

    if df["low"].isna().any():
        raise ValueError("La colonne 'low' contient des NaN")


def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    for column in (
        "timestamp",
        "open_time",
        "datetime",
        "date",
        "time",
    ):
        if column in df.columns:
            return column

    return None


def detect_swings(
    df: pd.DataFrame,
    params: SwingParams = SwingParams(),
) -> pd.DataFrame:

    _validate_input(df)

    out = df.copy()
    n = len(out)

    out["swing_high"] = False
    out["swing_low"] = False

    out["swing_high_price"] = np.nan
    out["swing_low_price"] = np.nan

    int_columns = (
        "swing_high_origin_idx",
        "swing_low_origin_idx",
        "swing_high_confirmation_idx",
        "swing_low_confirmation_idx",
        "swing_high_strength_left",
        "swing_high_strength_right",
        "swing_low_strength_left",
        "swing_low_strength_right",
    )

    for c in int_columns:
        out[c] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

    datetime_dtype = "datetime64[ns, UTC]"

    time_columns = (
        "swing_high_origin_time",
        "swing_low_origin_time",
        "swing_high_confirmation_time",
        "swing_low_confirmation_time",
    )

    for c in time_columns:
        out[c] = pd.Series(
            pd.array([pd.NaT] * n, dtype=datetime_dtype),
            index=out.index,
        )

    if n == 0:
        return out

    left = int(params.left)
    right = int(params.right)

    if n < left + right + 1:
        return out

    highs = out["high"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    lows = out["low"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    # ---------------------------------------------------------------------
    # VECTORISATION:
    #
    # pivot positions:
    #   left ... n-right-1
    #
    # On ne boucle que sur les petits offsets left/right,
    # jamais sur les millions de candles.
    # ---------------------------------------------------------------------

    lo = left
    hi = n - right

    pivot_high = highs[lo:hi]
    pivot_low = lows[lo:hi]

    is_high = np.ones(
        hi - lo,
        dtype=bool,
    )

    is_low = np.ones(
        hi - lo,
        dtype=bool,
    )

    for offset in range(1, left + 1):

        other_high = highs[
            lo - offset:
            hi - offset
        ]

        other_low = lows[
            lo - offset:
            hi - offset
        ]

        if params.strict:
            is_high &= pivot_high > other_high
            is_low &= pivot_low < other_low
        else:
            is_high &= pivot_high >= other_high
            is_low &= pivot_low <= other_low

    for offset in range(1, right + 1):

        other_high = highs[
            lo + offset:
            hi + offset
        ]

        other_low = lows[
            lo + offset:
            hi + offset
        ]

        if params.strict:
            is_high &= pivot_high > other_high
            is_low &= pivot_low < other_low
        else:
            is_high &= pivot_high >= other_high
            is_low &= pivot_low <= other_low

    high_origin = (
        np.flatnonzero(is_high)
        + lo
    )

    low_origin = (
        np.flatnonzero(is_low)
        + lo
    )

    high_confirm = (
        high_origin
        + right
    )

    low_confirm = (
        low_origin
        + right
    )

    # ---------------------------------------------------------------------
    # NumPy arrays intermédiaires.
    # ---------------------------------------------------------------------

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    swing_high_price = np.full(n, np.nan)
    swing_low_price = np.full(n, np.nan)

    swing_high[high_confirm] = True
    swing_low[low_confirm] = True

    swing_high_price[high_confirm] = highs[high_origin]
    swing_low_price[low_confirm] = lows[low_origin]

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low

    out["swing_high_price"] = swing_high_price
    out["swing_low_price"] = swing_low_price

    # Nullable Int64, comme l'ancienne implémentation.
    def nullable_int(values_idx, values):
        arr = pd.array(
            [pd.NA] * n,
            dtype="Int64",
        )

        if len(values_idx):
            arr[values_idx] = values

        return pd.Series(
            arr,
            index=out.index,
        )

    out["swing_high_origin_idx"] = nullable_int(
        high_confirm,
        high_origin,
    )

    out["swing_low_origin_idx"] = nullable_int(
        low_confirm,
        low_origin,
    )

    out["swing_high_confirmation_idx"] = nullable_int(
        high_confirm,
        high_confirm,
    )

    out["swing_low_confirmation_idx"] = nullable_int(
        low_confirm,
        low_confirm,
    )

    out["swing_high_strength_left"] = nullable_int(
        high_confirm,
        np.full(
            len(high_confirm),
            left,
            dtype=np.int64,
        ),
    )

    out["swing_high_strength_right"] = nullable_int(
        high_confirm,
        np.full(
            len(high_confirm),
            right,
            dtype=np.int64,
        ),
    )

    out["swing_low_strength_left"] = nullable_int(
        low_confirm,
        np.full(
            len(low_confirm),
            left,
            dtype=np.int64,
        ),
    )

    out["swing_low_strength_right"] = nullable_int(
        low_confirm,
        np.full(
            len(low_confirm),
            right,
            dtype=np.int64,
        ),
    )

    timestamp_column = _find_timestamp_column(out)

    if timestamp_column is not None:

        timestamps = pd.to_datetime(
            out[timestamp_column],
            utc=True,
        )

        high_origin_time = pd.Series(
            pd.array(
                [pd.NaT] * n,
                dtype=datetime_dtype,
            ),
            index=out.index,
        )

        high_confirmation_time = high_origin_time.copy()

        low_origin_time = high_origin_time.copy()
        low_confirmation_time = high_origin_time.copy()

        if len(high_origin):
            high_origin_time.iloc[
                high_confirm
            ] = timestamps.iloc[
                high_origin
            ].to_numpy()

            high_confirmation_time.iloc[
                high_confirm
            ] = timestamps.iloc[
                high_confirm
            ].to_numpy()

        if len(low_origin):
            low_origin_time.iloc[
                low_confirm
            ] = timestamps.iloc[
                low_origin
            ].to_numpy()

            low_confirmation_time.iloc[
                low_confirm
            ] = timestamps.iloc[
                low_confirm
            ].to_numpy()

        out["swing_high_origin_time"] = high_origin_time
        out["swing_high_confirmation_time"] = high_confirmation_time

        out["swing_low_origin_time"] = low_origin_time
        out["swing_low_confirmation_time"] = low_confirmation_time

    return out
