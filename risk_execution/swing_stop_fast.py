from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass(frozen=True)
class SwingStopParams:
    buffer_mode: str = "none"
    buffer: float = 0.0
    max_swing_age_bars: int | None = None
    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.buffer_mode not in {
            "none",
            "atr",
            "percent",
            "absolute",
        }:
            raise ValueError(
                "buffer_mode doit être none, atr, percent ou absolute"
            )

        if self.buffer < 0:
            raise ValueError(
                "buffer doit être >= 0"
            )

        if (
            self.max_swing_age_bars is not None
            and self.max_swing_age_bars < 0
        ):
            raise ValueError(
                "max_swing_age_bars doit être >= 0 ou None"
            )


@njit(cache=True)
def _swing_stop_kernel(
    close,
    atr,
    swing_high,
    swing_low,
    swing_high_price,
    swing_low_price,
    buffer_mode,
    buffer,
    max_age,
):

    n = len(close)

    bull_stop = np.full(n, np.nan)
    bear_stop = np.full(n, np.nan)

    bull_available = np.zeros(n, dtype=np.bool_)
    bear_available = np.zeros(n, dtype=np.bool_)

    bull_risk = np.full(n, np.nan)
    bear_risk = np.full(n, np.nan)

    bull_risk_atr = np.full(n, np.nan)
    bear_risk_atr = np.full(n, np.nan)

    bull_source_idx = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    bear_source_idx = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    bull_age = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    bear_age = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    latest_high_idx = -1
    latest_low_idx = -1

    latest_high_price = 0.0
    latest_low_price = 0.0

    for i in range(n):

        # Même ordre que l'original :
        # le swing confirmé sur i devient disponible sur i.
        if swing_high[i]:
            latest_high_idx = i
            latest_high_price = swing_high_price[i]

        if swing_low[i]:
            latest_low_idx = i
            latest_low_price = swing_low_price[i]

        # =============================================================
        # BULL STOP
        # =============================================================

        if latest_low_idx >= 0:

            age = (
                i
                - latest_low_idx
            )

            if (
                max_age < 0
                or age <= max_age
            ):

                if buffer_mode == 0:
                    buf = 0.0

                elif buffer_mode == 1:
                    if (
                        np.isfinite(atr[i])
                        and atr[i] > 0.0
                    ):
                        buf = (
                            atr[i]
                            * buffer
                        )
                    else:
                        buf = np.nan

                elif buffer_mode == 2:
                    buf = (
                        latest_low_price
                        * buffer
                    )

                else:
                    buf = buffer

                if np.isfinite(buf):

                    stop = (
                        latest_low_price
                        - buf
                    )

                    risk = (
                        close[i]
                        - stop
                    )

                    if risk > 0.0:

                        bull_stop[i] = stop
                        bull_available[i] = True
                        bull_risk[i] = risk

                        bull_source_idx[i] = (
                            latest_low_idx
                        )

                        bull_age[i] = age

                        if (
                            np.isfinite(atr[i])
                            and atr[i] > 0.0
                        ):
                            bull_risk_atr[i] = (
                                risk
                                / atr[i]
                            )

        # =============================================================
        # BEAR STOP
        # =============================================================

        if latest_high_idx >= 0:

            age = (
                i
                - latest_high_idx
            )

            if (
                max_age < 0
                or age <= max_age
            ):

                if buffer_mode == 0:
                    buf = 0.0

                elif buffer_mode == 1:
                    if (
                        np.isfinite(atr[i])
                        and atr[i] > 0.0
                    ):
                        buf = (
                            atr[i]
                            * buffer
                        )
                    else:
                        buf = np.nan

                elif buffer_mode == 2:
                    buf = (
                        latest_high_price
                        * buffer
                    )

                else:
                    buf = buffer

                if np.isfinite(buf):

                    stop = (
                        latest_high_price
                        + buf
                    )

                    risk = (
                        stop
                        - close[i]
                    )

                    if risk > 0.0:

                        bear_stop[i] = stop
                        bear_available[i] = True
                        bear_risk[i] = risk

                        bear_source_idx[i] = (
                            latest_high_idx
                        )

                        bear_age[i] = age

                        if (
                            np.isfinite(atr[i])
                            and atr[i] > 0.0
                        ):
                            bear_risk_atr[i] = (
                                risk
                                / atr[i]
                            )

    return (
        bull_stop,
        bear_stop,
        bull_available,
        bear_available,
        bull_risk,
        bear_risk,
        bull_risk_atr,
        bear_risk_atr,
        bull_source_idx,
        bear_source_idx,
        bull_age,
        bear_age,
    )


def _nullable_int(
    values: np.ndarray,
    index,
) -> pd.Series:

    arr = pd.array(
        values,
        dtype="Int64",
    )

    arr[values < 0] = pd.NA

    return pd.Series(
        arr,
        index=index,
    )


def detect_swing_stops(
    df: pd.DataFrame,
    params: SwingStopParams = SwingStopParams(),
) -> pd.DataFrame:

    required = {
        "close",
        "swing_high",
        "swing_low",
        "swing_high_price",
        "swing_low_price",
        params.atr_column,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_swing_stops(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    mode_map = {
        "none": 0,
        "atr": 1,
        "percent": 2,
        "absolute": 3,
    }

    result = _swing_stop_kernel(
        out["close"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        out[params.atr_column].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        out["swing_high"].to_numpy(
            dtype=np.bool_,
            copy=False,
        ),
        out["swing_low"].to_numpy(
            dtype=np.bool_,
            copy=False,
        ),
        out["swing_high_price"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        out["swing_low_price"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        mode_map[
            params.buffer_mode
        ],
        float(
            params.buffer
        ),
        (
            -1
            if params.max_swing_age_bars is None
            else int(
                params.max_swing_age_bars
            )
        ),
    )

    (
        bull_stop,
        bear_stop,
        bull_available,
        bear_available,
        bull_risk,
        bear_risk,
        bull_risk_atr,
        bear_risk_atr,
        bull_source_idx,
        bear_source_idx,
        bull_age,
        bear_age,
    ) = result

    out["swing_stop_bull"] = bull_stop
    out["swing_stop_bear"] = bear_stop

    out[
        "swing_stop_bull_available"
    ] = bull_available

    out[
        "swing_stop_bear_available"
    ] = bear_available

    out[
        "swing_stop_bull_risk"
    ] = bull_risk

    out[
        "swing_stop_bear_risk"
    ] = bear_risk

    out[
        "swing_stop_bull_risk_atr"
    ] = bull_risk_atr

    out[
        "swing_stop_bear_risk_atr"
    ] = bear_risk_atr

    out[
        "swing_stop_bull_source_idx"
    ] = _nullable_int(
        bull_source_idx,
        out.index,
    )

    out[
        "swing_stop_bear_source_idx"
    ] = _nullable_int(
        bear_source_idx,
        out.index,
    )

    out[
        "swing_stop_bull_age_bars"
    ] = _nullable_int(
        bull_age,
        out.index,
    )

    out[
        "swing_stop_bear_age_bars"
    ] = _nullable_int(
        bear_age,
        out.index,
    )

    return out
