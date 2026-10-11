from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DisplacementParams:
    """
    Détection paramétrique d'un displacement.

    Une bougie doit satisfaire:
        body_atr >= min_body_atr
        range_atr >= min_range_atr
        body_ratio >= min_body_ratio

    Pour bullish:
        close_location >= min_close_location

    Pour bearish:
        close_location <= 1 - min_close_location
    """

    atr_period: int = 14

    min_body_atr: float = 1.0
    min_range_atr: float = 1.5

    min_body_ratio: float = 0.5
    min_close_location: float = 0.7

    volume_period: int = 20

    def __post_init__(self) -> None:
        if self.atr_period < 1:
            raise ValueError("atr_period doit être >= 1")

        if self.volume_period < 1:
            raise ValueError("volume_period doit être >= 1")

        if not 0 <= self.min_body_ratio <= 1:
            raise ValueError("min_body_ratio doit être entre 0 et 1")

        if not 0 <= self.min_close_location <= 1:
            raise ValueError("min_close_location doit être entre 0 et 1")


def _atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:

    prev_close = np.empty_like(close)

    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )

    return (
        pd.Series(tr)
        .rolling(period, min_periods=period)
        .mean()
        .to_numpy()
    )


def detect_displacement(
    df: pd.DataFrame,
    params: DisplacementParams = DisplacementParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "high",
        "low",
        "close",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"detect_displacement(): colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    open_ = out["open"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)

    candle_range = high - low
    body = np.abs(close - open_)

    atr = _atr(
        high,
        low,
        close,
        params.atr_period,
    )

    with np.errstate(divide="ignore", invalid="ignore"):

        body_atr = body / atr
        range_atr = candle_range / atr

        body_ratio = np.where(
            candle_range > 0,
            body / candle_range,
            np.nan,
        )

        close_location = np.where(
            candle_range > 0,
            (close - low) / candle_range,
            np.nan,
        )

        upper_wick_ratio = np.where(
            candle_range > 0,
            (
                high
                - np.maximum(open_, close)
            ) / candle_range,
            np.nan,
        )

        lower_wick_ratio = np.where(
            candle_range > 0,
            (
                np.minimum(open_, close)
                - low
            ) / candle_range,
            np.nan,
        )

    out["displacement_atr"] = atr

    out["displacement_body"] = body
    out["displacement_range"] = candle_range

    out["displacement_body_atr"] = body_atr
    out["displacement_range_atr"] = range_atr

    out["displacement_body_ratio"] = body_ratio
    out["displacement_close_location"] = close_location

    out["displacement_upper_wick_ratio"] = upper_wick_ratio
    out["displacement_lower_wick_ratio"] = lower_wick_ratio

    previous_close = np.roll(close, 1)
    previous_close[0] = np.nan

    out["displacement_return"] = (
        close / previous_close - 1.0
    )

    out["displacement_log_return"] = np.log(
        close / previous_close
    )

    # -----------------------------------------------------------------
    # Volume characterization
    # -----------------------------------------------------------------

    if "volume" in out.columns:

        volume = out["volume"].to_numpy(dtype=np.float64)

        volume_mean = (
            pd.Series(volume)
            .rolling(
                params.volume_period,
                min_periods=params.volume_period,
            )
            .mean()
            .to_numpy()
        )

        volume_std = (
            pd.Series(volume)
            .rolling(
                params.volume_period,
                min_periods=params.volume_period,
            )
            .std(ddof=0)
            .to_numpy()
        )

        with np.errstate(divide="ignore", invalid="ignore"):

            out["displacement_volume_ratio"] = (
                volume / volume_mean
            )

            out["displacement_volume_zscore"] = (
                (volume - volume_mean)
                / volume_std
            )

    else:

        out["displacement_volume_ratio"] = np.nan
        out["displacement_volume_zscore"] = np.nan

    base_condition = (
        (body_atr >= params.min_body_atr)
        & (range_atr >= params.min_range_atr)
        & (body_ratio >= params.min_body_ratio)
    )

    bull = (
        base_condition
        & (close > open_)
        & (
            close_location
            >= params.min_close_location
        )
    )

    bear = (
        base_condition
        & (close < open_)
        & (
            close_location
            <= 1.0 - params.min_close_location
        )
    )

    out["displacement_bull"] = bull
    out["displacement_bear"] = bear

    return out
