from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolumeImbalanceParams:
    """
    Proxy OHLCV de Volume Imbalance.

    Sans données bid/ask ou order book, on ne mesure PAS un vrai
    imbalance d'ordres agressifs.

    Proxy:
        bougie directionnelle
        +
        volume anormalement élevé
        +
        corps dominant.

    volume_period:
        fenêtre rolling.

    min_volume_ratio:
        volume / moyenne rolling.

    min_volume_zscore:
        z-score minimal.

    min_body_ratio:
        body / range minimal.
    """

    volume_period: int = 20

    min_volume_ratio: float = 1.5
    min_volume_zscore: float = 1.0
    min_body_ratio: float = 0.6

    def __post_init__(self) -> None:

        if self.volume_period < 2:
            raise ValueError(
                "volume_period doit être >= 2"
            )

        if self.min_volume_ratio < 0:
            raise ValueError(
                "min_volume_ratio doit être >= 0"
            )

        if self.min_body_ratio < 0 or self.min_body_ratio > 1:
            raise ValueError(
                "min_body_ratio doit être entre 0 et 1"
            )


def detect_volume_imbalance(
    df: pd.DataFrame,
    params: VolumeImbalanceParams = VolumeImbalanceParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "detect_volume_imbalance(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    open_ = out["open"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)
    volume = out["volume"].to_numpy(dtype=np.float64)

    candle_range = (
        high - low
    )

    body = np.abs(
        close - open_
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        body_ratio = np.where(
            candle_range > 0,
            body / candle_range,
            np.nan,
        )

    volume_series = pd.Series(
        volume
    )

    volume_mean = (
        volume_series
        .rolling(
            params.volume_period,
            min_periods=params.volume_period,
        )
        .mean()
        .to_numpy()
    )

    volume_std = (
        volume_series
        .rolling(
            params.volume_period,
            min_periods=params.volume_period,
        )
        .std(ddof=0)
        .to_numpy()
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        volume_ratio = (
            volume / volume_mean
        )

        volume_zscore = (
            volume - volume_mean
        ) / volume_std

    out["volume_imbalance_volume_ratio"] = (
        volume_ratio
    )

    out["volume_imbalance_volume_zscore"] = (
        volume_zscore
    )

    out["volume_imbalance_body_ratio"] = (
        body_ratio
    )

    base = (
        (volume_ratio >= params.min_volume_ratio)
        & (
            volume_zscore
            >= params.min_volume_zscore
        )
        & (
            body_ratio
            >= params.min_body_ratio
        )
    )

    out["volume_imbalance_bull"] = (
        base
        & (close > open_)
    )

    out["volume_imbalance_bear"] = (
        base
        & (close < open_)
    )

    return out
