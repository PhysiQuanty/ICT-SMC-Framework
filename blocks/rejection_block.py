from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RejectionBlockParams:
    """
    Rejection Block proxy.

    Bullish rejection:
        grande lower wick
        + close situé haut dans le range.

    Bearish rejection:
        grande upper wick
        + close situé bas dans le range.

    min_wick_ratio:
        wick / candle range.

    min_close_location:
        bullish: close_location >= seuil
        bearish: close_location <= 1-seuil

    min_range_atr:
        filtre de taille minimale.
    """

    min_wick_ratio: float = 0.50
    min_close_location: float = 0.65
    min_range_atr: float = 0.5

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if not 0 <= self.min_wick_ratio <= 1:
            raise ValueError(
                "min_wick_ratio doit être entre 0 et 1"
            )

        if not 0 <= self.min_close_location <= 1:
            raise ValueError(
                "min_close_location doit être entre 0 et 1"
            )

        if self.min_range_atr < 0:
            raise ValueError(
                "min_range_atr doit être >= 0"
            )


def detect_rejection_blocks(
    df: pd.DataFrame,
    params: RejectionBlockParams = RejectionBlockParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "high",
        "low",
        "close",
        params.atr_column,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_rejection_blocks(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    open_ = out["open"].to_numpy(
        dtype=np.float64,
    )

    high = out["high"].to_numpy(
        dtype=np.float64,
    )

    low = out["low"].to_numpy(
        dtype=np.float64,
    )

    close = out["close"].to_numpy(
        dtype=np.float64,
    )

    atr = out[
        params.atr_column
    ].to_numpy(
        dtype=np.float64,
    )

    candle_range = (
        high - low
    )

    upper_wick = (
        high
        - np.maximum(
            open_,
            close,
        )
    )

    lower_wick = (
        np.minimum(
            open_,
            close,
        )
        - low
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        upper_wick_ratio = (
            upper_wick
            / candle_range
        )

        lower_wick_ratio = (
            lower_wick
            / candle_range
        )

        close_location = (
            close - low
        ) / candle_range

        range_atr = (
            candle_range
            / atr
        )

    bull = (
        (
            lower_wick_ratio
            >= params.min_wick_ratio
        )
        & (
            close_location
            >= params.min_close_location
        )
        & (
            range_atr
            >= params.min_range_atr
        )
    )

    bear = (
        (
            upper_wick_ratio
            >= params.min_wick_ratio
        )
        & (
            close_location
            <= (
                1.0
                - params.min_close_location
            )
        )
        & (
            range_atr
            >= params.min_range_atr
        )
    )

    out["rejection_bull"] = bull
    out["rejection_bear"] = bear

    out["rejection_bull_low"] = np.where(
        bull,
        low,
        np.nan,
    )

    out["rejection_bull_high"] = np.where(
        bull,
        np.minimum(
            open_,
            close,
        ),
        np.nan,
    )

    out["rejection_bear_low"] = np.where(
        bear,
        np.maximum(
            open_,
            close,
        ),
        np.nan,
    )

    out["rejection_bear_high"] = np.where(
        bear,
        high,
        np.nan,
    )

    out["rejection_bull_mid"] = (
        out[
            "rejection_bull_low"
        ]
        + out[
            "rejection_bull_high"
        ]
    ) / 2.0

    out["rejection_bear_mid"] = (
        out[
            "rejection_bear_low"
        ]
        + out[
            "rejection_bear_high"
        ]
    ) / 2.0

    out["rejection_lower_wick_ratio"] = (
        lower_wick_ratio
    )

    out["rejection_upper_wick_ratio"] = (
        upper_wick_ratio
    )

    out["rejection_close_location"] = (
        close_location
    )

    out["rejection_range_atr"] = (
        range_atr
    )

    return out
