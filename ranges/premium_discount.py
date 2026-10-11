from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PremiumDiscountParams:
    """
    Position du prix dans le Dealing Range.

    position = (price - range_low) / (range_high - range_low)

    position:
        0.0 = bas du range
        0.5 = equilibrium
        1.0 = haut du range

    Discount:
        position < discount_max

    Premium:
        position > premium_min

    price_mode:
        "close"
        "hl2"
        "ohlc4"

    clip_position:
        si True, position bornée entre 0 et 1.
        si False, permet de mesurer les excursions hors range.
    """

    discount_max: float = 0.5
    premium_min: float = 0.5

    price_mode: str = "close"

    clip_position: bool = False

    def __post_init__(self) -> None:

        if not 0 <= self.discount_max <= 1:
            raise ValueError(
                "discount_max doit être entre 0 et 1"
            )

        if not 0 <= self.premium_min <= 1:
            raise ValueError(
                "premium_min doit être entre 0 et 1"
            )

        if self.price_mode not in {
            "close",
            "hl2",
            "ohlc4",
        }:
            raise ValueError(
                "price_mode doit être close, hl2 ou ohlc4"
            )


def detect_premium_discount(
    df: pd.DataFrame,
    params: PremiumDiscountParams = PremiumDiscountParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "high",
        "low",
        "close",

        "dealing_range_active",
        "dealing_range_low",
        "dealing_range_high",
        "dealing_range_mid",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_premium_discount(): "
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

    range_low = out[
        "dealing_range_low"
    ].to_numpy(dtype=np.float64)

    range_high = out[
        "dealing_range_high"
    ].to_numpy(dtype=np.float64)

    active = out[
        "dealing_range_active"
    ].to_numpy(dtype=bool)

    if params.price_mode == "close":

        price = close

    elif params.price_mode == "hl2":

        price = (
            high + low
        ) / 2.0

    else:

        price = (
            open_
            + high
            + low
            + close
        ) / 4.0

    size = (
        range_high
        - range_low
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        position_raw = (
            price
            - range_low
        ) / size

    position = (
        np.clip(
            position_raw,
            0.0,
            1.0,
        )
        if params.clip_position
        else position_raw
    )

    valid = (
        active
        & np.isfinite(position)
        & (size > 0)
    )

    out[
        "range_position"
    ] = np.where(
        valid,
        position,
        np.nan,
    )

    out[
        "range_position_raw"
    ] = np.where(
        valid,
        position_raw,
        np.nan,
    )

    out[
        "discount"
    ] = (
        valid
        & (
            position
            < params.discount_max
        )
    )

    out[
        "premium"
    ] = (
        valid
        & (
            position
            > params.premium_min
        )
    )

    out[
        "range_below"
    ] = (
        valid
        & (
            position_raw < 0
        )
    )

    out[
        "range_above"
    ] = (
        valid
        & (
            position_raw > 1
        )
    )

    out[
        "distance_from_equilibrium_fraction"
    ] = np.where(
        valid,
        position_raw - 0.5,
        np.nan,
    )

    return out
