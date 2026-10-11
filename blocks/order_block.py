from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OrderBlockParams:
    """
    Order Block v0.1.

    Bullish OB:
        dernière bougie bearish avant un BOS bullish.

    Bearish OB:
        dernière bougie bullish avant un BOS bearish.

    search_back:
        nombre maximal de bougies avant le BOS pour chercher
        la dernière bougie opposée.

    zone_mode:
        "full" -> [low, high] de la bougie source
        "body" -> [min(open, close), max(open, close)]

    require_displacement:
        si True, le BOS doit être accompagné d'un displacement
        de même direction sur la bougie du BOS.

    invalidation_mode:
        "wick" ou "close".
    """

    search_back: int = 10
    zone_mode: str = "full"

    require_displacement: bool = False

    invalidation_mode: str = "close"

    def __post_init__(self) -> None:

        if self.search_back < 1:
            raise ValueError(
                "search_back doit être >= 1"
            )

        if self.zone_mode not in {
            "full",
            "body",
        }:
            raise ValueError(
                "zone_mode doit être 'full' ou 'body'"
            )

        if self.invalidation_mode not in {
            "wick",
            "close",
        }:
            raise ValueError(
                "invalidation_mode doit être 'wick' ou 'close'"
            )


def _zone_from_candle(
    open_: float,
    high: float,
    low: float,
    close: float,
    mode: str,
) -> tuple[float, float]:

    if mode == "full":
        return (
            low,
            high,
        )

    return (
        min(open_, close),
        max(open_, close),
    )


def detect_order_blocks(
    df: pd.DataFrame,
    params: OrderBlockParams = OrderBlockParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "high",
        "low",
        "close",

        "bos_bull",
        "bos_bear",
    }

    if params.require_displacement:
        required |= {
            "displacement_bull",
            "displacement_bear",
        }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_order_blocks(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    for side in (
        "ob_bull",
        "ob_bear",
    ):

        out[side] = False

        out[f"{side}_low"] = np.nan
        out[f"{side}_high"] = np.nan
        out[f"{side}_mid"] = np.nan

        out[f"{side}_active"] = False
        out[f"{side}_invalidated"] = False
        out[f"{side}_mitigated"] = False

        out[f"{side}_fill_ratio"] = np.nan

        for suffix in (
            "origin_idx",
            "confirmation_idx",
            "age_bars",
            "first_touch_idx",
            "invalidation_idx",
        ):

            out[f"{side}_{suffix}"] = pd.Series(
                pd.array(
                    [pd.NA] * n,
                    dtype="Int64",
                ),
                index=out.index,
            )

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

    bos_bull = out[
        "bos_bull"
    ].to_numpy(
        dtype=bool,
    )

    bos_bear = out[
        "bos_bear"
    ].to_numpy(
        dtype=bool,
    )

    if "displacement_bull" in out.columns:

        displacement_bull = out[
            "displacement_bull"
        ].to_numpy(
            dtype=bool,
        )

        displacement_bear = out[
            "displacement_bear"
        ].to_numpy(
            dtype=bool,
        )

    else:

        displacement_bull = np.ones(
            n,
            dtype=bool,
        )

        displacement_bear = np.ones(
            n,
            dtype=bool,
        )

    active_bull = []
    active_bear = []

    for i in range(n):

        # =====================================================================
        # UPDATE EXISTING BULLISH OB
        # =====================================================================

        surviving = []

        for zone in active_bull:

            zone["age"] += 1

            if i > zone["confirmation_idx"]:

                touched = (
                    low[i] <= zone["high"]
                    and high[i] >= zone["low"]
                )

                if touched:

                    if zone["first_touch_idx"] is None:
                        zone["first_touch_idx"] = i

                    penetration = (
                        zone["high"]
                        - max(
                            low[i],
                            zone["low"],
                        )
                    )

                    zone["fill_ratio"] = max(
                        zone["fill_ratio"],
                        float(
                            np.clip(
                                penetration
                                / zone["size"],
                                0.0,
                                1.0,
                            )
                        ),
                    )

                    zone["mitigated"] = True

                invalidation_price = (
                    low[i]
                    if params.invalidation_mode == "wick"
                    else close[i]
                )

                if (
                    invalidation_price
                    < zone["low"]
                ):

                    zone["invalidated"] = True
                    zone["invalidation_idx"] = i

                else:

                    surviving.append(
                        zone
                    )

        active_bull = surviving

        # =====================================================================
        # UPDATE EXISTING BEARISH OB
        # =====================================================================

        surviving = []

        for zone in active_bear:

            zone["age"] += 1

            if i > zone["confirmation_idx"]:

                touched = (
                    high[i] >= zone["low"]
                    and low[i] <= zone["high"]
                )

                if touched:

                    if zone["first_touch_idx"] is None:
                        zone["first_touch_idx"] = i

                    penetration = (
                        min(
                            high[i],
                            zone["high"],
                        )
                        - zone["low"]
                    )

                    zone["fill_ratio"] = max(
                        zone["fill_ratio"],
                        float(
                            np.clip(
                                penetration
                                / zone["size"],
                                0.0,
                                1.0,
                            )
                        ),
                    )

                    zone["mitigated"] = True

                invalidation_price = (
                    high[i]
                    if params.invalidation_mode == "wick"
                    else close[i]
                )

                if (
                    invalidation_price
                    > zone["high"]
                ):

                    zone["invalidated"] = True
                    zone["invalidation_idx"] = i

                else:

                    surviving.append(
                        zone
                    )

        active_bear = surviving

        # =====================================================================
        # NEW BULLISH ORDER BLOCK
        # =====================================================================

        bull_ok = (
            bos_bull[i]
            and (
                not params.require_displacement
                or displacement_bull[i]
            )
        )

        if bull_ok:

            start = max(
                0,
                i - params.search_back,
            )

            source = None

            for j in range(
                i - 1,
                start - 1,
                -1,
            ):

                if close[j] < open_[j]:

                    source = j
                    break

            if source is not None:

                zone_low, zone_high = (
                    _zone_from_candle(
                        open_[source],
                        high[source],
                        low[source],
                        close[source],
                        params.zone_mode,
                    )
                )

                size = (
                    zone_high
                    - zone_low
                )

                if size > 0:

                    mid = (
                        zone_low
                        + zone_high
                    ) / 2.0

                    label = out.index[i]

                    out.at[
                        label,
                        "ob_bull",
                    ] = True

                    out.at[
                        label,
                        "ob_bull_low",
                    ] = zone_low

                    out.at[
                        label,
                        "ob_bull_high",
                    ] = zone_high

                    out.at[
                        label,
                        "ob_bull_mid",
                    ] = mid

                    out.at[
                        label,
                        "ob_bull_active",
                    ] = True

                    out.at[
                        label,
                        "ob_bull_fill_ratio",
                    ] = 0.0

                    out.at[
                        label,
                        "ob_bull_origin_idx",
                    ] = source

                    out.at[
                        label,
                        "ob_bull_confirmation_idx",
                    ] = i

                    out.at[
                        label,
                        "ob_bull_age_bars",
                    ] = 0

                    active_bull.append(
                        {
                            "low":
                                zone_low,

                            "high":
                                zone_high,

                            "mid":
                                mid,

                            "size":
                                size,

                            "origin_idx":
                                source,

                            "confirmation_idx":
                                i,

                            "age":
                                0,

                            "first_touch_idx":
                                None,

                            "fill_ratio":
                                0.0,

                            "mitigated":
                                False,

                            "invalidated":
                                False,

                            "invalidation_idx":
                                None,
                        }
                    )

        # =====================================================================
        # NEW BEARISH ORDER BLOCK
        # =====================================================================

        bear_ok = (
            bos_bear[i]
            and (
                not params.require_displacement
                or displacement_bear[i]
            )
        )

        if bear_ok:

            start = max(
                0,
                i - params.search_back,
            )

            source = None

            for j in range(
                i - 1,
                start - 1,
                -1,
            ):

                if close[j] > open_[j]:

                    source = j
                    break

            if source is not None:

                zone_low, zone_high = (
                    _zone_from_candle(
                        open_[source],
                        high[source],
                        low[source],
                        close[source],
                        params.zone_mode,
                    )
                )

                size = (
                    zone_high
                    - zone_low
                )

                if size > 0:

                    mid = (
                        zone_low
                        + zone_high
                    ) / 2.0

                    label = out.index[i]

                    out.at[
                        label,
                        "ob_bear",
                    ] = True

                    out.at[
                        label,
                        "ob_bear_low",
                    ] = zone_low

                    out.at[
                        label,
                        "ob_bear_high",
                    ] = zone_high

                    out.at[
                        label,
                        "ob_bear_mid",
                    ] = mid

                    out.at[
                        label,
                        "ob_bear_active",
                    ] = True

                    out.at[
                        label,
                        "ob_bear_fill_ratio",
                    ] = 0.0

                    out.at[
                        label,
                        "ob_bear_origin_idx",
                    ] = source

                    out.at[
                        label,
                        "ob_bear_confirmation_idx",
                    ] = i

                    out.at[
                        label,
                        "ob_bear_age_bars",
                    ] = 0

                    active_bear.append(
                        {
                            "low":
                                zone_low,

                            "high":
                                zone_high,

                            "mid":
                                mid,

                            "size":
                                size,

                            "origin_idx":
                                source,

                            "confirmation_idx":
                                i,

                            "age":
                                0,

                            "first_touch_idx":
                                None,

                            "fill_ratio":
                                0.0,

                            "mitigated":
                                False,

                            "invalidated":
                                False,

                            "invalidation_idx":
                                None,
                        }
                    )

    return out
