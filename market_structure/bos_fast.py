from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass(frozen=True)
class BOSParams:
    break_mode: str = "close"
    min_break_atr: float = 0.0
    atr_column: str = "atr"
    consume_level: bool = True

    def __post_init__(self) -> None:
        if self.break_mode not in {"close", "wick"}:
            raise ValueError("break_mode doit être 'close' ou 'wick'")

        if self.min_break_atr < 0:
            raise ValueError("min_break_atr doit être >= 0")


@njit(cache=True)
def _bos_kernel(
    highs,
    lows,
    closes,
    atr,
    swing_high,
    swing_low,
    swing_high_price,
    swing_low_price,
    swing_high_origin_idx,
    swing_low_origin_idx,
    use_close,
    min_break_atr,
    consume_level,
):

    n = len(closes)

    bull = np.zeros(n, dtype=np.bool_)
    bear = np.zeros(n, dtype=np.bool_)

    bull_level = np.full(n, np.nan)
    bear_level = np.full(n, np.nan)

    bull_break_price = np.full(n, np.nan)
    bear_break_price = np.full(n, np.nan)

    bull_distance = np.full(n, np.nan)
    bear_distance = np.full(n, np.nan)

    bull_distance_atr = np.full(n, np.nan)
    bear_distance_atr = np.full(n, np.nan)

    bull_source_origin = np.full(n, -1, dtype=np.int64)
    bear_source_origin = np.full(n, -1, dtype=np.int64)

    bull_source_confirmation = np.full(n, -1, dtype=np.int64)
    bear_source_confirmation = np.full(n, -1, dtype=np.int64)

    bull_confirmation = np.full(n, -1, dtype=np.int64)
    bear_confirmation = np.full(n, -1, dtype=np.int64)

    bull_age = np.full(n, -1, dtype=np.int64)
    bear_age = np.full(n, -1, dtype=np.int64)

    have_high = False
    have_low = False

    latest_high_price = 0.0
    latest_low_price = 0.0

    latest_high_origin = -1
    latest_low_origin = -1

    latest_high_confirmation = -1
    latest_low_confirmation = -1

    high_consumed = False
    low_consumed = False

    for i in range(n):

        # -------------------------------------------------------------
        # Nouveau Swing High confirmé.
        # -------------------------------------------------------------

        if swing_high[i]:

            latest_high_price = swing_high_price[i]
            latest_high_origin = swing_high_origin_idx[i]
            latest_high_confirmation = i

            high_consumed = False
            have_high = True

        # -------------------------------------------------------------
        # Nouveau Swing Low confirmé.
        # -------------------------------------------------------------

        if swing_low[i]:

            latest_low_price = swing_low_price[i]
            latest_low_origin = swing_low_origin_idx[i]
            latest_low_confirmation = i

            low_consumed = False
            have_low = True

        # -------------------------------------------------------------
        # Bull BOS
        # -------------------------------------------------------------

        if have_high and not (
            consume_level
            and high_consumed
        ):

            if use_close:
                bp = closes[i]
            else:
                bp = highs[i]

            distance = (
                bp
                - latest_high_price
            )

            if (
                np.isfinite(atr[i])
                and atr[i] > 0.0
            ):
                d_atr = (
                    distance
                    / atr[i]
                )
            else:
                d_atr = np.nan

            ok = distance > 0.0

            if ok and min_break_atr != 0.0:
                ok = (
                    np.isfinite(d_atr)
                    and d_atr >= min_break_atr
                )

            if ok:

                bull[i] = True

                bull_level[i] = latest_high_price
                bull_break_price[i] = bp
                bull_distance[i] = distance
                bull_distance_atr[i] = d_atr

                bull_source_origin[i] = latest_high_origin
                bull_source_confirmation[i] = latest_high_confirmation
                bull_confirmation[i] = i

                bull_age[i] = (
                    i
                    - latest_high_confirmation
                )

                high_consumed = True

        # -------------------------------------------------------------
        # Bear BOS
        # -------------------------------------------------------------

        if have_low and not (
            consume_level
            and low_consumed
        ):

            if use_close:
                bp = closes[i]
            else:
                bp = lows[i]

            distance = (
                latest_low_price
                - bp
            )

            if (
                np.isfinite(atr[i])
                and atr[i] > 0.0
            ):
                d_atr = (
                    distance
                    / atr[i]
                )
            else:
                d_atr = np.nan

            ok = distance > 0.0

            if ok and min_break_atr != 0.0:
                ok = (
                    np.isfinite(d_atr)
                    and d_atr >= min_break_atr
                )

            if ok:

                bear[i] = True

                bear_level[i] = latest_low_price
                bear_break_price[i] = bp
                bear_distance[i] = distance
                bear_distance_atr[i] = d_atr

                bear_source_origin[i] = latest_low_origin
                bear_source_confirmation[i] = latest_low_confirmation
                bear_confirmation[i] = i

                bear_age[i] = (
                    i
                    - latest_low_confirmation
                )

                low_consumed = True

    return (
        bull,
        bear,

        bull_level,
        bear_level,

        bull_break_price,
        bear_break_price,

        bull_distance,
        bear_distance,

        bull_distance_atr,
        bear_distance_atr,

        bull_source_origin,
        bear_source_origin,

        bull_source_confirmation,
        bear_source_confirmation,

        bull_confirmation,
        bear_confirmation,

        bull_age,
        bear_age,
    )


def _nullable_int(
    values: np.ndarray,
    index,
) -> pd.Series:

    mask = (
        values < 0
    )

    arr = pd.array(
        values,
        dtype="Int64",
    )

    arr[mask] = pd.NA

    return pd.Series(
        arr,
        index=index,
    )


def detect_bos(
    df: pd.DataFrame,
    params: BOSParams = BOSParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
        "swing_high",
        "swing_low",
        "swing_high_price",
        "swing_low_price",
        "swing_high_origin_idx",
        "swing_low_origin_idx",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"detect_bos(): colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    highs = out["high"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    lows = out["low"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    closes = out["close"].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    if params.atr_column in out.columns:
        atr = out[
            params.atr_column
        ].to_numpy(
            dtype=np.float64,
            copy=False,
        )
    else:
        atr = np.full(
            n,
            np.nan,
        )

    swing_high = out[
        "swing_high"
    ].to_numpy(
        dtype=np.bool_,
        copy=False,
    )

    swing_low = out[
        "swing_low"
    ].to_numpy(
        dtype=np.bool_,
        copy=False,
    )

    swing_high_price = out[
        "swing_high_price"
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    swing_low_price = out[
        "swing_low_price"
    ].to_numpy(
        dtype=np.float64,
        copy=False,
    )

    # Nullable Int64 -> float -> sentinel -1.
    shi = (
        out["swing_high_origin_idx"]
        .fillna(-1)
        .to_numpy(
            dtype=np.int64,
        )
    )

    sli = (
        out["swing_low_origin_idx"]
        .fillna(-1)
        .to_numpy(
            dtype=np.int64,
        )
    )

    result = _bos_kernel(
        highs,
        lows,
        closes,
        atr,
        swing_high,
        swing_low,
        swing_high_price,
        swing_low_price,
        shi,
        sli,
        params.break_mode == "close",
        float(params.min_break_atr),
        bool(params.consume_level),
    )

    (
        bull,
        bear,

        bull_level,
        bear_level,

        bull_break_price,
        bear_break_price,

        bull_distance,
        bear_distance,

        bull_distance_atr,
        bear_distance_atr,

        bull_source_origin,
        bear_source_origin,

        bull_source_confirmation,
        bear_source_confirmation,

        bull_confirmation,
        bear_confirmation,

        bull_age,
        bear_age,
    ) = result

    out["bos_bull"] = bull
    out["bos_bear"] = bear

    out["bos_bull_level"] = bull_level
    out["bos_bear_level"] = bear_level

    out["bos_bull_break_price"] = bull_break_price
    out["bos_bear_break_price"] = bear_break_price

    out["bos_bull_break_distance"] = bull_distance
    out["bos_bear_break_distance"] = bear_distance

    out["bos_bull_break_distance_atr"] = bull_distance_atr
    out["bos_bear_break_distance_atr"] = bear_distance_atr

    out["bos_bull_source_origin_idx"] = _nullable_int(
        bull_source_origin,
        out.index,
    )

    out["bos_bear_source_origin_idx"] = _nullable_int(
        bear_source_origin,
        out.index,
    )

    out["bos_bull_source_confirmation_idx"] = _nullable_int(
        bull_source_confirmation,
        out.index,
    )

    out["bos_bear_source_confirmation_idx"] = _nullable_int(
        bear_source_confirmation,
        out.index,
    )

    out["bos_bull_confirmation_idx"] = _nullable_int(
        bull_confirmation,
        out.index,
    )

    out["bos_bear_confirmation_idx"] = _nullable_int(
        bear_confirmation,
        out.index,
    )

    out["bos_bull_source_age_bars"] = _nullable_int(
        bull_age,
        out.index,
    )

    out["bos_bear_source_age_bars"] = _nullable_int(
        bear_age,
        out.index,
    )

    return out
