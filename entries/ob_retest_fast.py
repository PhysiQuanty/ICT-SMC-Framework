from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass(frozen=True)
class OBRetestEntryParams:
    entry_mode: str = "touch_close"
    invalidation_mode: str = "close"
    max_age_bars: int = 100
    first_touch_only: bool = True

    def __post_init__(self) -> None:

        if self.entry_mode not in {
            "touch_close",
            "near",
            "mid",
            "deep",
        }:
            raise ValueError(
                "entry_mode doit être touch_close, near, mid ou deep"
            )

        if self.invalidation_mode not in {
            "close",
            "wick",
        }:
            raise ValueError(
                "invalidation_mode doit être close ou wick"
            )

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )


@njit(cache=True)
def _retest_kernel(
    high,
    low,
    close,
    ob_bull,
    ob_bear,
    ob_bull_low,
    ob_bull_high,
    ob_bull_mid,
    ob_bear_low,
    ob_bear_high,
    ob_bear_mid,
    entry_mode,
    invalidation_mode,
    max_age,
    first_touch_only,
):

    n = len(close)

    entry_bull = np.zeros(
        n,
        dtype=np.bool_,
    )

    entry_bear = np.zeros(
        n,
        dtype=np.bool_,
    )

    bull_price = np.full(
        n,
        np.nan,
    )

    bear_price = np.full(
        n,
        np.nan,
    )

    bull_ratio = np.full(
        n,
        np.nan,
    )

    bear_ratio = np.full(
        n,
        np.nan,
    )

    bull_source = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    bear_source = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    bull_confirmation = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    bear_confirmation = np.full(
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

    # =============================================================
    # IMPORTANT:
    #
    # On parcourt les OB dans l'ordre chronologique.
    #
    # Donc si plusieurs OB touchent la même candle, un OB plus récent
    # écrira APRÈS un OB plus ancien.
    #
    # C'est exactement la sémantique de la boucle active_bull /
    # active_bear originale.
    # =============================================================

    for source in range(n):

        # =========================================================
        # BULL OB
        # =========================================================

        if ob_bull[source]:

            zlow = ob_bull_low[source]
            zhigh = ob_bull_high[source]
            zmid = ob_bull_mid[source]

            size = (
                zhigh
                - zlow
            )

            last = min(
                n - 1,
                source + max_age,
            )

            # Le nouvel OB n'est actif qu'APRÈS sa candle
            # de confirmation.
            for i in range(
                source + 1,
                last + 1,
            ):

                # Invalidation testée AVANT touch,
                # comme dans l'original.
                if invalidation_mode == 0:
                    inv_price = close[i]
                else:
                    inv_price = low[i]

                if inv_price < zlow:
                    break

                if entry_mode == 1:
                    # mid
                    touched = (
                        low[i] <= zmid
                        and high[i] >= zmid
                    )

                    eprice = zmid

                elif entry_mode == 2:
                    # deep
                    touched = (
                        low[i] <= zlow
                    )

                    eprice = zlow

                else:
                    # touch_close ou near
                    touched = (
                        low[i] <= zhigh
                        and high[i] >= zlow
                    )

                    if entry_mode == 3:
                        # near
                        eprice = zhigh
                    else:
                        eprice = close[i]

                penetration = (
                    zhigh
                    - max(
                        low[i],
                        zlow,
                    )
                )

                if size > 0.0:
                    ratio = (
                        penetration
                        / size
                    )

                    if ratio < 0.0:
                        ratio = 0.0
                    elif ratio > 1.0:
                        ratio = 1.0

                else:
                    ratio = np.nan

                if touched:

                    entry_bull[i] = True
                    bull_price[i] = eprice
                    bull_ratio[i] = ratio

                    bull_source[i] = source
                    bull_confirmation[i] = i
                    bull_age[i] = (
                        i - source
                    )

                    if first_touch_only:
                        break

        # =========================================================
        # BEAR OB
        # =========================================================

        if ob_bear[source]:

            zlow = ob_bear_low[source]
            zhigh = ob_bear_high[source]
            zmid = ob_bear_mid[source]

            size = (
                zhigh
                - zlow
            )

            last = min(
                n - 1,
                source + max_age,
            )

            for i in range(
                source + 1,
                last + 1,
            ):

                if invalidation_mode == 0:
                    inv_price = close[i]
                else:
                    inv_price = high[i]

                if inv_price > zhigh:
                    break

                if entry_mode == 1:
                    # mid
                    touched = (
                        high[i] >= zmid
                        and low[i] <= zmid
                    )

                    eprice = zmid

                elif entry_mode == 2:
                    # deep
                    touched = (
                        high[i] >= zhigh
                    )

                    eprice = zhigh

                else:
                    touched = (
                        high[i] >= zlow
                        and low[i] <= zhigh
                    )

                    if entry_mode == 3:
                        # near
                        eprice = zlow
                    else:
                        eprice = close[i]

                penetration = (
                    min(
                        high[i],
                        zhigh,
                    )
                    - zlow
                )

                if size > 0.0:
                    ratio = (
                        penetration
                        / size
                    )

                    if ratio < 0.0:
                        ratio = 0.0
                    elif ratio > 1.0:
                        ratio = 1.0

                else:
                    ratio = np.nan

                if touched:

                    entry_bear[i] = True
                    bear_price[i] = eprice
                    bear_ratio[i] = ratio

                    bear_source[i] = source
                    bear_confirmation[i] = i
                    bear_age[i] = (
                        i - source
                    )

                    if first_touch_only:
                        break

    return (
        entry_bull,
        entry_bear,
        bull_price,
        bear_price,
        bull_ratio,
        bear_ratio,
        bull_source,
        bear_source,
        bull_confirmation,
        bear_confirmation,
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


def detect_ob_retest_entries(
    df: pd.DataFrame,
    params: OBRetestEntryParams = OBRetestEntryParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",

        "ob_bull",
        "ob_bear",

        "ob_bull_low",
        "ob_bull_high",
        "ob_bull_mid",

        "ob_bear_low",
        "ob_bear_high",
        "ob_bear_mid",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_ob_retest_entries(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    entry_mode_map = {
        "touch_close": 0,
        "mid": 1,
        "deep": 2,
        "near": 3,
    }

    invalidation_mode_map = {
        "close": 0,
        "wick": 1,
    }

    result = _retest_kernel(
        out["high"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        out["low"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
        out["close"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),

        out["ob_bull"].to_numpy(
            dtype=np.bool_,
            copy=False,
        ),

        out["ob_bear"].to_numpy(
            dtype=np.bool_,
            copy=False,
        ),

        out["ob_bull_low"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),

        out["ob_bull_high"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),

        out["ob_bull_mid"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),

        out["ob_bear_low"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),

        out["ob_bear_high"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),

        out["ob_bear_mid"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),

        entry_mode_map[
            params.entry_mode
        ],

        invalidation_mode_map[
            params.invalidation_mode
        ],

        int(
            params.max_age_bars
        ),

        bool(
            params.first_touch_only
        ),
    )

    (
        entry_bull,
        entry_bear,
        bull_price,
        bear_price,
        bull_ratio,
        bear_ratio,
        bull_source,
        bear_source,
        bull_confirmation,
        bear_confirmation,
        bull_age,
        bear_age,
    ) = result

    out[
        "entry_ob_bull"
    ] = entry_bull

    out[
        "entry_ob_bear"
    ] = entry_bear

    out[
        "entry_ob_bull_price"
    ] = bull_price

    out[
        "entry_ob_bear_price"
    ] = bear_price

    out[
        "entry_ob_bull_penetration_ratio"
    ] = bull_ratio

    out[
        "entry_ob_bear_penetration_ratio"
    ] = bear_ratio

    out[
        "entry_ob_bull_source_idx"
    ] = _nullable_int(
        bull_source,
        out.index,
    )

    out[
        "entry_ob_bear_source_idx"
    ] = _nullable_int(
        bear_source,
        out.index,
    )

    out[
        "entry_ob_bull_confirmation_idx"
    ] = _nullable_int(
        bull_confirmation,
        out.index,
    )

    out[
        "entry_ob_bear_confirmation_idx"
    ] = _nullable_int(
        bear_confirmation,
        out.index,
    )

    out[
        "entry_ob_bull_age_bars"
    ] = _nullable_int(
        bull_age,
        out.index,
    )

    out[
        "entry_ob_bear_age_bars"
    ] = _nullable_int(
        bear_age,
        out.index,
    )

    return out
