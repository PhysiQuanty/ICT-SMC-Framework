from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass(frozen=True)
class OrderBlockParams:
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


@njit(cache=True)
def _order_block_kernel(
    open_,
    high,
    low,
    close,
    bos_bull,
    bos_bear,
    displacement_bull,
    displacement_bear,
    search_back,
    zone_mode,
    require_displacement,
):
    """
    Reproduit les SORTIES réellement écrites par l'implémentation originale.

    IMPORTANT:
    l'ancienne gestion active_bull/active_bear met à jour des dictionnaires
    internes (mitigated, invalidated, first_touch, fill_ratio...) mais ne
    réécrit jamais ces états dans les lignes du DataFrame déjà créées.

    Ces scans sont donc sans effet sur le DataFrame retourné.

    Ici on conserve strictement:
      - création OB
      - source candle
      - zone low/high/mid
      - confirmation_idx
      - age_bars = 0
      - active = True à la ligne de confirmation
      - fill_ratio = 0 à la ligne de confirmation
    """

    n = len(close)

    ob_bull = np.zeros(
        n,
        dtype=np.bool_,
    )

    ob_bear = np.zeros(
        n,
        dtype=np.bool_,
    )

    bull_low = np.full(
        n,
        np.nan,
    )

    bull_high = np.full(
        n,
        np.nan,
    )

    bull_mid = np.full(
        n,
        np.nan,
    )

    bear_low = np.full(
        n,
        np.nan,
    )

    bear_high = np.full(
        n,
        np.nan,
    )

    bear_mid = np.full(
        n,
        np.nan,
    )

    bull_active = np.zeros(
        n,
        dtype=np.bool_,
    )

    bear_active = np.zeros(
        n,
        dtype=np.bool_,
    )

    bull_fill_ratio = np.full(
        n,
        np.nan,
    )

    bear_fill_ratio = np.full(
        n,
        np.nan,
    )

    bull_origin = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    bear_origin = np.full(
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

    for i in range(n):

        # =============================================================
        # BULLISH ORDER BLOCK
        # dernière candle bearish avant BOS bullish
        # =============================================================

        bull_ok = bos_bull[i]

        if require_displacement:
            bull_ok = (
                bull_ok
                and displacement_bull[i]
            )

        if bull_ok:

            start = i - search_back

            if start < 0:
                start = 0

            source = -1

            for j in range(
                i - 1,
                start - 1,
                -1,
            ):

                if close[j] < open_[j]:
                    source = j
                    break

            if source >= 0:

                if zone_mode == 0:
                    zlow = low[source]
                    zhigh = high[source]

                else:
                    a = open_[source]
                    b = close[source]

                    if a < b:
                        zlow = a
                        zhigh = b
                    else:
                        zlow = b
                        zhigh = a

                size = (
                    zhigh
                    - zlow
                )

                if size > 0.0:

                    ob_bull[i] = True

                    bull_low[i] = zlow
                    bull_high[i] = zhigh
                    bull_mid[i] = (
                        zlow
                        + zhigh
                    ) * 0.5

                    bull_active[i] = True
                    bull_fill_ratio[i] = 0.0

                    bull_origin[i] = source
                    bull_confirmation[i] = i
                    bull_age[i] = 0

        # =============================================================
        # BEARISH ORDER BLOCK
        # dernière candle bullish avant BOS bearish
        # =============================================================

        bear_ok = bos_bear[i]

        if require_displacement:
            bear_ok = (
                bear_ok
                and displacement_bear[i]
            )

        if bear_ok:

            start = i - search_back

            if start < 0:
                start = 0

            source = -1

            for j in range(
                i - 1,
                start - 1,
                -1,
            ):

                if close[j] > open_[j]:
                    source = j
                    break

            if source >= 0:

                if zone_mode == 0:
                    zlow = low[source]
                    zhigh = high[source]

                else:
                    a = open_[source]
                    b = close[source]

                    if a < b:
                        zlow = a
                        zhigh = b
                    else:
                        zlow = b
                        zhigh = a

                size = (
                    zhigh
                    - zlow
                )

                if size > 0.0:

                    ob_bear[i] = True

                    bear_low[i] = zlow
                    bear_high[i] = zhigh
                    bear_mid[i] = (
                        zlow
                        + zhigh
                    ) * 0.5

                    bear_active[i] = True
                    bear_fill_ratio[i] = 0.0

                    bear_origin[i] = source
                    bear_confirmation[i] = i
                    bear_age[i] = 0

    return (
        ob_bull,
        ob_bear,

        bull_low,
        bull_high,
        bull_mid,

        bear_low,
        bear_high,
        bear_mid,

        bull_active,
        bear_active,

        bull_fill_ratio,
        bear_fill_ratio,

        bull_origin,
        bear_origin,

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

    arr[
        values < 0
    ] = pd.NA

    return pd.Series(
        arr,
        index=index,
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

    if "displacement_bull" in out.columns:

        displacement_bull = out[
            "displacement_bull"
        ].to_numpy(
            dtype=np.bool_,
            copy=False,
        )

        displacement_bear = out[
            "displacement_bear"
        ].to_numpy(
            dtype=np.bool_,
            copy=False,
        )

    else:

        displacement_bull = np.ones(
            n,
            dtype=np.bool_,
        )

        displacement_bear = np.ones(
            n,
            dtype=np.bool_,
        )

    result = _order_block_kernel(
        out["open"].to_numpy(
            dtype=np.float64,
            copy=False,
        ),
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
        out["bos_bull"].to_numpy(
            dtype=np.bool_,
            copy=False,
        ),
        out["bos_bear"].to_numpy(
            dtype=np.bool_,
            copy=False,
        ),
        displacement_bull,
        displacement_bear,
        int(
            params.search_back
        ),
        (
            0
            if params.zone_mode == "full"
            else 1
        ),
        bool(
            params.require_displacement
        ),
    )

    (
        ob_bull,
        ob_bear,

        bull_low,
        bull_high,
        bull_mid,

        bear_low,
        bear_high,
        bear_mid,

        bull_active,
        bear_active,

        bull_fill_ratio,
        bear_fill_ratio,

        bull_origin,
        bear_origin,

        bull_confirmation,
        bear_confirmation,

        bull_age,
        bear_age,
    ) = result

    # =================================================================
    # BULL
    # =================================================================

    out["ob_bull"] = ob_bull

    out["ob_bull_low"] = bull_low
    out["ob_bull_high"] = bull_high
    out["ob_bull_mid"] = bull_mid

    out["ob_bull_active"] = bull_active

    # Dans l'original, ces colonnes restent toujours False.
    out["ob_bull_invalidated"] = False
    out["ob_bull_mitigated"] = False

    out[
        "ob_bull_fill_ratio"
    ] = bull_fill_ratio

    out[
        "ob_bull_origin_idx"
    ] = _nullable_int(
        bull_origin,
        out.index,
    )

    out[
        "ob_bull_confirmation_idx"
    ] = _nullable_int(
        bull_confirmation,
        out.index,
    )

    out[
        "ob_bull_age_bars"
    ] = _nullable_int(
        bull_age,
        out.index,
    )

    out[
        "ob_bull_first_touch_idx"
    ] = _nullable_int(
        np.full(
            n,
            -1,
            dtype=np.int64,
        ),
        out.index,
    )

    out[
        "ob_bull_invalidation_idx"
    ] = _nullable_int(
        np.full(
            n,
            -1,
            dtype=np.int64,
        ),
        out.index,
    )

    # =================================================================
    # BEAR
    # =================================================================

    out["ob_bear"] = ob_bear

    out["ob_bear_low"] = bear_low
    out["ob_bear_high"] = bear_high
    out["ob_bear_mid"] = bear_mid

    out["ob_bear_active"] = bear_active

    out["ob_bear_invalidated"] = False
    out["ob_bear_mitigated"] = False

    out[
        "ob_bear_fill_ratio"
    ] = bear_fill_ratio

    out[
        "ob_bear_origin_idx"
    ] = _nullable_int(
        bear_origin,
        out.index,
    )

    out[
        "ob_bear_confirmation_idx"
    ] = _nullable_int(
        bear_confirmation,
        out.index,
    )

    out[
        "ob_bear_age_bars"
    ] = _nullable_int(
        bear_age,
        out.index,
    )

    out[
        "ob_bear_first_touch_idx"
    ] = _nullable_int(
        np.full(
            n,
            -1,
            dtype=np.int64,
        ),
        out.index,
    )

    out[
        "ob_bear_invalidation_idx"
    ] = _nullable_int(
        np.full(
            n,
            -1,
            dtype=np.int64,
        ),
        out.index,
    )

    return out
