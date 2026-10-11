from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpposingLiquidityTargetParams:
    """
    Opposing Liquidity Target v0.1.

    Consomme les niveaux de liquidité déjà construits par Narrative.

    Pour un mouvement / setup bullish:
        cible = liquidité disponible AU-DESSUS.

    Pour un mouvement / setup bearish:
        cible = liquidité disponible EN DESSOUS.

    source_mode:

        "nearest"
            nearest_liquidity_above/below
            ERL ou IRL, whichever is closest.

        "external"
            nearest_external_above/below uniquement.

        "internal"
            nearest_internal_above/below uniquement.

        "draw"
            draw_on_liquidity_up/down.

    Cette feature ne génère pas de trade.
    Elle expose seulement des cibles candidates dans les deux directions.
    """

    source_mode: str = "nearest"

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.source_mode not in {
            "nearest",
            "external",
            "internal",
            "draw",
        }:
            raise ValueError(
                "source_mode doit être nearest, "
                "external, internal ou draw"
            )


def detect_opposing_liquidity_targets(
    df: pd.DataFrame,
    params: OpposingLiquidityTargetParams = (
        OpposingLiquidityTargetParams()
    ),
) -> pd.DataFrame:

    required = {
        "close",
        params.atr_column,
    }

    if params.source_mode == "nearest":

        required.update(
            {
                "nearest_liquidity_above",
                "nearest_liquidity_below",
                "nearest_liquidity_above_class",
                "nearest_liquidity_below_class",
            }
        )

        up_col = "nearest_liquidity_above"
        down_col = "nearest_liquidity_below"

        up_type_col = "nearest_liquidity_above_class"
        down_type_col = "nearest_liquidity_below_class"

    elif params.source_mode == "external":

        required.update(
            {
                "nearest_external_above",
                "nearest_external_below",
                "nearest_external_above_type",
                "nearest_external_below_type",
            }
        )

        up_col = "nearest_external_above"
        down_col = "nearest_external_below"

        up_type_col = "nearest_external_above_type"
        down_type_col = "nearest_external_below_type"

    elif params.source_mode == "internal":

        required.update(
            {
                "nearest_internal_above",
                "nearest_internal_below",
                "nearest_internal_above_type",
                "nearest_internal_below_type",
            }
        )

        up_col = "nearest_internal_above"
        down_col = "nearest_internal_below"

        up_type_col = "nearest_internal_above_type"
        down_type_col = "nearest_internal_below_type"

    else:

        required.update(
            {
                "draw_on_liquidity_up",
                "draw_on_liquidity_down",
                "draw_on_liquidity_up_class",
                "draw_on_liquidity_down_class",
            }
        )

        up_col = "draw_on_liquidity_up"
        down_col = "draw_on_liquidity_down"

        up_type_col = "draw_on_liquidity_up_class"
        down_type_col = "draw_on_liquidity_down_class"

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_opposing_liquidity_targets(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    close = out[
        "close"
    ].to_numpy(
        dtype=np.float64,
    )

    atr = out[
        params.atr_column
    ].to_numpy(
        dtype=np.float64,
    )

    up = out[
        up_col
    ].to_numpy(
        dtype=np.float64,
    )

    down = out[
        down_col
    ].to_numpy(
        dtype=np.float64,
    )

    # =========================================================================
    # TARGET LEVELS
    # =========================================================================

    out[
        "target_opposing_liquidity_bull"
    ] = up

    out[
        "target_opposing_liquidity_bear"
    ] = down

    out[
        "target_opposing_liquidity_bull_type"
    ] = out[
        up_type_col
    ].astype(
        "object"
    )

    out[
        "target_opposing_liquidity_bear_type"
    ] = out[
        down_type_col
    ].astype(
        "object"
    )

    # =========================================================================
    # AVAILABILITY
    # =========================================================================

    bull_available = (
        np.isfinite(up)
        & (up > close)
    )

    bear_available = (
        np.isfinite(down)
        & (down < close)
    )

    out[
        "target_opposing_liquidity_bull_available"
    ] = bull_available

    out[
        "target_opposing_liquidity_bear_available"
    ] = bear_available

    # =========================================================================
    # DISTANCES
    # =========================================================================

    bull_distance = (
        up - close
    )

    bear_distance = (
        close - down
    )

    out[
        "target_opposing_liquidity_bull_distance"
    ] = np.where(
        bull_available,
        bull_distance,
        np.nan,
    )

    out[
        "target_opposing_liquidity_bear_distance"
    ] = np.where(
        bear_available,
        bear_distance,
        np.nan,
    )

    valid_atr = (
        np.isfinite(atr)
        & (atr > 0)
    )

    out[
        "target_opposing_liquidity_bull_distance_atr"
    ] = np.where(
        bull_available
        & valid_atr,
        bull_distance / atr,
        np.nan,
    )

    out[
        "target_opposing_liquidity_bear_distance_atr"
    ] = np.where(
        bear_available
        & valid_atr,
        bear_distance / atr,
        np.nan,
    )

    return out
