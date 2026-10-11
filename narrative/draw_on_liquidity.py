from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawOnLiquidityParams:
    """
    Draw on Liquidity (DOL) proxy v0.1.

    Le framework ne prétend PAS inférer une intention du marché.

    Il expose seulement une cible candidate dans chaque direction.

    mode:

        "nearest"
            niveau de liquidité le plus proche.

        "external_first"
            priorité ERL; IRL utilisé seulement si aucun ERL
            n'existe dans la direction.

        "internal_first"
            priorité IRL; ERL utilisé seulement si aucun IRL.

    On produit indépendamment:
        draw_on_liquidity_up
        draw_on_liquidity_down

    Puis une préférence contextuelle optionnelle fondée sur
    `dealing_range_direction`.

    Cette préférence reste une variable descriptive,
    PAS un signal de trade.
    """

    mode: str = "nearest"
    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.mode not in {
            "nearest",
            "external_first",
            "internal_first",
        }:

            raise ValueError(
                "mode doit être nearest, "
                "external_first ou internal_first"
            )


def _choose_above(
    erl: float,
    irl: float,
    mode: str,
) -> tuple[
    float,
    str | None,
]:

    erl_ok = np.isfinite(
        erl
    )

    irl_ok = np.isfinite(
        irl
    )

    if mode == "external_first":

        if erl_ok:
            return (
                erl,
                "ERL",
            )

        if irl_ok:
            return (
                irl,
                "IRL",
            )

    elif mode == "internal_first":

        if irl_ok:
            return (
                irl,
                "IRL",
            )

        if erl_ok:
            return (
                erl,
                "ERL",
            )

    else:

        candidates = []

        if erl_ok:
            candidates.append(
                (
                    erl,
                    "ERL",
                )
            )

        if irl_ok:
            candidates.append(
                (
                    irl,
                    "IRL",
                )
            )

        if candidates:

            return min(
                candidates,
                key=lambda x: x[0],
            )

    return (
        np.nan,
        None,
    )


def _choose_below(
    erl: float,
    irl: float,
    mode: str,
) -> tuple[
    float,
    str | None,
]:

    erl_ok = np.isfinite(
        erl
    )

    irl_ok = np.isfinite(
        irl
    )

    if mode == "external_first":

        if erl_ok:
            return (
                erl,
                "ERL",
            )

        if irl_ok:
            return (
                irl,
                "IRL",
            )

    elif mode == "internal_first":

        if irl_ok:
            return (
                irl,
                "IRL",
            )

        if erl_ok:
            return (
                erl,
                "ERL",
            )

    else:

        candidates = []

        if erl_ok:
            candidates.append(
                (
                    erl,
                    "ERL",
                )
            )

        if irl_ok:
            candidates.append(
                (
                    irl,
                    "IRL",
                )
            )

        if candidates:

            # Closest level below price is the largest one.
            return max(
                candidates,
                key=lambda x: x[0],
            )

    return (
        np.nan,
        None,
    )


def detect_draw_on_liquidity(
    df: pd.DataFrame,
    params: DrawOnLiquidityParams = DrawOnLiquidityParams(),
) -> pd.DataFrame:

    required = {
        "close",
        params.atr_column,

        "nearest_external_above",
        "nearest_external_below",

        "nearest_internal_above",
        "nearest_internal_below",

        "dealing_range_direction",
    }

    missing = required.difference(
        df.columns
    )

    if missing:

        raise ValueError(
            "detect_draw_on_liquidity(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

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

    erl_up = out[
        "nearest_external_above"
    ].to_numpy(
        dtype=np.float64,
    )

    erl_down = out[
        "nearest_external_below"
    ].to_numpy(
        dtype=np.float64,
    )

    irl_up = out[
        "nearest_internal_above"
    ].to_numpy(
        dtype=np.float64,
    )

    irl_down = out[
        "nearest_internal_below"
    ].to_numpy(
        dtype=np.float64,
    )

    direction = out[
        "dealing_range_direction"
    ].to_numpy()

    # =========================================================================
    # OUTPUTS
    # =========================================================================

    out[
        "draw_on_liquidity_up"
    ] = np.nan

    out[
        "draw_on_liquidity_down"
    ] = np.nan

    out[
        "draw_on_liquidity_up_class"
    ] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out[
        "draw_on_liquidity_down_class"
    ] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out[
        "draw_on_liquidity_up_distance"
    ] = np.nan

    out[
        "draw_on_liquidity_down_distance"
    ] = np.nan

    out[
        "draw_on_liquidity_up_distance_atr"
    ] = np.nan

    out[
        "draw_on_liquidity_down_distance_atr"
    ] = np.nan

    # Contextual directional projection.
    out[
        "draw_on_liquidity_context"
    ] = np.nan

    out[
        "draw_on_liquidity_context_direction"
    ] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out[
        "draw_on_liquidity_context_class"
    ] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out[
        "draw_on_liquidity_context_distance_atr"
    ] = np.nan

    # =========================================================================
    # LOOP
    # =========================================================================

    for i in range(n):

        up, up_class = _choose_above(
            erl_up[i],
            irl_up[i],
            params.mode,
        )

        down, down_class = _choose_below(
            erl_down[i],
            irl_down[i],
            params.mode,
        )

        label = out.index[i]

        if np.isfinite(
            up
        ):

            distance = (
                up - close[i]
            )

            out.at[
                label,
                "draw_on_liquidity_up",
            ] = up

            out.at[
                label,
                "draw_on_liquidity_up_class",
            ] = up_class

            out.at[
                label,
                "draw_on_liquidity_up_distance",
            ] = distance

            if (
                np.isfinite(atr[i])
                and atr[i] > 0
            ):

                out.at[
                    label,
                    "draw_on_liquidity_up_distance_atr",
                ] = (
                    distance
                    / atr[i]
                )

        if np.isfinite(
            down
        ):

            distance = (
                close[i] - down
            )

            out.at[
                label,
                "draw_on_liquidity_down",
            ] = down

            out.at[
                label,
                "draw_on_liquidity_down_class",
            ] = down_class

            out.at[
                label,
                "draw_on_liquidity_down_distance",
            ] = distance

            if (
                np.isfinite(atr[i])
                and atr[i] > 0
            ):

                out.at[
                    label,
                    "draw_on_liquidity_down_distance_atr",
                ] = (
                    distance
                    / atr[i]
                )

        # =====================================================================
        # CONTEXTUAL DOL
        #
        # Bull dealing range -> upward candidate
        # Bear dealing range -> downward candidate
        #
        # Still descriptive, not an entry signal.
        # =====================================================================

        if (
            direction[i] == "bull"
            and np.isfinite(up)
        ):

            out.at[
                label,
                "draw_on_liquidity_context",
            ] = up

            out.at[
                label,
                "draw_on_liquidity_context_direction",
            ] = "up"

            out.at[
                label,
                "draw_on_liquidity_context_class",
            ] = up_class

            out.at[
                label,
                "draw_on_liquidity_context_distance_atr",
            ] = out.at[
                label,
                "draw_on_liquidity_up_distance_atr",
            ]

        elif (
            direction[i] == "bear"
            and np.isfinite(down)
        ):

            out.at[
                label,
                "draw_on_liquidity_context",
            ] = down

            out.at[
                label,
                "draw_on_liquidity_context_direction",
            ] = "down"

            out.at[
                label,
                "draw_on_liquidity_context_class",
            ] = down_class

            out.at[
                label,
                "draw_on_liquidity_context_distance_atr",
            ] = out.at[
                label,
                "draw_on_liquidity_down_distance_atr",
            ]

    return out
