from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ERLIRLParams:
    """
    ERL / IRL v0.1.

    ERL:
        External Range Liquidity.

    IRL:
        Internal Range Liquidity.

    Cette feature consomme la classification produite par
    external_internal_liquidity.py et expose des distances
    quantitatives depuis le prix courant.

    Aucune direction de trade n'est imposée.
    """

    atr_column: str = "atr"


def detect_erl_irl(
    df: pd.DataFrame,
    params: ERLIRLParams = ERLIRLParams(),
) -> pd.DataFrame:

    required = {
        "close",
        params.atr_column,

        "nearest_external_above",
        "nearest_external_below",

        "nearest_internal_above",
        "nearest_internal_below",

        "nearest_external_above_type",
        "nearest_external_below_type",

        "nearest_internal_above_type",
        "nearest_internal_below_type",
    }

    missing = required.difference(
        df.columns
    )

    if missing:

        raise ValueError(
            "detect_erl_irl(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    close = out[
        "close"
    ].to_numpy(dtype=np.float64)

    atr = out[
        params.atr_column
    ].to_numpy(dtype=np.float64)

    erl_above = out[
        "nearest_external_above"
    ].to_numpy(dtype=np.float64)

    erl_below = out[
        "nearest_external_below"
    ].to_numpy(dtype=np.float64)

    irl_above = out[
        "nearest_internal_above"
    ].to_numpy(dtype=np.float64)

    irl_below = out[
        "nearest_internal_below"
    ].to_numpy(dtype=np.float64)

    # =========================================================================
    # EXISTENCE
    # =========================================================================

    out[
        "erl_above_available"
    ] = np.isfinite(
        erl_above
    )

    out[
        "erl_below_available"
    ] = np.isfinite(
        erl_below
    )

    out[
        "irl_above_available"
    ] = np.isfinite(
        irl_above
    )

    out[
        "irl_below_available"
    ] = np.isfinite(
        irl_below
    )

    # =========================================================================
    # RAW DISTANCES
    # =========================================================================

    out[
        "erl_above_distance"
    ] = (
        erl_above
        - close
    )

    out[
        "erl_below_distance"
    ] = (
        close
        - erl_below
    )

    out[
        "irl_above_distance"
    ] = (
        irl_above
        - close
    )

    out[
        "irl_below_distance"
    ] = (
        close
        - irl_below
    )

    # =========================================================================
    # ATR DISTANCES
    # =========================================================================

    valid_atr = (
        np.isfinite(atr)
        & (atr > 0)
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        out[
            "erl_above_distance_atr"
        ] = np.where(
            valid_atr,
            (
                erl_above
                - close
            ) / atr,
            np.nan,
        )

        out[
            "erl_below_distance_atr"
        ] = np.where(
            valid_atr,
            (
                close
                - erl_below
            ) / atr,
            np.nan,
        )

        out[
            "irl_above_distance_atr"
        ] = np.where(
            valid_atr,
            (
                irl_above
                - close
            ) / atr,
            np.nan,
        )

        out[
            "irl_below_distance_atr"
        ] = np.where(
            valid_atr,
            (
                close
                - irl_below
            ) / atr,
            np.nan,
        )

    # =========================================================================
    # NEAREST CLASS
    #
    # Which class is closest in each direction?
    # =========================================================================

    out[
        "nearest_liquidity_above_class"
    ] = pd.Series(
        [None] * len(out),
        dtype="object",
        index=out.index,
    )

    out[
        "nearest_liquidity_below_class"
    ] = pd.Series(
        [None] * len(out),
        dtype="object",
        index=out.index,
    )

    out[
        "nearest_liquidity_above"
    ] = np.nan

    out[
        "nearest_liquidity_below"
    ] = np.nan

    for i in range(
        len(out)
    ):

        # ABOVE
        candidates = []

        if np.isfinite(
            erl_above[i]
        ):

            candidates.append(
                (
                    erl_above[i],
                    "ERL",
                )
            )

        if np.isfinite(
            irl_above[i]
        ):

            candidates.append(
                (
                    irl_above[i],
                    "IRL",
                )
            )

        if candidates:

            level, cls = min(
                candidates,
                key=lambda x: x[0],
            )

            out.at[
                out.index[i],
                "nearest_liquidity_above",
            ] = level

            out.at[
                out.index[i],
                "nearest_liquidity_above_class",
            ] = cls

        # BELOW
        candidates = []

        if np.isfinite(
            erl_below[i]
        ):

            candidates.append(
                (
                    erl_below[i],
                    "ERL",
                )
            )

        if np.isfinite(
            irl_below[i]
        ):

            candidates.append(
                (
                    irl_below[i],
                    "IRL",
                )
            )

        if candidates:

            level, cls = max(
                candidates,
                key=lambda x: x[0],
            )

            out.at[
                out.index[i],
                "nearest_liquidity_below",
            ] = level

            out.at[
                out.index[i],
                "nearest_liquidity_below_class",
            ] = cls

    return out
