from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EquilibriumParams:
    """
    Equilibrium du Dealing Range.

    EQ = (range_low + range_high) / 2

    tolerance_mode:
        "atr"
        "range_fraction"
        "absolute"

    tolerance:
        largeur autour de l'EQ considérée comme zone d'équilibre.

    Exemple:
        tolerance_mode="range_fraction"
        tolerance=0.05

        => zone EQ = 45%-55% du range.
    """

    tolerance_mode: str = "range_fraction"
    tolerance: float = 0.05

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.tolerance_mode not in {
            "atr",
            "range_fraction",
            "absolute",
        }:
            raise ValueError(
                "tolerance_mode invalide"
            )

        if self.tolerance < 0:
            raise ValueError(
                "tolerance doit être >= 0"
            )


def detect_equilibrium(
    df: pd.DataFrame,
    params: EquilibriumParams = EquilibriumParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",

        "dealing_range_active",
        "dealing_range_low",
        "dealing_range_high",
        "dealing_range_mid",
        "dealing_range_size",
    }

    if params.tolerance_mode == "atr":
        required.add(
            params.atr_column
        )

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_equilibrium(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    high = out["high"].to_numpy(
        dtype=np.float64,
    )

    low = out["low"].to_numpy(
        dtype=np.float64,
    )

    close = out["close"].to_numpy(
        dtype=np.float64,
    )

    eq = out[
        "dealing_range_mid"
    ].to_numpy(dtype=np.float64)

    size = out[
        "dealing_range_size"
    ].to_numpy(dtype=np.float64)

    active = out[
        "dealing_range_active"
    ].to_numpy(dtype=bool)

    if params.atr_column in out.columns:

        atr = out[
            params.atr_column
        ].to_numpy(dtype=np.float64)

    else:

        atr = np.full(
            len(out),
            np.nan,
        )

    if params.tolerance_mode == "atr":

        half_width = (
            atr
            * params.tolerance
        )

    elif params.tolerance_mode == "range_fraction":

        half_width = (
            size
            * params.tolerance
        )

    else:

        half_width = np.full(
            len(out),
            params.tolerance,
            dtype=np.float64,
        )

    eq_low = (
        eq - half_width
    )

    eq_high = (
        eq + half_width
    )

    valid = (
        active
        & np.isfinite(eq)
        & np.isfinite(half_width)
    )

    touch = (
        valid
        & (
            low <= eq_high
        )
        & (
            high >= eq_low
        )
    )

    out[
        "equilibrium"
    ] = valid

    out[
        "equilibrium_level"
    ] = np.where(
        valid,
        eq,
        np.nan,
    )

    out[
        "equilibrium_low"
    ] = np.where(
        valid,
        eq_low,
        np.nan,
    )

    out[
        "equilibrium_high"
    ] = np.where(
        valid,
        eq_high,
        np.nan,
    )

    out[
        "equilibrium_touch"
    ] = touch

    out[
        "equilibrium_distance"
    ] = np.where(
        valid,
        close - eq,
        np.nan,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        out[
            "equilibrium_distance_range"
        ] = np.where(
            valid & (size > 0),
            (
                close - eq
            ) / size,
            np.nan,
        )

        out[
            "equilibrium_distance_atr"
        ] = np.where(
            valid
            & np.isfinite(atr)
            & (atr > 0),
            (
                close - eq
            ) / atr,
            np.nan,
        )

    return out
