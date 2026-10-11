from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PartialPlanParams:
    """
    Partial exit plan v0.1.

    Décrit un plan de sortie fractionnée.

    Exemple:
        rr_levels = (1.0, 2.0, 3.0)
        fractions = (0.50, 0.25, 0.25)

    => 50% à 1R
       25% à 2R
       25% à 3R

    Les fractions doivent totaliser 1.

    Cette feature ne simule PAS encore l'exécution;
    elle expose les barrières et poids.
    """

    rr_levels: tuple[float, ...] = (
        1.0,
        2.0,
        3.0,
    )

    fractions: tuple[float, ...] = (
        0.50,
        0.25,
        0.25,
    )

    entry_bull_column: str = "rr_bull_entry"
    entry_bear_column: str = "rr_bear_entry"

    risk_bull_column: str = "rr_bull_risk"
    risk_bear_column: str = "rr_bear_risk"

    def __post_init__(self) -> None:

        if (
            len(self.rr_levels)
            != len(self.fractions)
        ):
            raise ValueError(
                "rr_levels et fractions doivent "
                "avoir la même longueur"
            )

        if not self.rr_levels:
            raise ValueError(
                "Le plan de partials ne peut pas être vide"
            )

        if any(
            rr <= 0
            for rr in self.rr_levels
        ):
            raise ValueError(
                "Tous les RR doivent être > 0"
            )

        if any(
            fraction <= 0
            for fraction in self.fractions
        ):
            raise ValueError(
                "Toutes les fractions doivent être > 0"
            )

        if not np.isclose(
            sum(self.fractions),
            1.0,
        ):
            raise ValueError(
                "La somme des fractions doit être égale à 1"
            )


def detect_partial_plan(
    df: pd.DataFrame,
    params: PartialPlanParams = PartialPlanParams(),
) -> pd.DataFrame:

    required = {
        params.entry_bull_column,
        params.entry_bear_column,

        params.risk_bull_column,
        params.risk_bear_column,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_partial_plan(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    bull_entry = out[
        params.entry_bull_column
    ].to_numpy(
        dtype=np.float64,
    )

    bear_entry = out[
        params.entry_bear_column
    ].to_numpy(
        dtype=np.float64,
    )

    bull_risk = out[
        params.risk_bull_column
    ].to_numpy(
        dtype=np.float64,
    )

    bear_risk = out[
        params.risk_bear_column
    ].to_numpy(
        dtype=np.float64,
    )

    bull_valid = (
        np.isfinite(bull_entry)
        & np.isfinite(bull_risk)
        & (bull_risk > 0)
    )

    bear_valid = (
        np.isfinite(bear_entry)
        & np.isfinite(bear_risk)
        & (bear_risk > 0)
    )

    out[
        "partial_bull_valid"
    ] = bull_valid

    out[
        "partial_bear_valid"
    ] = bear_valid

    # Expected full-plan R if every target is hit.
    weighted_rr = float(
        sum(
            rr * fraction
            for rr, fraction
            in zip(
                params.rr_levels,
                params.fractions,
            )
        )
    )

    out[
        "partial_plan_max_realized_r"
    ] = weighted_rr

    for j, (
        rr,
        fraction,
    ) in enumerate(
        zip(
            params.rr_levels,
            params.fractions,
        ),
        start=1,
    ):

        out[
            f"partial_{j}_rr"
        ] = rr

        out[
            f"partial_{j}_fraction"
        ] = fraction

        out[
            f"partial_bull_{j}_target"
        ] = np.where(
            bull_valid,
            (
                bull_entry
                + rr * bull_risk
            ),
            np.nan,
        )

        out[
            f"partial_bear_{j}_target"
        ] = np.where(
            bear_valid,
            (
                bear_entry
                - rr * bear_risk
            ),
            np.nan,
        )

    return out
