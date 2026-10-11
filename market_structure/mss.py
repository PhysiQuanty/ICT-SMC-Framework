from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MSSParams:
    """
    Market Structure Shift.

    Définition retenue pour cette primitive:

        CHoCH
        +
        displacement directionnel

    max_displacement_lag:
        nombre maximal de bougies APRÈS le CHoCH pendant lesquelles
        un displacement peut confirmer le MSS.

        0 -> même bougie uniquement
        1 -> même bougie ou bougie suivante
        etc.

    Strictement causal:
        aucun displacement futur n'est utilisé avant qu'il arrive.
    """

    max_displacement_lag: int = 2

    def __post_init__(self) -> None:
        if self.max_displacement_lag < 0:
            raise ValueError(
                "max_displacement_lag doit être >= 0"
            )


def detect_mss(
    df: pd.DataFrame,
    params: MSSParams = MSSParams(),
) -> pd.DataFrame:

    required = {
        "choch_bull",
        "choch_bear",
        "displacement_bull",
        "displacement_bear",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"detect_mss(): colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["mss_bull"] = False
    out["mss_bear"] = False

    out["mss_bull_choch_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["mss_bear_choch_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["mss_bull_confirmation_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["mss_bear_confirmation_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["mss_bull_confirmation_lag"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["mss_bear_confirmation_lag"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    pending_bull = None
    pending_bear = None

    for i in range(n):

        row = out.iloc[i]
        label = out.index[i]

        # -------------------------------------------------------------
        # Nouveau CHoCH.
        # -------------------------------------------------------------

        if bool(row["choch_bull"]):
            pending_bull = i

        if bool(row["choch_bear"]):
            pending_bear = i

        # -------------------------------------------------------------
        # Expiration.
        # -------------------------------------------------------------

        if (
            pending_bull is not None
            and i - pending_bull > params.max_displacement_lag
        ):
            pending_bull = None

        if (
            pending_bear is not None
            and i - pending_bear > params.max_displacement_lag
        ):
            pending_bear = None

        # -------------------------------------------------------------
        # Confirmation MSS bullish.
        # -------------------------------------------------------------

        if (
            pending_bull is not None
            and bool(row["displacement_bull"])
        ):

            out.at[label, "mss_bull"] = True

            out.at[
                label,
                "mss_bull_choch_idx",
            ] = pending_bull

            out.at[
                label,
                "mss_bull_confirmation_idx",
            ] = i

            out.at[
                label,
                "mss_bull_confirmation_lag",
            ] = i - pending_bull

            pending_bull = None

        # -------------------------------------------------------------
        # Confirmation MSS bearish.
        # -------------------------------------------------------------

        if (
            pending_bear is not None
            and bool(row["displacement_bear"])
        ):

            out.at[label, "mss_bear"] = True

            out.at[
                label,
                "mss_bear_choch_idx",
            ] = pending_bear

            out.at[
                label,
                "mss_bear_confirmation_idx",
            ] = i

            out.at[
                label,
                "mss_bear_confirmation_lag",
            ] = i - pending_bear

            pending_bear = None

    return out
