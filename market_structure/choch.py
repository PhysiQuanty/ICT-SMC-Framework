from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CHoCHParams:
    """
    CHoCH bullish:
        cassure d'un Protected High.

    CHoCH bearish:
        cassure d'un Protected Low.
    """

    enabled: bool = True


def detect_choch(
    df: pd.DataFrame,
    params: CHoCHParams = CHoCHParams(),
) -> pd.DataFrame:

    required = {
        "protected_low_broken",
        "protected_high_broken",
        "protected_low_price",
        "protected_high_price",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"detect_choch(): colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["choch_bull"] = False
    out["choch_bear"] = False

    out["choch_bull_level"] = np.nan
    out["choch_bear_level"] = np.nan

    out["choch_bull_confirmation_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["choch_bear_confirmation_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    if not params.enabled:
        return out

    for i in range(n):

        label = out.index[i]

        if bool(out.iloc[i]["protected_high_broken"]):

            out.at[label, "choch_bull"] = True
            out.at[label, "choch_bull_level"] = out.iloc[i][
                "protected_high_price"
            ]
            out.at[label, "choch_bull_confirmation_idx"] = i

        if bool(out.iloc[i]["protected_low_broken"]):

            out.at[label, "choch_bear"] = True
            out.at[label, "choch_bear_level"] = out.iloc[i][
                "protected_low_price"
            ]
            out.at[label, "choch_bear_confirmation_idx"] = i

    return out
