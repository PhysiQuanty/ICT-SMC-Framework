from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BISISIBIParams:
    """
    BISI / SIBI comme classification directionnelle des FVG.

    BISI:
        Buy-Side Imbalance / Sell-Side Inefficiency
        -> FVG bullish

    SIBI:
        Sell-Side Imbalance / Buy-Side Inefficiency
        -> FVG bearish

    Cette feature ne recalcule pas les FVG.
    Elle dépend explicitement de fvg.py.
    """

    enabled: bool = True


def detect_bisi_sibi(
    df: pd.DataFrame,
    params: BISISIBIParams = BISISIBIParams(),
) -> pd.DataFrame:

    required = {
        "fvg_bull",
        "fvg_bear",

        "fvg_bull_low",
        "fvg_bull_high",
        "fvg_bull_mid",

        "fvg_bear_low",
        "fvg_bear_high",
        "fvg_bear_mid",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"detect_bisi_sibi(): colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    if not params.enabled:
        out["bisi"] = False
        out["sibi"] = False
        return out

    out["bisi"] = out["fvg_bull"].astype(bool)
    out["sibi"] = out["fvg_bear"].astype(bool)

    out["bisi_low"] = out["fvg_bull_low"]
    out["bisi_high"] = out["fvg_bull_high"]
    out["bisi_mid"] = out["fvg_bull_mid"]

    out["sibi_low"] = out["fvg_bear_low"]
    out["sibi_high"] = out["fvg_bear_high"]
    out["sibi_mid"] = out["fvg_bear_mid"]

    return out
