from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FVGParams:
    """
    Fair Value Gap OHLC sur 3 bougies.

    Bullish FVG à i:
        high[i-2] < low[i]

    Bearish FVG à i:
        low[i-2] > high[i]

    La feature est confirmée à la troisième bougie i.

    min_gap_atr:
        taille minimale du gap en ATR.

    atr_column:
        colonne ATR utilisée pour normalisation.

    mitigation_mode:
        "touch" -> premier contact avec la zone
        "mid"   -> mitigation quand CE / midpoint atteint
        "full"  -> zone entièrement traversée

    Une FVG est une zone persistante avec état:
        active
        first_touch
        fill_ratio
        mitigated
    """

    min_gap_atr: float = 0.0
    atr_column: str = "atr"
    mitigation_mode: str = "full"

    def __post_init__(self) -> None:
        if self.min_gap_atr < 0:
            raise ValueError("min_gap_atr doit être >= 0")

        if self.mitigation_mode not in {"touch", "mid", "full"}:
            raise ValueError(
                "mitigation_mode doit être 'touch', 'mid' ou 'full'"
            )


def detect_fvg(
    df: pd.DataFrame,
    params: FVGParams = FVGParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"detect_fvg(): colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["fvg_bull"] = False
    out["fvg_bear"] = False

    out["fvg_bull_low"] = np.nan
    out["fvg_bull_high"] = np.nan
    out["fvg_bull_mid"] = np.nan

    out["fvg_bear_low"] = np.nan
    out["fvg_bear_high"] = np.nan
    out["fvg_bear_mid"] = np.nan

    out["fvg_bull_size"] = np.nan
    out["fvg_bear_size"] = np.nan

    out["fvg_bull_size_atr"] = np.nan
    out["fvg_bear_size_atr"] = np.nan

    out["fvg_bull_active"] = False
    out["fvg_bear_active"] = False

    out["fvg_bull_mitigated"] = False
    out["fvg_bear_mitigated"] = False

    out["fvg_bull_fill_ratio"] = np.nan
    out["fvg_bear_fill_ratio"] = np.nan

    for prefix in ("fvg_bull", "fvg_bear"):

        out[f"{prefix}_origin_idx"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

        out[f"{prefix}_confirmation_idx"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

        out[f"{prefix}_age_bars"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

        out[f"{prefix}_first_touch_idx"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)

    if params.atr_column in out.columns:
        atr = out[params.atr_column].to_numpy(dtype=np.float64)
    else:
        atr = np.full(n, np.nan)

    active_bull = []
    active_bear = []

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # UPDATE EXISTING BULLISH FVGs
        # =====================================================================

        still_active = []

        for zone in active_bull:

            zone["age"] += 1

            if low[i] <= zone["high"]:

                if zone["first_touch_idx"] is None:
                    zone["first_touch_idx"] = i

                penetration = zone["high"] - low[i]

                fill_ratio = np.clip(
                    penetration / zone["size"],
                    0.0,
                    1.0,
                )

                zone["fill_ratio"] = max(
                    zone["fill_ratio"],
                    fill_ratio,
                )

            if params.mitigation_mode == "touch":
                mitigated = (
                    zone["first_touch_idx"]
                    is not None
                )

            elif params.mitigation_mode == "mid":
                mitigated = (
                    low[i] <= zone["mid"]
                )

            else:
                mitigated = (
                    low[i] <= zone["low"]
                )

            if mitigated:
                zone["active"] = False
                zone["mitigated_idx"] = i

            else:
                still_active.append(zone)

        active_bull = still_active

        # =====================================================================
        # UPDATE EXISTING BEARISH FVGs
        # =====================================================================

        still_active = []

        for zone in active_bear:

            zone["age"] += 1

            if high[i] >= zone["low"]:

                if zone["first_touch_idx"] is None:
                    zone["first_touch_idx"] = i

                penetration = high[i] - zone["low"]

                fill_ratio = np.clip(
                    penetration / zone["size"],
                    0.0,
                    1.0,
                )

                zone["fill_ratio"] = max(
                    zone["fill_ratio"],
                    fill_ratio,
                )

            if params.mitigation_mode == "touch":
                mitigated = (
                    zone["first_touch_idx"]
                    is not None
                )

            elif params.mitigation_mode == "mid":
                mitigated = (
                    high[i] >= zone["mid"]
                )

            else:
                mitigated = (
                    high[i] >= zone["high"]
                )

            if mitigated:
                zone["active"] = False
                zone["mitigated_idx"] = i

            else:
                still_active.append(zone)

        active_bear = still_active

        # =====================================================================
        # NEW FVG
        # =====================================================================

        if i >= 2:

            # -----------------------------------------------------------------
            # Bullish FVG
            # -----------------------------------------------------------------

            if high[i - 2] < low[i]:

                zone_low = high[i - 2]
                zone_high = low[i]

                size = zone_high - zone_low

                size_atr = (
                    size / atr[i]
                    if (
                        np.isfinite(atr[i])
                        and atr[i] > 0
                    )
                    else np.nan
                )

                threshold_ok = (
                    params.min_gap_atr == 0
                    or (
                        np.isfinite(size_atr)
                        and size_atr >= params.min_gap_atr
                    )
                )

                if threshold_ok:

                    mid = (
                        zone_low
                        + zone_high
                    ) / 2.0

                    out.at[label, "fvg_bull"] = True
                    out.at[label, "fvg_bull_low"] = zone_low
                    out.at[label, "fvg_bull_high"] = zone_high
                    out.at[label, "fvg_bull_mid"] = mid
                    out.at[label, "fvg_bull_size"] = size
                    out.at[label, "fvg_bull_size_atr"] = size_atr
                    out.at[label, "fvg_bull_active"] = True
                    out.at[label, "fvg_bull_fill_ratio"] = 0.0
                    out.at[label, "fvg_bull_origin_idx"] = i - 2
                    out.at[label, "fvg_bull_confirmation_idx"] = i
                    out.at[label, "fvg_bull_age_bars"] = 0

                    active_bull.append(
                        {
                            "low": zone_low,
                            "high": zone_high,
                            "mid": mid,
                            "size": size,
                            "confirmation_idx": i,
                            "age": 0,
                            "fill_ratio": 0.0,
                            "first_touch_idx": None,
                            "active": True,
                        }
                    )

            # -----------------------------------------------------------------
            # Bearish FVG
            # -----------------------------------------------------------------

            if low[i - 2] > high[i]:

                zone_low = high[i]
                zone_high = low[i - 2]

                size = zone_high - zone_low

                size_atr = (
                    size / atr[i]
                    if (
                        np.isfinite(atr[i])
                        and atr[i] > 0
                    )
                    else np.nan
                )

                threshold_ok = (
                    params.min_gap_atr == 0
                    or (
                        np.isfinite(size_atr)
                        and size_atr >= params.min_gap_atr
                    )
                )

                if threshold_ok:

                    mid = (
                        zone_low
                        + zone_high
                    ) / 2.0

                    out.at[label, "fvg_bear"] = True
                    out.at[label, "fvg_bear_low"] = zone_low
                    out.at[label, "fvg_bear_high"] = zone_high
                    out.at[label, "fvg_bear_mid"] = mid
                    out.at[label, "fvg_bear_size"] = size
                    out.at[label, "fvg_bear_size_atr"] = size_atr
                    out.at[label, "fvg_bear_active"] = True
                    out.at[label, "fvg_bear_fill_ratio"] = 0.0
                    out.at[label, "fvg_bear_origin_idx"] = i - 2
                    out.at[label, "fvg_bear_confirmation_idx"] = i
                    out.at[label, "fvg_bear_age_bars"] = 0

                    active_bear.append(
                        {
                            "low": zone_low,
                            "high": zone_high,
                            "mid": mid,
                            "size": size,
                            "confirmation_idx": i,
                            "age": 0,
                            "fill_ratio": 0.0,
                            "first_touch_idx": None,
                            "active": True,
                        }
                    )

    return out
