from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BOSParams:
    """
    Break Of Structure.

    break_mode:
        "close" -> cassure confirmée par le close
        "wick"  -> cassure confirmée par high/low

    min_break_atr:
        profondeur minimale de cassure exprimée en ATR.

    atr_column:
        colonne ATR utilisée pour normaliser la cassure.

    consume_level:
        si True, un swing ne peut produire qu'un seul BOS.
    """

    break_mode: str = "close"
    min_break_atr: float = 0.0
    atr_column: str = "atr"
    consume_level: bool = True

    def __post_init__(self) -> None:
        if self.break_mode not in {"close", "wick"}:
            raise ValueError("break_mode doit être 'close' ou 'wick'")

        if self.min_break_atr < 0:
            raise ValueError("min_break_atr doit être >= 0")


def detect_bos(
    df: pd.DataFrame,
    params: BOSParams = BOSParams(),
) -> pd.DataFrame:
    """
    Détecte les cassures des derniers Swing High / Swing Low confirmés.

    Dépendances attendues:
        swing_high
        swing_low
        swing_high_price
        swing_low_price
        swing_high_origin_idx
        swing_low_origin_idx

    La feature ne détermine PAS la tendance.
    Elle dit seulement qu'un niveau structurel confirmé a été cassé.
    """

    required = {
        "high",
        "low",
        "close",
        "swing_high",
        "swing_low",
        "swing_high_price",
        "swing_low_price",
        "swing_high_origin_idx",
        "swing_low_origin_idx",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"detect_bos(): colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["bos_bull"] = False
    out["bos_bear"] = False

    for prefix in ("bos_bull", "bos_bear"):
        out[f"{prefix}_level"] = np.nan
        out[f"{prefix}_break_price"] = np.nan
        out[f"{prefix}_break_distance"] = np.nan
        out[f"{prefix}_break_distance_atr"] = np.nan

        out[f"{prefix}_source_origin_idx"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

        out[f"{prefix}_source_confirmation_idx"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

        out[f"{prefix}_confirmation_idx"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

        out[f"{prefix}_source_age_bars"] = pd.Series(
            pd.array([pd.NA] * n, dtype="Int64"),
            index=out.index,
        )

    highs = out["high"].to_numpy(dtype=np.float64)
    lows = out["low"].to_numpy(dtype=np.float64)
    closes = out["close"].to_numpy(dtype=np.float64)

    if params.atr_column in out.columns:
        atr = out[params.atr_column].to_numpy(dtype=np.float64)
    else:
        atr = np.full(n, np.nan)

    latest_high = None
    latest_low = None

    for i in range(n):

        # -------------------------------------------------------------
        # Nouveaux swings disponibles à CETTE bougie.
        # -------------------------------------------------------------

        if bool(out.iloc[i]["swing_high"]):
            latest_high = {
                "price": float(out.iloc[i]["swing_high_price"]),
                "origin_idx": int(out.iloc[i]["swing_high_origin_idx"]),
                "confirmation_idx": i,
                "consumed": False,
            }

        if bool(out.iloc[i]["swing_low"]):
            latest_low = {
                "price": float(out.iloc[i]["swing_low_price"]),
                "origin_idx": int(out.iloc[i]["swing_low_origin_idx"]),
                "confirmation_idx": i,
                "consumed": False,
            }

        label = out.index[i]

        # -------------------------------------------------------------
        # Bullish BOS
        # -------------------------------------------------------------

        if latest_high is not None:
            if not (params.consume_level and latest_high["consumed"]):

                break_price = (
                    closes[i]
                    if params.break_mode == "close"
                    else highs[i]
                )

                distance = break_price - latest_high["price"]

                atr_distance = (
                    distance / atr[i]
                    if np.isfinite(atr[i]) and atr[i] > 0
                    else np.nan
                )

                threshold_ok = (
                    distance > 0
                    and (
                        params.min_break_atr == 0
                        or (
                            np.isfinite(atr_distance)
                            and atr_distance >= params.min_break_atr
                        )
                    )
                )

                if threshold_ok:
                    out.at[label, "bos_bull"] = True
                    out.at[label, "bos_bull_level"] = latest_high["price"]
                    out.at[label, "bos_bull_break_price"] = break_price
                    out.at[label, "bos_bull_break_distance"] = distance
                    out.at[label, "bos_bull_break_distance_atr"] = atr_distance
                    out.at[label, "bos_bull_source_origin_idx"] = latest_high["origin_idx"]
                    out.at[label, "bos_bull_source_confirmation_idx"] = latest_high["confirmation_idx"]
                    out.at[label, "bos_bull_confirmation_idx"] = i
                    out.at[label, "bos_bull_source_age_bars"] = (
                        i - latest_high["confirmation_idx"]
                    )

                    latest_high["consumed"] = True

        # -------------------------------------------------------------
        # Bearish BOS
        # -------------------------------------------------------------

        if latest_low is not None:
            if not (params.consume_level and latest_low["consumed"]):

                break_price = (
                    closes[i]
                    if params.break_mode == "close"
                    else lows[i]
                )

                distance = latest_low["price"] - break_price

                atr_distance = (
                    distance / atr[i]
                    if np.isfinite(atr[i]) and atr[i] > 0
                    else np.nan
                )

                threshold_ok = (
                    distance > 0
                    and (
                        params.min_break_atr == 0
                        or (
                            np.isfinite(atr_distance)
                            and atr_distance >= params.min_break_atr
                        )
                    )
                )

                if threshold_ok:
                    out.at[label, "bos_bear"] = True
                    out.at[label, "bos_bear_level"] = latest_low["price"]
                    out.at[label, "bos_bear_break_price"] = break_price
                    out.at[label, "bos_bear_break_distance"] = distance
                    out.at[label, "bos_bear_break_distance_atr"] = atr_distance
                    out.at[label, "bos_bear_source_origin_idx"] = latest_low["origin_idx"]
                    out.at[label, "bos_bear_source_confirmation_idx"] = latest_low["confirmation_idx"]
                    out.at[label, "bos_bear_confirmation_idx"] = i
                    out.at[label, "bos_bear_source_age_bars"] = (
                        i - latest_low["confirmation_idx"]
                    )

                    latest_low["consumed"] = True

    return out
