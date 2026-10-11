from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskRewardParams:
    """
    Risk / Reward barrier geometry v0.1.

    RR target:

        bullish:
            target = entry + RR * risk

        bearish:
            target = entry - RR * risk

    entry_mode:
        "close"
            utilise le close courant.

        "columns"
            utilise entry_bull_column / entry_bear_column.

    stop columns:
        par défaut les swing stops.

    rr_values:
        barrières exposées indépendamment.
    """

    rr_values: tuple[float, ...] = (
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
    )

    entry_mode: str = "close"

    entry_bull_column: str | None = None
    entry_bear_column: str | None = None

    stop_bull_column: str = "swing_stop_bull"
    stop_bear_column: str = "swing_stop_bear"

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.entry_mode not in {
            "close",
            "columns",
        }:
            raise ValueError(
                "entry_mode doit être close ou columns"
            )

        if (
            self.entry_mode == "columns"
            and (
                self.entry_bull_column is None
                or self.entry_bear_column is None
            )
        ):
            raise ValueError(
                "entry_bull_column et entry_bear_column "
                "requis avec entry_mode='columns'"
            )

        if not self.rr_values:
            raise ValueError(
                "rr_values ne peut pas être vide"
            )

        for rr in self.rr_values:

            if rr <= 0:
                raise ValueError(
                    "Tous les RR doivent être > 0"
                )


def _rr_name(
    rr: float,
) -> str:

    text = (
        f"{rr:g}"
        .replace(".", "_")
    )

    return text


def detect_risk_reward_levels(
    df: pd.DataFrame,
    params: RiskRewardParams = RiskRewardParams(),
) -> pd.DataFrame:

    required = {
        params.stop_bull_column,
        params.stop_bear_column,
        params.atr_column,
    }

    if params.entry_mode == "close":

        required.add(
            "close"
        )

    else:

        required.update(
            {
                params.entry_bull_column,
                params.entry_bear_column,
            }
        )

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_risk_reward_levels(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    if params.entry_mode == "close":

        bull_entry = out[
            "close"
        ].to_numpy(
            dtype=np.float64,
        )

        bear_entry = (
            bull_entry.copy()
        )

    else:

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

    bull_stop = out[
        params.stop_bull_column
    ].to_numpy(
        dtype=np.float64,
    )

    bear_stop = out[
        params.stop_bear_column
    ].to_numpy(
        dtype=np.float64,
    )

    atr = out[
        params.atr_column
    ].to_numpy(
        dtype=np.float64,
    )

    bull_risk = (
        bull_entry
        - bull_stop
    )

    bear_risk = (
        bear_stop
        - bear_entry
    )

    bull_valid = (
        np.isfinite(bull_entry)
        & np.isfinite(bull_stop)
        & (bull_risk > 0)
    )

    bear_valid = (
        np.isfinite(bear_entry)
        & np.isfinite(bear_stop)
        & (bear_risk > 0)
    )

    out[
        "rr_bull_valid"
    ] = bull_valid

    out[
        "rr_bear_valid"
    ] = bear_valid

    out[
        "rr_bull_entry"
    ] = np.where(
        bull_valid,
        bull_entry,
        np.nan,
    )

    out[
        "rr_bear_entry"
    ] = np.where(
        bear_valid,
        bear_entry,
        np.nan,
    )

    out[
        "rr_bull_stop"
    ] = np.where(
        bull_valid,
        bull_stop,
        np.nan,
    )

    out[
        "rr_bear_stop"
    ] = np.where(
        bear_valid,
        bear_stop,
        np.nan,
    )

    out[
        "rr_bull_risk"
    ] = np.where(
        bull_valid,
        bull_risk,
        np.nan,
    )

    out[
        "rr_bear_risk"
    ] = np.where(
        bear_valid,
        bear_risk,
        np.nan,
    )

    valid_atr = (
        np.isfinite(atr)
        & (atr > 0)
    )

    out[
        "rr_bull_risk_atr"
    ] = np.where(
        bull_valid
        & valid_atr,
        bull_risk / atr,
        np.nan,
    )

    out[
        "rr_bear_risk_atr"
    ] = np.where(
        bear_valid
        & valid_atr,
        bear_risk / atr,
        np.nan,
    )

    for rr in params.rr_values:

        suffix = _rr_name(
            rr
        )

        out[
            f"rr_bull_target_{suffix}"
        ] = np.where(
            bull_valid,
            (
                bull_entry
                + rr * bull_risk
            ),
            np.nan,
        )

        out[
            f"rr_bear_target_{suffix}"
        ] = np.where(
            bear_valid,
            (
                bear_entry
                - rr * bear_risk
            ),
            np.nan,
        )

    return out
