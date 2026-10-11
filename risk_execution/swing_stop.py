from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwingStopParams:
    """
    Swing Stop v0.1.

    Bullish candidate:
        SL derrière le dernier swing low confirmé.

    Bearish candidate:
        SL derrière le dernier swing high confirmé.

    buffer_mode:
        "none"
        "atr"
        "percent"
        "absolute"

    buffer:
        distance supplémentaire derrière le swing.

    max_swing_age_bars:
        ignore les swings trop anciens.

    atr_column:
        ATR utilisé pour normalisation / buffer.
    """

    buffer_mode: str = "none"
    buffer: float = 0.0

    max_swing_age_bars: int | None = None

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.buffer_mode not in {
            "none",
            "atr",
            "percent",
            "absolute",
        }:
            raise ValueError(
                "buffer_mode doit être none, atr, percent ou absolute"
            )

        if self.buffer < 0:
            raise ValueError(
                "buffer doit être >= 0"
            )

        if (
            self.max_swing_age_bars is not None
            and self.max_swing_age_bars < 0
        ):
            raise ValueError(
                "max_swing_age_bars doit être >= 0 ou None"
            )


def detect_swing_stops(
    df: pd.DataFrame,
    params: SwingStopParams = SwingStopParams(),
) -> pd.DataFrame:

    required = {
        "close",

        "swing_high",
        "swing_low",

        "swing_high_price",
        "swing_low_price",

        params.atr_column,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_swing_stops(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    close = out[
        "close"
    ].to_numpy(dtype=np.float64)

    atr = out[
        params.atr_column
    ].to_numpy(dtype=np.float64)

    swing_high = out[
        "swing_high"
    ].to_numpy(dtype=bool)

    swing_low = out[
        "swing_low"
    ].to_numpy(dtype=bool)

    swing_high_price = out[
        "swing_high_price"
    ].to_numpy(dtype=np.float64)

    swing_low_price = out[
        "swing_low_price"
    ].to_numpy(dtype=np.float64)

    out["swing_stop_bull"] = np.nan
    out["swing_stop_bear"] = np.nan

    out["swing_stop_bull_available"] = False
    out["swing_stop_bear_available"] = False

    out["swing_stop_bull_risk"] = np.nan
    out["swing_stop_bear_risk"] = np.nan

    out["swing_stop_bull_risk_atr"] = np.nan
    out["swing_stop_bear_risk_atr"] = np.nan

    for prefix in (
        "swing_stop_bull",
        "swing_stop_bear",
    ):

        out[
            f"{prefix}_source_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_age_bars"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    latest_high = None
    latest_low = None

    def buffer_value(
        price: float,
        atr_value: float,
    ) -> float:

        if params.buffer_mode == "none":
            return 0.0

        if params.buffer_mode == "absolute":
            return params.buffer

        if params.buffer_mode == "percent":
            return (
                price
                * params.buffer
            )

        if (
            np.isfinite(atr_value)
            and atr_value > 0
        ):
            return (
                atr_value
                * params.buffer
            )

        return np.nan

    for i in range(n):

        # Swing booleans represent CONFIRMED swings,
        # so updating state here remains causal.

        if swing_high[i]:

            latest_high = {
                "idx":
                    i,

                "price":
                    float(
                        swing_high_price[i]
                    ),
            }

        if swing_low[i]:

            latest_low = {
                "idx":
                    i,

                "price":
                    float(
                        swing_low_price[i]
                    ),
            }

        label = out.index[i]

        # =====================================================================
        # BULL STOP
        # =====================================================================

        if latest_low is not None:

            age = (
                i
                - latest_low["idx"]
            )

            age_ok = (
                params.max_swing_age_bars is None
                or age
                <= params.max_swing_age_bars
            )

            if age_ok:

                buf = buffer_value(
                    latest_low["price"],
                    atr[i],
                )

                if np.isfinite(buf):

                    stop = (
                        latest_low["price"]
                        - buf
                    )

                    risk = (
                        close[i]
                        - stop
                    )

                    if risk > 0:

                        out.at[
                            label,
                            "swing_stop_bull",
                        ] = stop

                        out.at[
                            label,
                            "swing_stop_bull_available",
                        ] = True

                        out.at[
                            label,
                            "swing_stop_bull_risk",
                        ] = risk

                        out.at[
                            label,
                            "swing_stop_bull_source_idx",
                        ] = latest_low[
                            "idx"
                        ]

                        out.at[
                            label,
                            "swing_stop_bull_age_bars",
                        ] = age

                        if (
                            np.isfinite(atr[i])
                            and atr[i] > 0
                        ):

                            out.at[
                                label,
                                "swing_stop_bull_risk_atr",
                            ] = (
                                risk / atr[i]
                            )

        # =====================================================================
        # BEAR STOP
        # =====================================================================

        if latest_high is not None:

            age = (
                i
                - latest_high["idx"]
            )

            age_ok = (
                params.max_swing_age_bars is None
                or age
                <= params.max_swing_age_bars
            )

            if age_ok:

                buf = buffer_value(
                    latest_high["price"],
                    atr[i],
                )

                if np.isfinite(buf):

                    stop = (
                        latest_high["price"]
                        + buf
                    )

                    risk = (
                        stop
                        - close[i]
                    )

                    if risk > 0:

                        out.at[
                            label,
                            "swing_stop_bear",
                        ] = stop

                        out.at[
                            label,
                            "swing_stop_bear_available",
                        ] = True

                        out.at[
                            label,
                            "swing_stop_bear_risk",
                        ] = risk

                        out.at[
                            label,
                            "swing_stop_bear_source_idx",
                        ] = latest_high[
                            "idx"
                        ]

                        out.at[
                            label,
                            "swing_stop_bear_age_bars",
                        ] = age

                        if (
                            np.isfinite(atr[i])
                            and atr[i] > 0
                        ):

                            out.at[
                                label,
                                "swing_stop_bear_risk_atr",
                            ] = (
                                risk / atr[i]
                            )

    return out
