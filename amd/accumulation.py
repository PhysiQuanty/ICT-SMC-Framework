from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smc_framework.time._utils import (
    in_clock_window,
    local_datetime,
    minute_of_day,
)


@dataclass(frozen=True)
class AccumulationParams:
    """
    Accumulation v0.1.

    Proxy:
        range construit causalement pendant une fenêtre temporelle.

    Default:
        00:00 -> 02:00 America/New_York.

    Une accumulation n'est "completed" qu'après la fin
    de la fenêtre.

    min_bars:
        nombre minimal de bougies.

    max_range_atr:
        largeur maximale du range en ATR pour considérer
        la fenêtre comme compression / accumulation.

        None => aucun filtre.

    ATR utilisé au moment de la clôture de l'accumulation.
    """

    timezone: str = "America/New_York"

    start: str = "00:00"
    end: str = "02:00"

    min_bars: int = 2

    max_range_atr: float | None = 3.0

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.min_bars < 1:
            raise ValueError(
                "min_bars doit être >= 1"
            )

        if (
            self.max_range_atr is not None
            and self.max_range_atr <= 0
        ):
            raise ValueError(
                "max_range_atr doit être > 0 ou None"
            )


def detect_accumulation(
    df: pd.DataFrame,
    params: AccumulationParams = AccumulationParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
        params.atr_column,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_accumulation(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    dt = local_datetime(
        out,
        params.timezone,
    )

    minutes = minute_of_day(
        dt
    )

    active = in_clock_window(
        minutes,
        params.start,
        params.end,
    )

    high = out[
        "high"
    ].to_numpy(dtype=np.float64)

    low = out[
        "low"
    ].to_numpy(dtype=np.float64)

    atr = out[
        params.atr_column
    ].to_numpy(dtype=np.float64)

    out["accumulation_active"] = active

    out["accumulation_current_high"] = np.nan
    out["accumulation_current_low"] = np.nan
    out["accumulation_current_mid"] = np.nan
    out["accumulation_current_range"] = np.nan

    out["accumulation_completed"] = False
    out["accumulation_valid"] = False

    out["accumulation_high"] = np.nan
    out["accumulation_low"] = np.nan
    out["accumulation_mid"] = np.nan
    out["accumulation_range"] = np.nan
    out["accumulation_range_atr"] = np.nan

    for suffix in (
        "start_idx",
        "end_idx",
        "confirmation_idx",
        "bars",
        "age_bars",
    ):

        out[
            f"accumulation_{suffix}"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    current_high = None
    current_low = None

    current_start = None
    current_end = None
    current_bars = 0

    completed = None

    was_active = False

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # SESSION START / UPDATE
        # =====================================================================

        if active[i]:

            if not was_active:

                current_high = high[i]
                current_low = low[i]

                current_start = i
                current_end = i
                current_bars = 1

            else:

                current_high = max(
                    current_high,
                    high[i],
                )

                current_low = min(
                    current_low,
                    low[i],
                )

                current_end = i
                current_bars += 1

            out.at[
                label,
                "accumulation_current_high",
            ] = current_high

            out.at[
                label,
                "accumulation_current_low",
            ] = current_low

            out.at[
                label,
                "accumulation_current_mid",
            ] = (
                current_high
                + current_low
            ) / 2.0

            out.at[
                label,
                "accumulation_current_range",
            ] = (
                current_high
                - current_low
            )

        # =====================================================================
        # SESSION JUST ENDED
        # =====================================================================

        if (
            was_active
            and not active[i]
            and current_high is not None
        ):

            range_size = (
                current_high
                - current_low
            )

            range_atr = (
                range_size / atr[i]
                if (
                    np.isfinite(atr[i])
                    and atr[i] > 0
                )
                else np.nan
            )

            valid = (
                current_bars
                >= params.min_bars
            )

            if (
                params.max_range_atr
                is not None
            ):

                valid = (
                    valid
                    and np.isfinite(
                        range_atr
                    )
                    and (
                        range_atr
                        <= params.max_range_atr
                    )
                )

            completed = {
                "high":
                    current_high,

                "low":
                    current_low,

                "mid":
                    (
                        current_high
                        + current_low
                    ) / 2.0,

                "range":
                    range_size,

                "range_atr":
                    range_atr,

                "start_idx":
                    current_start,

                "end_idx":
                    current_end,

                "confirmation_idx":
                    i,

                "bars":
                    current_bars,

                "valid":
                    valid,
            }

            out.at[
                label,
                "accumulation_completed",
            ] = True

            current_high = None
            current_low = None

            current_start = None
            current_end = None
            current_bars = 0

        # =====================================================================
        # COMPLETED STATE
        # =====================================================================

        if completed is not None:

            out.at[
                label,
                "accumulation_valid",
            ] = completed[
                "valid"
            ]

            out.at[
                label,
                "accumulation_high",
            ] = completed[
                "high"
            ]

            out.at[
                label,
                "accumulation_low",
            ] = completed[
                "low"
            ]

            out.at[
                label,
                "accumulation_mid",
            ] = completed[
                "mid"
            ]

            out.at[
                label,
                "accumulation_range",
            ] = completed[
                "range"
            ]

            out.at[
                label,
                "accumulation_range_atr",
            ] = completed[
                "range_atr"
            ]

            out.at[
                label,
                "accumulation_start_idx",
            ] = completed[
                "start_idx"
            ]

            out.at[
                label,
                "accumulation_end_idx",
            ] = completed[
                "end_idx"
            ]

            out.at[
                label,
                "accumulation_confirmation_idx",
            ] = completed[
                "confirmation_idx"
            ]

            out.at[
                label,
                "accumulation_bars",
            ] = completed[
                "bars"
            ]

            out.at[
                label,
                "accumulation_age_bars",
            ] = (
                i
                - completed[
                    "confirmation_idx"
                ]
            )

        was_active = bool(
            active[i]
        )

    return out
