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
class JudasSwingParams:
    """
    Judas Swing proxy v0.1.

    On utilise un niveau d'open connu
    (par défaut midnight_open).

    Pendant une fenêtre donnée:

        Judas High:
            excursion au-dessus de l'open,
            retour sous l'open,
            puis displacement bearish dans max_reversal_lag.

        Judas Low:
            excursion sous l'open,
            retour au-dessus,
            puis displacement bullish.

    Bullish Judas:
        faux mouvement baissier -> reversal bullish.

    Bearish Judas:
        faux mouvement haussier -> reversal bearish.

    Ce n'est qu'un proxy observable du concept.
    """

    timezone: str = "America/New_York"

    start: str = "00:00"
    end: str = "05:00"

    open_column: str = "midnight_open"

    min_excursion_atr: float = 0.0

    max_reversal_lag: int = 5

    atr_column: str = "atr"

    require_displacement: bool = True

    def __post_init__(self) -> None:

        if self.min_excursion_atr < 0:
            raise ValueError(
                "min_excursion_atr doit être >= 0"
            )

        if self.max_reversal_lag < 0:
            raise ValueError(
                "max_reversal_lag doit être >= 0"
            )


def detect_judas_swing(
    df: pd.DataFrame,
    params: JudasSwingParams = JudasSwingParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",

        params.open_column,
        params.atr_column,
    }

    if params.require_displacement:

        required.update(
            {
                "displacement_bull",
                "displacement_bear",
            }
        )

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_judas_swing(): "
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

    window = in_clock_window(
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

    close = out[
        "close"
    ].to_numpy(dtype=np.float64)

    reference = out[
        params.open_column
    ].to_numpy(dtype=np.float64)

    atr = out[
        params.atr_column
    ].to_numpy(dtype=np.float64)

    if params.require_displacement:

        disp_bull = out[
            "displacement_bull"
        ].to_numpy(dtype=bool)

        disp_bear = out[
            "displacement_bear"
        ].to_numpy(dtype=bool)

    else:

        disp_bull = np.ones(
            n,
            dtype=bool,
        )

        disp_bear = np.ones(
            n,
            dtype=bool,
        )

    out["judas_window_active"] = window

    out["judas_bull"] = False
    out["judas_bear"] = False

    out["judas_low_excursion"] = np.nan
    out["judas_high_excursion"] = np.nan

    out["judas_low_excursion_atr"] = np.nan
    out["judas_high_excursion_atr"] = np.nan

    out["judas_reference_level"] = np.nan

    for side in (
        "judas_bull",
        "judas_bear",
    ):

        for suffix in (
            "excursion_idx",
            "confirmation_idx",
            "lag",
        ):

            out[
                f"{side}_{suffix}"
            ] = pd.Series(
                pd.array(
                    [pd.NA] * n,
                    dtype="Int64",
                ),
                index=out.index,
            )

    pending_bull = None
    pending_bear = None

    for i in range(n):

        # Outside window => clear pending state.
        if not window[i]:

            pending_bull = None
            pending_bear = None

            continue

        if not np.isfinite(
            reference[i]
        ):
            continue

        level = reference[i]

        # =====================================================================
        # LOW-SIDE FALSE BREAK
        # candidate bullish Judas
        # =====================================================================

        low_excursion = (
            level - low[i]
        )

        low_excursion_atr = (
            low_excursion / atr[i]
            if (
                np.isfinite(atr[i])
                and atr[i] > 0
            )
            else np.nan
        )

        low_ok = (
            low_excursion > 0
            and close[i] > level
            and (
                params.min_excursion_atr == 0
                or (
                    np.isfinite(
                        low_excursion_atr
                    )
                    and low_excursion_atr
                    >= params.min_excursion_atr
                )
            )
        )

        if low_ok:

            pending_bull = {
                "idx":
                    i,

                "reference":
                    level,

                "excursion":
                    low_excursion,

                "excursion_atr":
                    low_excursion_atr,
            }

        # =====================================================================
        # HIGH-SIDE FALSE BREAK
        # candidate bearish Judas
        # =====================================================================

        high_excursion = (
            high[i] - level
        )

        high_excursion_atr = (
            high_excursion / atr[i]
            if (
                np.isfinite(atr[i])
                and atr[i] > 0
            )
            else np.nan
        )

        high_ok = (
            high_excursion > 0
            and close[i] < level
            and (
                params.min_excursion_atr == 0
                or (
                    np.isfinite(
                        high_excursion_atr
                    )
                    and high_excursion_atr
                    >= params.min_excursion_atr
                )
            )
        )

        if high_ok:

            pending_bear = {
                "idx":
                    i,

                "reference":
                    level,

                "excursion":
                    high_excursion,

                "excursion_atr":
                    high_excursion_atr,
            }

        # =====================================================================
        # EXPIRE
        # =====================================================================

        if (
            pending_bull is not None
            and (
                i - pending_bull["idx"]
                > params.max_reversal_lag
            )
        ):

            pending_bull = None

        if (
            pending_bear is not None
            and (
                i - pending_bear["idx"]
                > params.max_reversal_lag
            )
        ):

            pending_bear = None

        # =====================================================================
        # BULL CONFIRMATION
        # =====================================================================

        if (
            pending_bull is not None
            and i >= pending_bull["idx"]
            and disp_bull[i]
        ):

            label = out.index[i]

            out.at[
                label,
                "judas_bull",
            ] = True

            out.at[
                label,
                "judas_low_excursion",
            ] = pending_bull[
                "excursion"
            ]

            out.at[
                label,
                "judas_low_excursion_atr",
            ] = pending_bull[
                "excursion_atr"
            ]

            out.at[
                label,
                "judas_reference_level",
            ] = pending_bull[
                "reference"
            ]

            out.at[
                label,
                "judas_bull_excursion_idx",
            ] = pending_bull[
                "idx"
            ]

            out.at[
                label,
                "judas_bull_confirmation_idx",
            ] = i

            out.at[
                label,
                "judas_bull_lag",
            ] = (
                i
                - pending_bull[
                    "idx"
                ]
            )

            pending_bull = None

        # =====================================================================
        # BEAR CONFIRMATION
        # =====================================================================

        if (
            pending_bear is not None
            and i >= pending_bear["idx"]
            and disp_bear[i]
        ):

            label = out.index[i]

            out.at[
                label,
                "judas_bear",
            ] = True

            out.at[
                label,
                "judas_high_excursion",
            ] = pending_bear[
                "excursion"
            ]

            out.at[
                label,
                "judas_high_excursion_atr",
            ] = pending_bear[
                "excursion_atr"
            ]

            out.at[
                label,
                "judas_reference_level",
            ] = pending_bear[
                "reference"
            ]

            out.at[
                label,
                "judas_bear_excursion_idx",
            ] = pending_bear[
                "idx"
            ]

            out.at[
                label,
                "judas_bear_confirmation_idx",
            ] = i

            out.at[
                label,
                "judas_bear_lag",
            ] = (
                i
                - pending_bear[
                    "idx"
                ]
            )

            pending_bear = None

    return out
