from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeStopParams:
    """
    Time Stop / expiry v0.1.

    Construit un horizon temporel causal à partir d'un événement
    d'entrée candidat.

    bull_event_column / bear_event_column:
        booléens d'entrée.

    max_bars:
        nombre maximal de bougies pendant lequel l'idée reste active.

    Expiry:
        entrée à i
        max_bars = 10

        => expiration à i + 10.

    emit_expiry_event:
        True -> booléen uniquement sur la barre d'expiration.

    Cette feature ne décide pas à quel prix fermer.
    """

    bull_event_column: str = "entry_mss_bull"
    bear_event_column: str = "entry_mss_bear"

    max_bars: int = 50

    consume_previous: bool = True

    def __post_init__(self) -> None:

        if self.max_bars < 1:
            raise ValueError(
                "max_bars doit être >= 1"
            )


def detect_time_stops(
    df: pd.DataFrame,
    params: TimeStopParams = TimeStopParams(),
) -> pd.DataFrame:

    required = {
        params.bull_event_column,
        params.bear_event_column,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_time_stops(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    bull_event = out[
        params.bull_event_column
    ].fillna(False).to_numpy(
        dtype=bool,
    )

    bear_event = out[
        params.bear_event_column
    ].fillna(False).to_numpy(
        dtype=bool,
    )

    out[
        "time_stop_bull_active"
    ] = False

    out[
        "time_stop_bear_active"
    ] = False

    out[
        "time_stop_bull_expired"
    ] = False

    out[
        "time_stop_bear_expired"
    ] = False

    for prefix in (
        "time_stop_bull",
        "time_stop_bear",
    ):

        out[
            f"{prefix}_entry_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_expiry_idx"
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

        out[
            f"{prefix}_bars_remaining"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    active_bull = []
    active_bear = []

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # ADD NEW ENTRIES
        # =====================================================================

        if bull_event[i]:

            if params.consume_previous:
                active_bull = []

            active_bull.append(
                {
                    "entry_idx":
                        i,

                    "expiry_idx":
                        i + params.max_bars,
                }
            )

        if bear_event[i]:

            if params.consume_previous:
                active_bear = []

            active_bear.append(
                {
                    "entry_idx":
                        i,

                    "expiry_idx":
                        i + params.max_bars,
                }
            )

        # =====================================================================
        # BULL
        # =====================================================================

        bull_survivors = []

        for event in active_bull:

            age = (
                i - event["entry_idx"]
            )

            remaining = (
                event["expiry_idx"]
                - i
            )

            if i < event["expiry_idx"]:

                # With consume_previous=True only one exists.
                out.at[
                    label,
                    "time_stop_bull_active",
                ] = True

                out.at[
                    label,
                    "time_stop_bull_entry_idx",
                ] = event[
                    "entry_idx"
                ]

                out.at[
                    label,
                    "time_stop_bull_expiry_idx",
                ] = event[
                    "expiry_idx"
                ]

                out.at[
                    label,
                    "time_stop_bull_age_bars",
                ] = age

                out.at[
                    label,
                    "time_stop_bull_bars_remaining",
                ] = remaining

                bull_survivors.append(
                    event
                )

            elif i == event["expiry_idx"]:

                out.at[
                    label,
                    "time_stop_bull_expired",
                ] = True

                out.at[
                    label,
                    "time_stop_bull_entry_idx",
                ] = event[
                    "entry_idx"
                ]

                out.at[
                    label,
                    "time_stop_bull_expiry_idx",
                ] = event[
                    "expiry_idx"
                ]

                out.at[
                    label,
                    "time_stop_bull_age_bars",
                ] = age

                out.at[
                    label,
                    "time_stop_bull_bars_remaining",
                ] = 0

        active_bull = bull_survivors

        # =====================================================================
        # BEAR
        # =====================================================================

        bear_survivors = []

        for event in active_bear:

            age = (
                i - event["entry_idx"]
            )

            remaining = (
                event["expiry_idx"]
                - i
            )

            if i < event["expiry_idx"]:

                out.at[
                    label,
                    "time_stop_bear_active",
                ] = True

                out.at[
                    label,
                    "time_stop_bear_entry_idx",
                ] = event[
                    "entry_idx"
                ]

                out.at[
                    label,
                    "time_stop_bear_expiry_idx",
                ] = event[
                    "expiry_idx"
                ]

                out.at[
                    label,
                    "time_stop_bear_age_bars",
                ] = age

                out.at[
                    label,
                    "time_stop_bear_bars_remaining",
                ] = remaining

                bear_survivors.append(
                    event
                )

            elif i == event["expiry_idx"]:

                out.at[
                    label,
                    "time_stop_bear_expired",
                ] = True

                out.at[
                    label,
                    "time_stop_bear_entry_idx",
                ] = event[
                    "entry_idx"
                ]

                out.at[
                    label,
                    "time_stop_bear_expiry_idx",
                ] = event[
                    "expiry_idx"
                ]

                out.at[
                    label,
                    "time_stop_bear_age_bars",
                ] = age

                out.at[
                    label,
                    "time_stop_bear_bars_remaining",
                ] = 0

        active_bear = bear_survivors

    return out
