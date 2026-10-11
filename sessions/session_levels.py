from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from smc_framework.time._utils import (
    local_datetime,
    minute_of_day,
    parse_hhmm,
)


DEFAULT_SESSION_WINDOWS = (
    (
        "london",
        "02:00",
        "05:00",
    ),
    (
        "new_york",
        "07:00",
        "10:00",
    ),
    (
        "london_close",
        "10:00",
        "12:00",
    ),
)


@dataclass(frozen=True)
class SessionLevelParams:
    """
    Generic intraday Session High / Low v0.1.

    windows:
        tuple:
            (name, start HH:MM, end HH:MM)

    Toutes les fenêtres utilisent le même timezone.

    Une fenêtre peut traverser minuit.

    Ex:
        ("session_x", "22:00", "02:00")

    Outputs:

        session_<name>_active

        session_<name>_current_high
        session_<name>_current_low
            -> évoluent uniquement pendant la session.

        session_<name>_completed_high
        session_<name>_completed_low
            -> dernière session TERMINÉE,
               disponibles seulement après sa fin.

        session_<name>_completed_event
            -> première bougie observée après la fermeture.

    completed levels persist until the next session is finalized.
    """

    timezone: str = "America/New_York"

    windows: tuple[
        tuple[str, str, str],
        ...
    ] = DEFAULT_SESSION_WINDOWS


def _safe_name(
    name: str,
) -> str:

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _session_membership_and_key(
    dt: pd.Series,
    start: str,
    end: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    start_min = parse_hhmm(
        start
    )

    end_min = parse_hhmm(
        end
    )

    minutes = minute_of_day(
        dt
    )

    dates = dt.dt.date.to_numpy()

    active = np.zeros(
        len(dt),
        dtype=bool,
    )

    keys = np.full(
        len(dt),
        None,
        dtype=object,
    )

    # Normal intraday session.
    if start_min < end_min:

        active = (
            (minutes >= start_min)
            & (minutes < end_min)
        )

        for i in np.flatnonzero(
            active
        ):
            keys[i] = dates[i]

    # Cross-midnight session.
    elif start_min > end_min:

        active = (
            (minutes >= start_min)
            | (minutes < end_min)
        )

        for i in np.flatnonzero(
            active
        ):

            if minutes[i] >= start_min:

                keys[i] = dates[i]

            else:

                keys[i] = (
                    dates[i]
                    - timedelta(days=1)
                )

    # start == end => 24h session.
    else:

        active[:] = True

        for i in range(
            len(dt)
        ):
            keys[i] = dates[i]

    return (
        active,
        keys,
    )


def detect_session_levels(
    df: pd.DataFrame,
    params: SessionLevelParams = SessionLevelParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_session_levels(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    dt = local_datetime(
        out,
        params.timezone,
    )

    high = out[
        "high"
    ].to_numpy(
        dtype=np.float64,
    )

    low = out[
        "low"
    ].to_numpy(
        dtype=np.float64,
    )

    close = out[
        "close"
    ].to_numpy(
        dtype=np.float64,
    )

    for (
        raw_name,
        start,
        end,
    ) in params.windows:

        name = _safe_name(
            raw_name
        )

        prefix = (
            f"session_{name}"
        )

        (
            active,
            keys,
        ) = _session_membership_and_key(
            dt,
            start,
            end,
        )

        # =====================================================================
        # OUTPUTS
        # =====================================================================

        out[
            f"{prefix}_active"
        ] = active

        out[
            f"{prefix}_current_high"
        ] = np.nan

        out[
            f"{prefix}_current_low"
        ] = np.nan

        out[
            f"{prefix}_current_mid"
        ] = np.nan

        out[
            f"{prefix}_current_range"
        ] = np.nan

        out[
            f"{prefix}_completed_high"
        ] = np.nan

        out[
            f"{prefix}_completed_low"
        ] = np.nan

        out[
            f"{prefix}_completed_mid"
        ] = np.nan

        out[
            f"{prefix}_completed_range"
        ] = np.nan

        out[
            f"{prefix}_completed_event"
        ] = False

        out[
            f"{prefix}_high_taken"
        ] = False

        out[
            f"{prefix}_low_taken"
        ] = False

        out[
            f"{prefix}_above_completed_high"
        ] = False

        out[
            f"{prefix}_below_completed_low"
        ] = False

        out[
            f"{prefix}_completed_high_distance"
        ] = np.nan

        out[
            f"{prefix}_completed_low_distance"
        ] = np.nan

        out[
            f"{prefix}_start_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_completed_start_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_completed_end_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        # =====================================================================
        # STATE
        # =====================================================================

        current_key = None

        current_high = None
        current_low = None

        current_start_idx = None
        current_end_idx = None

        completed_high = None
        completed_low = None

        completed_start_idx = None
        completed_end_idx = None

        high_taken = False
        low_taken = False

        was_active = False

        for i in range(n):

            label = out.index[i]

            # ================================================================
            # SESSION JUST FINISHED
            # ================================================================

            if (
                was_active
                and not active[i]
                and current_key is not None
            ):

                completed_high = (
                    current_high
                )

                completed_low = (
                    current_low
                )

                completed_start_idx = (
                    current_start_idx
                )

                completed_end_idx = (
                    current_end_idx
                )

                high_taken = False
                low_taken = False

                out.at[
                    label,
                    f"{prefix}_completed_event",
                ] = True

                current_key = None

                current_high = None
                current_low = None

                current_start_idx = None
                current_end_idx = None

            # ================================================================
            # ACTIVE SESSION
            # ================================================================

            if active[i]:

                key = keys[i]

                # New session key.
                if (
                    current_key is None
                    or key != current_key
                ):

                    # Handles datasets where one session immediately
                    # follows another without an inactive bar.
                    if (
                        current_key is not None
                    ):

                        completed_high = (
                            current_high
                        )

                        completed_low = (
                            current_low
                        )

                        completed_start_idx = (
                            current_start_idx
                        )

                        completed_end_idx = (
                            current_end_idx
                        )

                        high_taken = False
                        low_taken = False

                        out.at[
                            label,
                            f"{prefix}_completed_event",
                        ] = True

                    current_key = key

                    current_high = (
                        high[i]
                    )

                    current_low = (
                        low[i]
                    )

                    current_start_idx = i
                    current_end_idx = i

                else:

                    current_high = max(
                        current_high,
                        high[i],
                    )

                    current_low = min(
                        current_low,
                        low[i],
                    )

                    current_end_idx = i

                out.at[
                    label,
                    f"{prefix}_current_high",
                ] = current_high

                out.at[
                    label,
                    f"{prefix}_current_low",
                ] = current_low

                out.at[
                    label,
                    f"{prefix}_current_mid",
                ] = (
                    current_high
                    + current_low
                ) / 2.0

                out.at[
                    label,
                    f"{prefix}_current_range",
                ] = (
                    current_high
                    - current_low
                )

                out.at[
                    label,
                    f"{prefix}_start_idx",
                ] = current_start_idx

            # ================================================================
            # LAST COMPLETED SESSION LEVELS
            # ================================================================

            if (
                completed_high is not None
                and completed_low is not None
            ):

                out.at[
                    label,
                    f"{prefix}_completed_high",
                ] = completed_high

                out.at[
                    label,
                    f"{prefix}_completed_low",
                ] = completed_low

                out.at[
                    label,
                    f"{prefix}_completed_mid",
                ] = (
                    completed_high
                    + completed_low
                ) / 2.0

                out.at[
                    label,
                    f"{prefix}_completed_range",
                ] = (
                    completed_high
                    - completed_low
                )

                out.at[
                    label,
                    f"{prefix}_completed_start_idx",
                ] = completed_start_idx

                out.at[
                    label,
                    f"{prefix}_completed_end_idx",
                ] = completed_end_idx

                out.at[
                    label,
                    f"{prefix}_completed_high_distance",
                ] = (
                    close[i]
                    - completed_high
                )

                out.at[
                    label,
                    f"{prefix}_completed_low_distance",
                ] = (
                    close[i]
                    - completed_low
                )

                out.at[
                    label,
                    f"{prefix}_above_completed_high",
                ] = (
                    close[i]
                    > completed_high
                )

                out.at[
                    label,
                    f"{prefix}_below_completed_low",
                ] = (
                    close[i]
                    < completed_low
                )

                # Do not count the completion row as a previous-session
                # level break caused by information from inside that session.
                if not bool(
                    out.at[
                        label,
                        f"{prefix}_completed_event",
                    ]
                ):

                    if (
                        not high_taken
                        and high[i]
                        > completed_high
                    ):

                        high_taken = True

                        out.at[
                            label,
                            f"{prefix}_high_taken",
                        ] = True

                    if (
                        not low_taken
                        and low[i]
                        < completed_low
                    ):

                        low_taken = True

                        out.at[
                            label,
                            f"{prefix}_low_taken",
                        ] = True

            was_active = bool(
                active[i]
            )

    return out
