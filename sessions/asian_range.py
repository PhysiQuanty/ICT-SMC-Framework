from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .session_levels import (
    SessionLevelParams,
    detect_session_levels,
)


@dataclass(frozen=True)
class AsianRangeParams:
    """
    Asian Session High / Low v0.1.

    Default:
        20:00 -> 00:00 America/New_York

    Le moteur générique SessionLevel est réutilisé.

    Les aliases `asian_*` évitent que les consommateurs
    du framework aient besoin de connaître le nom interne
    session_asia_*.
    """

    timezone: str = "America/New_York"

    start: str = "20:00"
    end: str = "00:00"


def detect_asian_range(
    df: pd.DataFrame,
    params: AsianRangeParams = AsianRangeParams(),
) -> pd.DataFrame:

    out = detect_session_levels(
        df,
        SessionLevelParams(
            timezone=params.timezone,
            windows=(
                (
                    "asia",
                    params.start,
                    params.end,
                ),
            ),
        ),
    )

    mapping = {
        "session_asia_active":
            "asian_session_active",

        "session_asia_current_high":
            "asian_current_high",

        "session_asia_current_low":
            "asian_current_low",

        "session_asia_current_mid":
            "asian_current_mid",

        "session_asia_current_range":
            "asian_current_range",

        "session_asia_completed_high":
            "asian_high",

        "session_asia_completed_low":
            "asian_low",

        "session_asia_completed_mid":
            "asian_mid",

        "session_asia_completed_range":
            "asian_range",

        "session_asia_completed_event":
            "asian_range_completed",

        "session_asia_high_taken":
            "asian_high_taken",

        "session_asia_low_taken":
            "asian_low_taken",

        "session_asia_above_completed_high":
            "above_asian_high",

        "session_asia_below_completed_low":
            "below_asian_low",

        "session_asia_completed_high_distance":
            "asian_high_distance",

        "session_asia_completed_low_distance":
            "asian_low_distance",

        "session_asia_completed_start_idx":
            "asian_source_start_idx",

        "session_asia_completed_end_idx":
            "asian_source_end_idx",
    }

    for source, target in mapping.items():

        out[target] = out[
            source
        ]

    return out
