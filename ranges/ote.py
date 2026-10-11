from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OTEParams:
    """
    Optimal Trade Entry zone v0.1.

    OTE est représentée comme une zone de retracement du dernier
    Dealing Range directionnel confirmé.

    Valeurs par défaut:
        retracement_min = 0.62
        retracement_max = 0.79

    Bullish range:
        low -> high

        retracement mesuré depuis le HIGH vers le LOW.

        62%:
            high - 0.62 * range

        79%:
            high - 0.79 * range

    Bearish range:
        high -> low

        retracement mesuré depuis le LOW vers le HIGH.

        62%:
            low + 0.62 * range

        79%:
            low + 0.79 * range

    first_touch_only:
        booléen event ote_touch_* seulement au premier touch.

    max_age_bars:
        expiration du range pour OTE.
    """

    retracement_min: float = 0.62
    retracement_max: float = 0.79

    max_age_bars: int = 100

    first_touch_only: bool = True

    def __post_init__(self) -> None:

        if not (
            0
            <= self.retracement_min
            <= 1
        ):
            raise ValueError(
                "retracement_min doit être entre 0 et 1"
            )

        if not (
            0
            <= self.retracement_max
            <= 1
        ):
            raise ValueError(
                "retracement_max doit être entre 0 et 1"
            )

        if (
            self.retracement_min
            > self.retracement_max
        ):
            raise ValueError(
                "retracement_min doit être <= retracement_max"
            )

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )


def detect_ote(
    df: pd.DataFrame,
    params: OTEParams = OTEParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",

        "dealing_range_active",
        "dealing_range_direction",
        "dealing_range_low",
        "dealing_range_high",
        "dealing_range_confirmation_idx",
        "dealing_range_age_bars",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_ote(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["ote_bull_active"] = False
    out["ote_bear_active"] = False

    out["ote_bull_touch"] = False
    out["ote_bear_touch"] = False

    out["ote_bull_low"] = np.nan
    out["ote_bull_high"] = np.nan
    out["ote_bull_mid"] = np.nan

    out["ote_bear_low"] = np.nan
    out["ote_bear_high"] = np.nan
    out["ote_bear_mid"] = np.nan

    out["ote_retracement"] = np.nan

    for prefix in (
        "ote_bull",
        "ote_bear",
    ):

        out[
            f"{prefix}_source_range_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_first_touch_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    high = out["high"].to_numpy(
        dtype=np.float64,
    )

    low = out["low"].to_numpy(
        dtype=np.float64,
    )

    close = out["close"].to_numpy(
        dtype=np.float64,
    )

    active = out[
        "dealing_range_active"
    ].to_numpy(dtype=bool)

    direction = out[
        "dealing_range_direction"
    ].to_numpy()

    range_low = out[
        "dealing_range_low"
    ].to_numpy(dtype=np.float64)

    range_high = out[
        "dealing_range_high"
    ].to_numpy(dtype=np.float64)

    range_confirmation = out[
        "dealing_range_confirmation_idx"
    ]

    last_range_id = None
    range_touched = False
    first_touch_idx = None

    for i in range(n):

        if not active[i]:
            continue

        confirmation = (
            range_confirmation.iloc[i]
        )

        if pd.isna(confirmation):
            continue

        confirmation = int(
            confirmation
        )

        # New dealing range.
        if confirmation != last_range_id:

            last_range_id = confirmation
            range_touched = False
            first_touch_idx = None

        age = (
            i - confirmation
        )

        if age > params.max_age_bars:
            continue

        r_low = range_low[i]
        r_high = range_high[i]

        size = (
            r_high - r_low
        )

        if (
            not np.isfinite(size)
            or size <= 0
        ):
            continue

        label = out.index[i]

        # =====================================================================
        # BULLISH OTE
        # =====================================================================

        if direction[i] == "bull":

            level_62 = (
                r_high
                - params.retracement_min
                * size
            )

            level_79 = (
                r_high
                - params.retracement_max
                * size
            )

            zone_low = min(
                level_62,
                level_79,
            )

            zone_high = max(
                level_62,
                level_79,
            )

            zone_mid = (
                zone_low
                + zone_high
            ) / 2.0

            out.at[
                label,
                "ote_bull_active",
            ] = True

            out.at[
                label,
                "ote_bull_low",
            ] = zone_low

            out.at[
                label,
                "ote_bull_high",
            ] = zone_high

            out.at[
                label,
                "ote_bull_mid",
            ] = zone_mid

            out.at[
                label,
                "ote_bull_source_range_idx",
            ] = confirmation

            retracement = (
                (
                    r_high - close[i]
                )
                / size
            )

            out.at[
                label,
                "ote_retracement",
            ] = retracement

            # Do not touch on creation candle.
            touched_now = (
                i > confirmation
                and low[i] <= zone_high
                and high[i] >= zone_low
            )

            emit = (
                touched_now
                and (
                    not params.first_touch_only
                    or not range_touched
                )
            )

            if emit:

                out.at[
                    label,
                    "ote_bull_touch",
                ] = True

                if first_touch_idx is None:
                    first_touch_idx = i

                range_touched = True

            if first_touch_idx is not None:

                out.at[
                    label,
                    "ote_bull_first_touch_idx",
                ] = first_touch_idx

        # =====================================================================
        # BEARISH OTE
        # =====================================================================

        elif direction[i] == "bear":

            level_62 = (
                r_low
                + params.retracement_min
                * size
            )

            level_79 = (
                r_low
                + params.retracement_max
                * size
            )

            zone_low = min(
                level_62,
                level_79,
            )

            zone_high = max(
                level_62,
                level_79,
            )

            zone_mid = (
                zone_low
                + zone_high
            ) / 2.0

            out.at[
                label,
                "ote_bear_active",
            ] = True

            out.at[
                label,
                "ote_bear_low",
            ] = zone_low

            out.at[
                label,
                "ote_bear_high",
            ] = zone_high

            out.at[
                label,
                "ote_bear_mid",
            ] = zone_mid

            out.at[
                label,
                "ote_bear_source_range_idx",
            ] = confirmation

            retracement = (
                (
                    close[i] - r_low
                )
                / size
            )

            out.at[
                label,
                "ote_retracement",
            ] = retracement

            touched_now = (
                i > confirmation
                and high[i] >= zone_low
                and low[i] <= zone_high
            )

            emit = (
                touched_now
                and (
                    not params.first_touch_only
                    or not range_touched
                )
            )

            if emit:

                out.at[
                    label,
                    "ote_bear_touch",
                ] = True

                if first_touch_idx is None:
                    first_touch_idx = i

                range_touched = True

            if first_touch_idx is not None:

                out.at[
                    label,
                    "ote_bear_first_touch_idx",
                ] = first_touch_idx

    return out
