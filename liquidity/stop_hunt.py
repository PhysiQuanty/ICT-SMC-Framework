from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StopHuntParams:
    """
    Proxy falsifiable de Stop Hunt.

    IMPORTANT:
        Cette feature NE prétend PAS observer des stops réels.

    Proxy bullish:
        sweep bullish de SSL
        +
        displacement bullish dans les N bougies suivantes.

    Proxy bearish:
        sweep bearish de BSL
        +
        displacement bearish.

    max_displacement_lag:
        0 = même bougie
        1 = même bougie ou suivante
        etc.
    """

    max_displacement_lag: int = 2

    def __post_init__(self) -> None:

        if self.max_displacement_lag < 0:
            raise ValueError(
                "max_displacement_lag doit être >= 0"
            )


def detect_stop_hunts(
    df: pd.DataFrame,
    params: StopHuntParams = StopHuntParams(),
) -> pd.DataFrame:

    required = {
        "sweep_bull",
        "sweep_bear",

        "displacement_bull",
        "displacement_bear",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_stop_hunts(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    n = len(out)

    out["stop_hunt_bull"] = False
    out["stop_hunt_bear"] = False

    for prefix in (
        "stop_hunt_bull",
        "stop_hunt_bear",
    ):

        out[
            f"{prefix}_sweep_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_confirmation_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{prefix}_confirmation_lag"
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

        row = out.iloc[i]
        label = out.index[i]

        if bool(
            row["sweep_bull"]
        ):
            pending_bull = i

        if bool(
            row["sweep_bear"]
        ):
            pending_bear = i

        if (
            pending_bull is not None
            and (
                i - pending_bull
                > params.max_displacement_lag
            )
        ):
            pending_bull = None

        if (
            pending_bear is not None
            and (
                i - pending_bear
                > params.max_displacement_lag
            )
        ):
            pending_bear = None

        if (
            pending_bull is not None
            and bool(
                row[
                    "displacement_bull"
                ]
            )
        ):

            out.at[
                label,
                "stop_hunt_bull",
            ] = True

            out.at[
                label,
                "stop_hunt_bull_sweep_idx",
            ] = pending_bull

            out.at[
                label,
                "stop_hunt_bull_confirmation_idx",
            ] = i

            out.at[
                label,
                "stop_hunt_bull_confirmation_lag",
            ] = (
                i - pending_bull
            )

            pending_bull = None

        if (
            pending_bear is not None
            and bool(
                row[
                    "displacement_bear"
                ]
            )
        ):

            out.at[
                label,
                "stop_hunt_bear",
            ] = True

            out.at[
                label,
                "stop_hunt_bear_sweep_idx",
            ] = pending_bear

            out.at[
                label,
                "stop_hunt_bear_confirmation_idx",
            ] = i

            out.at[
                label,
                "stop_hunt_bear_confirmation_lag",
            ] = (
                i - pending_bear
            )

            pending_bear = None

    return out
