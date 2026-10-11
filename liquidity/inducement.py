from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InducementParams:
    """
    Proxy algorithmique/falsifiable d'Inducement.

    IMPORTANT:
        On ne prétend pas observer une intention institutionnelle.

    Inducement bullish proxy:
        sweep bullish d'une SSL
        puis BOS bullish dans les N bougies suivantes.

    Inducement bearish proxy:
        sweep bearish d'une BSL
        puis BOS bearish.

    Cette définition est volontairement simple pour permettre
    calibration, ablation et falsification.
    """

    max_bos_lag: int = 10

    def __post_init__(self) -> None:

        if self.max_bos_lag < 0:
            raise ValueError(
                "max_bos_lag doit être >= 0"
            )


def detect_inducement(
    df: pd.DataFrame,
    params: InducementParams = InducementParams(),
) -> pd.DataFrame:

    required = {
        "sweep_bull",
        "sweep_bear",

        "bos_bull",
        "bos_bear",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_inducement(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    n = len(out)

    out["inducement_bull"] = False
    out["inducement_bear"] = False

    for prefix in (
        "inducement_bull",
        "inducement_bear",
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
                > params.max_bos_lag
            )
        ):
            pending_bull = None

        if (
            pending_bear is not None
            and (
                i - pending_bear
                > params.max_bos_lag
            )
        ):
            pending_bear = None

        if (
            pending_bull is not None
            and bool(
                row["bos_bull"]
            )
        ):

            out.at[
                label,
                "inducement_bull",
            ] = True

            out.at[
                label,
                "inducement_bull_sweep_idx",
            ] = pending_bull

            out.at[
                label,
                "inducement_bull_confirmation_idx",
            ] = i

            out.at[
                label,
                "inducement_bull_confirmation_lag",
            ] = (
                i - pending_bull
            )

            pending_bull = None

        if (
            pending_bear is not None
            and bool(
                row["bos_bear"]
            )
        ):

            out.at[
                label,
                "inducement_bear",
            ] = True

            out.at[
                label,
                "inducement_bear_sweep_idx",
            ] = pending_bear

            out.at[
                label,
                "inducement_bear_confirmation_idx",
            ] = i

            out.at[
                label,
                "inducement_bear_confirmation_lag",
            ] = (
                i - pending_bear
            )

            pending_bear = None

    return out
