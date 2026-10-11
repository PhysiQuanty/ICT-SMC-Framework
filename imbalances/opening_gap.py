from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OpeningGapParams:
    """
    Opening Gap.

    Défini comme:
        open[i] - close[i-1]

    min_gap_atr:
        taille minimale en ATR.

    min_gap_percent:
        taille minimale relative au close précédent.

    Les deux seuils doivent être satisfaits s'ils sont > 0.

    session_reset_only:
        si True, on ne considère que les ruptures de date UTC.
        utile pour marchés avec fermeture/session.

        Pour crypto 24/7, mettre False pour mesurer tout gap OHLC.
    """

    min_gap_atr: float = 0.0
    min_gap_percent: float = 0.0

    atr_column: str = "atr"

    session_reset_only: bool = False

    def __post_init__(self) -> None:

        if self.min_gap_atr < 0:
            raise ValueError(
                "min_gap_atr doit être >= 0"
            )

        if self.min_gap_percent < 0:
            raise ValueError(
                "min_gap_percent doit être >= 0"
            )


def detect_opening_gap(
    df: pd.DataFrame,
    params: OpeningGapParams = OpeningGapParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "close",
    }

    if params.min_gap_atr > 0:
        required.add(
            params.atr_column
        )

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "detect_opening_gap(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    open_ = out[
        "open"
    ].to_numpy(
        dtype=np.float64,
    )

    close = out[
        "close"
    ].to_numpy(
        dtype=np.float64,
    )

    previous_close = np.roll(
        close,
        1,
    )

    previous_close[0] = np.nan

    gap = (
        open_
        - previous_close
    )

    absolute_gap = np.abs(
        gap
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        gap_percent = (
            absolute_gap
            / np.abs(previous_close)
        )

    if params.atr_column in out.columns:

        atr = out[
            params.atr_column
        ].to_numpy(
            dtype=np.float64,
        )

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):

            gap_atr = (
                absolute_gap / atr
            )

    else:

        gap_atr = np.full(
            n,
            np.nan,
        )

    valid = np.ones(
        n,
        dtype=bool,
    )

    valid[0] = False

    if params.min_gap_atr > 0:

        valid &= (
            gap_atr
            >= params.min_gap_atr
        )

    if params.min_gap_percent > 0:

        valid &= (
            gap_percent
            >= params.min_gap_percent
        )

    if params.session_reset_only:

        if "datetime" not in out.columns:
            raise ValueError(
                "session_reset_only=True nécessite "
                "une colonne 'datetime'"
            )

        dt = pd.to_datetime(
            out["datetime"],
            utc=True,
        )

        dates = dt.dt.date.to_numpy()

        new_session = np.zeros(
            n,
            dtype=bool,
        )

        new_session[1:] = (
            dates[1:]
            != dates[:-1]
        )

        valid &= new_session

    out["opening_gap_bull"] = (
        valid
        & (gap > 0)
    )

    out["opening_gap_bear"] = (
        valid
        & (gap < 0)
    )

    out["opening_gap_size"] = (
        absolute_gap
    )

    out["opening_gap_size_atr"] = (
        gap_atr
    )

    out["opening_gap_percent"] = (
        gap_percent
    )

    out["opening_gap_previous_close"] = (
        previous_close
    )

    out["opening_gap_open"] = (
        open_
    )

    out["opening_gap_confirmation_idx"] = pd.Series(
        pd.array(
            [
                i if valid[i] else pd.NA
                for i in range(n)
            ],
            dtype="Int64",
        ),
        index=out.index,
    )

    return out
