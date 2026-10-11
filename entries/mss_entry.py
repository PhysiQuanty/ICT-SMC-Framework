from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MSSEntryParams:
    """
    MSS Entry candidate v0.1.

    mode:

        "confirmation"
            entrée candidate directement sur la bougie MSS.
            prix = close.

        "retracement"
            après MSS, attente d'un retracement vers un niveau
            de référence.

    retracement_reference:

        "mss_candle_mid"
            midpoint du high/low de la bougie MSS.

        "displacement_mid"
            alias v0.1 du midpoint de la bougie de confirmation MSS,
            puisque MSS est confirmé par displacement dans notre DAG.

    max_retrace_bars:
        expiration après MSS.

    first_touch_only:
        une entrée par MSS.
    """

    mode: str = "confirmation"

    retracement_reference: str = "mss_candle_mid"

    max_retrace_bars: int = 10

    first_touch_only: bool = True

    def __post_init__(self) -> None:

        if self.mode not in {
            "confirmation",
            "retracement",
        }:
            raise ValueError(
                "mode doit être confirmation ou retracement"
            )

        if self.retracement_reference not in {
            "mss_candle_mid",
            "displacement_mid",
        }:
            raise ValueError(
                "retracement_reference invalide"
            )

        if self.max_retrace_bars < 1:
            raise ValueError(
                "max_retrace_bars doit être >= 1"
            )


def detect_mss_entries(
    df: pd.DataFrame,
    params: MSSEntryParams = MSSEntryParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",

        "mss_bull",
        "mss_bear",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "detect_mss_entries(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)

    mss_bull = out[
        "mss_bull"
    ].to_numpy(dtype=bool)

    mss_bear = out[
        "mss_bear"
    ].to_numpy(dtype=bool)

    out["entry_mss_bull"] = False
    out["entry_mss_bear"] = False

    out["entry_mss_bull_price"] = np.nan
    out["entry_mss_bear_price"] = np.nan

    out["entry_mss_bull_reference"] = np.nan
    out["entry_mss_bear_reference"] = np.nan

    for side in (
        "entry_mss_bull",
        "entry_mss_bear",
    ):
        for suffix in (
            "source_idx",
            "confirmation_idx",
            "lag",
        ):
            out[f"{side}_{suffix}"] = pd.Series(
                pd.array(
                    [pd.NA] * n,
                    dtype="Int64",
                ),
                index=out.index,
            )

    # =========================================================================
    # DIRECT CONFIRMATION MODE
    # =========================================================================

    if params.mode == "confirmation":

        for i in range(n):

            label = out.index[i]

            if mss_bull[i]:

                out.at[
                    label,
                    "entry_mss_bull",
                ] = True

                out.at[
                    label,
                    "entry_mss_bull_price",
                ] = close[i]

                out.at[
                    label,
                    "entry_mss_bull_reference",
                ] = (
                    high[i] + low[i]
                ) / 2.0

                out.at[
                    label,
                    "entry_mss_bull_source_idx",
                ] = i

                out.at[
                    label,
                    "entry_mss_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "entry_mss_bull_lag",
                ] = 0

            if mss_bear[i]:

                out.at[
                    label,
                    "entry_mss_bear",
                ] = True

                out.at[
                    label,
                    "entry_mss_bear_price",
                ] = close[i]

                out.at[
                    label,
                    "entry_mss_bear_reference",
                ] = (
                    high[i] + low[i]
                ) / 2.0

                out.at[
                    label,
                    "entry_mss_bear_source_idx",
                ] = i

                out.at[
                    label,
                    "entry_mss_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "entry_mss_bear_lag",
                ] = 0

        return out

    # =========================================================================
    # RETRACEMENT MODE
    # =========================================================================

    pending_bull = []
    pending_bear = []

    for i in range(n):

        label = out.index[i]

        # ---------------------------------------------------------------------
        # EXISTING BULL MSS
        # ---------------------------------------------------------------------

        survivors = []

        for event in pending_bull:

            lag = i - event["idx"]

            if lag > params.max_retrace_bars:
                continue

            if (
                low[i] <= event["reference"]
                and high[i] >= event["reference"]
            ):

                out.at[
                    label,
                    "entry_mss_bull",
                ] = True

                out.at[
                    label,
                    "entry_mss_bull_price",
                ] = event["reference"]

                out.at[
                    label,
                    "entry_mss_bull_reference",
                ] = event["reference"]

                out.at[
                    label,
                    "entry_mss_bull_source_idx",
                ] = event["idx"]

                out.at[
                    label,
                    "entry_mss_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "entry_mss_bull_lag",
                ] = lag

                if not params.first_touch_only:
                    survivors.append(event)

            else:
                survivors.append(event)

        pending_bull = survivors

        # ---------------------------------------------------------------------
        # EXISTING BEAR MSS
        # ---------------------------------------------------------------------

        survivors = []

        for event in pending_bear:

            lag = i - event["idx"]

            if lag > params.max_retrace_bars:
                continue

            if (
                high[i] >= event["reference"]
                and low[i] <= event["reference"]
            ):

                out.at[
                    label,
                    "entry_mss_bear",
                ] = True

                out.at[
                    label,
                    "entry_mss_bear_price",
                ] = event["reference"]

                out.at[
                    label,
                    "entry_mss_bear_reference",
                ] = event["reference"]

                out.at[
                    label,
                    "entry_mss_bear_source_idx",
                ] = event["idx"]

                out.at[
                    label,
                    "entry_mss_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "entry_mss_bear_lag",
                ] = lag

                if not params.first_touch_only:
                    survivors.append(event)

            else:
                survivors.append(event)

        pending_bear = survivors

        # ---------------------------------------------------------------------
        # NEW MSS
        #
        # Added after current-bar retracement checks:
        # no same-candle retrospective retracement.
        # ---------------------------------------------------------------------

        if mss_bull[i]:

            reference = (
                high[i] + low[i]
            ) / 2.0

            pending_bull.append(
                {
                    "idx":
                        i,

                    "reference":
                        reference,
                }
            )

        if mss_bear[i]:

            reference = (
                high[i] + low[i]
            ) / 2.0

            pending_bear.append(
                {
                    "idx":
                        i,

                    "reference":
                        reference,
                }
            )

    return out
