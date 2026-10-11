from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smc_framework.time._utils import get_datetime_utc


@dataclass(frozen=True)
class HTFResampleParams:
    """
    Causal HTF resampling v0.1.

    Exemple avec timeframe="1h":

        LTF bars:
            10:00
            10:15
            10:30
            10:45

        HTF candle:
            open  = 10:00
            close = fin de 10:45 candle

        Elle devient disponible à:
            11:00

    La colonne `htf_available_time` représente donc
    la première date à laquelle le contenu HTF est causalement connu.
    """

    timeframe: str = "1h"

    volume_column: str = "volume"


def resample_htf_ohlcv(
    df: pd.DataFrame,
    params: HTFResampleParams = HTFResampleParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "high",
        "low",
        "close",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "resample_htf_ohlcv(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    dt = get_datetime_utc(
        df
    )

    work = df.copy()

    work["_mtf_datetime"] = dt

    work = work.set_index(
        "_mtf_datetime",
        drop=True,
    )

    aggregation = {
        "open":
            "first",

        "high":
            "max",

        "low":
            "min",

        "close":
            "last",
    }

    if (
        params.volume_column
        in work.columns
    ):

        aggregation[
            params.volume_column
        ] = "sum"

    # -------------------------------------------------------------------------
    # closed="left", label="right"
    #
    # [10:00, 11:00) is labelled 11:00.
    #
    # Therefore the HTF candle appears exactly when it is complete.
    # -------------------------------------------------------------------------

    htf = (
        work
        .resample(
            params.timeframe,
            closed="left",
            label="right",
            origin="epoch",
        )
        .agg(
            aggregation
        )
    )

    htf = htf.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    htf = htf.reset_index()

    htf = htf.rename(
        columns={
            "_mtf_datetime":
                "datetime",
        }
    )

    htf[
        "htf_available_time"
    ] = htf[
        "datetime"
    ]

    htf[
        "htf_bar_idx"
    ] = np.arange(
        len(htf),
        dtype=np.int64,
    )

    return htf


def align_completed_htf_to_ltf(
    ltf: pd.DataFrame,
    htf: pd.DataFrame,
    prefix: str = "htf_",
) -> pd.DataFrame:
    """
    Forward-align completed HTF data onto LTF.

    Uses merge_asof(backward):

        LTF 10:45
            cannot see HTF candle available at 11:00.

        LTF 11:00
            can see it.

    This is the core causal MTF invariant.
    """

    if "htf_available_time" not in htf.columns:

        raise ValueError(
            "align_completed_htf_to_ltf(): "
            "htf_available_time absent"
        )

    ltf_out = ltf.copy()

    ltf_time = get_datetime_utc(
        ltf_out
    )

    ltf_out[
        "_mtf_datetime"
    ] = ltf_time

    htf_work = htf.copy()

    htf_work[
        "_mtf_available"
    ] = pd.to_datetime(
        htf_work[
            "htf_available_time"
        ],
        utc=True,
    )

    protected = {
        "datetime",
        "htf_available_time",
        "_mtf_available",
    }

    rename = {}

    for column in htf_work.columns:

        if column in protected:
            continue

        if column.startswith(
            prefix
        ):
            continue

        rename[column] = (
            f"{prefix}{column}"
        )

    htf_work = htf_work.rename(
        columns=rename
    )

    ltf_out = ltf_out.sort_values(
        "_mtf_datetime"
    )

    htf_work = htf_work.sort_values(
        "_mtf_available"
    )

    merged = pd.merge_asof(
        ltf_out,
        htf_work.drop(
            columns=[
                "datetime",
                "htf_available_time",
            ],
            errors="ignore",
        ),
        left_on="_mtf_datetime",
        right_on="_mtf_available",
        direction="backward",
        allow_exact_matches=True,
    )

    merged = merged.drop(
        columns=[
            "_mtf_datetime",
            "_mtf_available",
        ],
        errors="ignore",
    )

    merged.index = ltf_out.index

    return merged.sort_index()
