from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EqualHighLowParams:
    """
    Equal High / Equal Low.

    tolerance_mode:
        "atr"
        "percent"
        "absolute"

    tolerance:
        distance maximale entre deux swings.

        Exemple ATR:
            tolerance = 0.10
            => différence <= 0.10 ATR

    min_cluster_size:
        nombre minimal de swings pour considérer une zone
        comme Equal High / Equal Low.
    """

    tolerance_mode: str = "atr"
    tolerance: float = 0.10

    atr_column: str = "atr"

    min_cluster_size: int = 2

    def __post_init__(self) -> None:

        if self.tolerance_mode not in {
            "atr",
            "percent",
            "absolute",
        }:
            raise ValueError(
                "tolerance_mode invalide"
            )

        if self.tolerance < 0:
            raise ValueError(
                "tolerance doit être >= 0"
            )

        if self.min_cluster_size < 2:
            raise ValueError(
                "min_cluster_size doit être >= 2"
            )


def _within_tolerance(
    price_a: float,
    price_b: float,
    atr: float,
    params: EqualHighLowParams,
) -> tuple[bool, float, float]:

    distance = abs(
        price_a - price_b
    )

    if params.tolerance_mode == "atr":

        normalized = (
            distance / atr
            if np.isfinite(atr) and atr > 0
            else np.nan
        )

        ok = (
            np.isfinite(normalized)
            and normalized <= params.tolerance
        )

        return (
            ok,
            distance,
            normalized,
        )

    if params.tolerance_mode == "percent":

        reference = (
            abs(price_b)
        )

        normalized = (
            distance / reference
            if reference > 0
            else np.nan
        )

        ok = (
            np.isfinite(normalized)
            and normalized <= params.tolerance
        )

        return (
            ok,
            distance,
            normalized,
        )

    normalized = distance

    return (
        distance <= params.tolerance,
        distance,
        normalized,
    )


def detect_equal_high_low(
    df: pd.DataFrame,
    params: EqualHighLowParams = EqualHighLowParams(),
) -> pd.DataFrame:

    required = {
        "swing_high",
        "swing_low",
        "swing_high_price",
        "swing_low_price",
    }

    if params.tolerance_mode == "atr":
        required.add(
            params.atr_column
        )

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_equal_high_low(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    n = len(out)

    out["equal_high"] = False
    out["equal_low"] = False

    out["equal_high_level"] = np.nan
    out["equal_low_level"] = np.nan

    out["equal_high_distance"] = np.nan
    out["equal_low_distance"] = np.nan

    out["equal_high_distance_normalized"] = np.nan
    out["equal_low_distance_normalized"] = np.nan

    out["equal_high_cluster_size"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["equal_low_cluster_size"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["equal_high_previous_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["equal_low_previous_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    swing_high = out[
        "swing_high"
    ].to_numpy(
        dtype=bool,
    )

    swing_low = out[
        "swing_low"
    ].to_numpy(
        dtype=bool,
    )

    high_price = out[
        "swing_high_price"
    ].to_numpy(
        dtype=np.float64,
    )

    low_price = out[
        "swing_low_price"
    ].to_numpy(
        dtype=np.float64,
    )

    if params.atr_column in out.columns:

        atr = out[
            params.atr_column
        ].to_numpy(
            dtype=np.float64,
        )

    else:

        atr = np.full(
            n,
            np.nan,
        )

    previous_high = None
    previous_low = None

    high_cluster_size = 1
    low_cluster_size = 1

    high_cluster_level = np.nan
    low_cluster_level = np.nan

    for i in range(n):

        label = out.index[i]

        # =====================================================================
        # EQUAL HIGH
        # =====================================================================

        if swing_high[i]:

            price = high_price[i]

            if previous_high is not None:

                ok, distance, normalized = (
                    _within_tolerance(
                        price,
                        previous_high["price"],
                        atr[i],
                        params,
                    )
                )

                if ok:

                    high_cluster_size += 1

                    if np.isfinite(
                        high_cluster_level
                    ):
                        high_cluster_level = (
                            (
                                high_cluster_level
                                * (
                                    high_cluster_size
                                    - 1
                                )
                            )
                            + price
                        ) / high_cluster_size

                    else:
                        high_cluster_level = (
                            price
                            + previous_high["price"]
                        ) / 2.0

                    if (
                        high_cluster_size
                        >= params.min_cluster_size
                    ):

                        out.at[
                            label,
                            "equal_high",
                        ] = True

                        out.at[
                            label,
                            "equal_high_level",
                        ] = high_cluster_level

                        out.at[
                            label,
                            "equal_high_distance",
                        ] = distance

                        out.at[
                            label,
                            "equal_high_distance_normalized",
                        ] = normalized

                        out.at[
                            label,
                            "equal_high_cluster_size",
                        ] = high_cluster_size

                        out.at[
                            label,
                            "equal_high_previous_idx",
                        ] = previous_high["idx"]

                else:

                    high_cluster_size = 1
                    high_cluster_level = np.nan

            previous_high = {
                "price":
                    price,

                "idx":
                    i,
            }

        # =====================================================================
        # EQUAL LOW
        # =====================================================================

        if swing_low[i]:

            price = low_price[i]

            if previous_low is not None:

                ok, distance, normalized = (
                    _within_tolerance(
                        price,
                        previous_low["price"],
                        atr[i],
                        params,
                    )
                )

                if ok:

                    low_cluster_size += 1

                    if np.isfinite(
                        low_cluster_level
                    ):

                        low_cluster_level = (
                            (
                                low_cluster_level
                                * (
                                    low_cluster_size
                                    - 1
                                )
                            )
                            + price
                        ) / low_cluster_size

                    else:

                        low_cluster_level = (
                            price
                            + previous_low["price"]
                        ) / 2.0

                    if (
                        low_cluster_size
                        >= params.min_cluster_size
                    ):

                        out.at[
                            label,
                            "equal_low",
                        ] = True

                        out.at[
                            label,
                            "equal_low_level",
                        ] = low_cluster_level

                        out.at[
                            label,
                            "equal_low_distance",
                        ] = distance

                        out.at[
                            label,
                            "equal_low_distance_normalized",
                        ] = normalized

                        out.at[
                            label,
                            "equal_low_cluster_size",
                        ] = low_cluster_size

                        out.at[
                            label,
                            "equal_low_previous_idx",
                        ] = previous_low["idx"]

                else:

                    low_cluster_size = 1
                    low_cluster_level = np.nan

            previous_low = {
                "price":
                    price,

                "idx":
                    i,
            }

    return out
