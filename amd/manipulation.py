from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ManipulationParams:
    """
    Manipulation v0.1.

    Proxy:
        excursion hors de l'accumulation terminée
        + clôture de retour dans le range.

    manipulation_high:
        high > accumulation_high
        puis close < accumulation_high

        => excursion haussière, candidate à une future
           distribution bearish.

    manipulation_low:
        low < accumulation_low
        puis close > accumulation_low

        => excursion baissière, candidate à une future
           distribution bullish.

    min_depth_atr:
        profondeur minimale hors du range.

    max_age_bars:
        âge maximal de l'accumulation.
    """

    min_depth_atr: float = 0.0
    max_age_bars: int = 50

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.min_depth_atr < 0:
            raise ValueError(
                "min_depth_atr doit être >= 0"
            )

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )


def detect_manipulation(
    df: pd.DataFrame,
    params: ManipulationParams = ManipulationParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
        params.atr_column,

        "accumulation_valid",
        "accumulation_high",
        "accumulation_low",
        "accumulation_confirmation_idx",
        "accumulation_age_bars",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_manipulation(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    high = out[
        "high"
    ].to_numpy(dtype=np.float64)

    low = out[
        "low"
    ].to_numpy(dtype=np.float64)

    close = out[
        "close"
    ].to_numpy(dtype=np.float64)

    atr = out[
        params.atr_column
    ].to_numpy(dtype=np.float64)

    valid = out[
        "accumulation_valid"
    ].to_numpy(dtype=bool)

    acc_high = out[
        "accumulation_high"
    ].to_numpy(dtype=np.float64)

    acc_low = out[
        "accumulation_low"
    ].to_numpy(dtype=np.float64)

    age = out[
        "accumulation_age_bars"
    ]

    out["manipulation_high"] = False
    out["manipulation_low"] = False

    out["manipulation_high_level"] = np.nan
    out["manipulation_low_level"] = np.nan

    out["manipulation_high_depth"] = np.nan
    out["manipulation_low_depth"] = np.nan

    out["manipulation_high_depth_atr"] = np.nan
    out["manipulation_low_depth_atr"] = np.nan

    out["manipulation_high_close_back"] = np.nan
    out["manipulation_low_close_back"] = np.nan

    for side in (
        "manipulation_high",
        "manipulation_low",
    ):

        for suffix in (
            "source_accumulation_idx",
            "confirmation_idx",
        ):

            out[
                f"{side}_{suffix}"
            ] = pd.Series(
                pd.array(
                    [pd.NA] * n,
                    dtype="Int64",
                ),
                index=out.index,
            )

    consumed_high = set()
    consumed_low = set()

    for i in range(n):

        if not valid[i]:
            continue

        if pd.isna(
            age.iloc[i]
        ):
            continue

        if int(
            age.iloc[i]
        ) > params.max_age_bars:
            continue

        source_idx = out.iloc[i][
            "accumulation_confirmation_idx"
        ]

        if pd.isna(source_idx):
            continue

        source_idx = int(
            source_idx
        )

        if i <= source_idx:
            continue

        label = out.index[i]

        # =====================================================================
        # HIGH-SIDE MANIPULATION
        # =====================================================================

        depth = (
            high[i]
            - acc_high[i]
        )

        depth_atr = (
            depth / atr[i]
            if (
                np.isfinite(atr[i])
                and atr[i] > 0
            )
            else np.nan
        )

        threshold_ok = (
            depth > 0
            and (
                params.min_depth_atr == 0
                or (
                    np.isfinite(depth_atr)
                    and depth_atr
                    >= params.min_depth_atr
                )
            )
        )

        if (
            source_idx
            not in consumed_high
            and threshold_ok
            and close[i] < acc_high[i]
        ):

            out.at[
                label,
                "manipulation_high",
            ] = True

            out.at[
                label,
                "manipulation_high_level",
            ] = acc_high[i]

            out.at[
                label,
                "manipulation_high_depth",
            ] = depth

            out.at[
                label,
                "manipulation_high_depth_atr",
            ] = depth_atr

            out.at[
                label,
                "manipulation_high_close_back",
            ] = (
                acc_high[i]
                - close[i]
            )

            out.at[
                label,
                "manipulation_high_source_accumulation_idx",
            ] = source_idx

            out.at[
                label,
                "manipulation_high_confirmation_idx",
            ] = i

            consumed_high.add(
                source_idx
            )

        # =====================================================================
        # LOW-SIDE MANIPULATION
        # =====================================================================

        depth = (
            acc_low[i]
            - low[i]
        )

        depth_atr = (
            depth / atr[i]
            if (
                np.isfinite(atr[i])
                and atr[i] > 0
            )
            else np.nan
        )

        threshold_ok = (
            depth > 0
            and (
                params.min_depth_atr == 0
                or (
                    np.isfinite(depth_atr)
                    and depth_atr
                    >= params.min_depth_atr
                )
            )
        )

        if (
            source_idx
            not in consumed_low
            and threshold_ok
            and close[i] > acc_low[i]
        ):

            out.at[
                label,
                "manipulation_low",
            ] = True

            out.at[
                label,
                "manipulation_low_level",
            ] = acc_low[i]

            out.at[
                label,
                "manipulation_low_depth",
            ] = depth

            out.at[
                label,
                "manipulation_low_depth_atr",
            ] = depth_atr

            out.at[
                label,
                "manipulation_low_close_back",
            ] = (
                close[i]
                - acc_low[i]
            )

            out.at[
                label,
                "manipulation_low_source_accumulation_idx",
            ] = source_idx

            out.at[
                label,
                "manipulation_low_confirmation_idx",
            ] = i

            consumed_low.add(
                source_idx
            )

    return out
