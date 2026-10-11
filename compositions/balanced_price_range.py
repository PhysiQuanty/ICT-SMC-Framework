from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BalancedPriceRangeParams:
    """
    Balanced Price Range (BPR) v0.1.

    Proxy algorithmique:
        overlap spatial entre deux FVG de directions opposées.

    Exemple:
        FVG bull = [100, 105]
        FVG bear = [103, 108]

        BPR = [103, 105]

    max_age_bars:
        âge maximal d'une FVG candidate.

    min_overlap_atr:
        overlap minimal normalisé par ATR.

    consume_pair:
        si True, une paire utilisée ne produit qu'un seul BPR.
    """

    max_age_bars: int = 50
    min_overlap_atr: float = 0.0

    atr_column: str = "atr"

    consume_pair: bool = True

    def __post_init__(self) -> None:

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )

        if self.min_overlap_atr < 0:
            raise ValueError(
                "min_overlap_atr doit être >= 0"
            )


def detect_balanced_price_range(
    df: pd.DataFrame,
    params: BalancedPriceRangeParams = BalancedPriceRangeParams(),
) -> pd.DataFrame:

    required = {
        "fvg_bull",
        "fvg_bear",

        "fvg_bull_low",
        "fvg_bull_high",

        "fvg_bear_low",
        "fvg_bear_high",
    }

    if params.min_overlap_atr > 0:
        required.add(
            params.atr_column
        )

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_balanced_price_range(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["bpr"] = False

    out["bpr_low"] = np.nan
    out["bpr_high"] = np.nan
    out["bpr_mid"] = np.nan

    out["bpr_size"] = np.nan
    out["bpr_size_atr"] = np.nan

    out["bpr_bull_source_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["bpr_bear_source_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["bpr_confirmation_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["bpr_source_gap_bars"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
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

    bull_zones = []
    bear_zones = []

    used_pairs = set()

    for i in range(n):

        # Remove old zones.
        bull_zones = [
            z
            for z in bull_zones
            if (
                i - z["idx"]
                <= params.max_age_bars
            )
        ]

        bear_zones = [
            z
            for z in bear_zones
            if (
                i - z["idx"]
                <= params.max_age_bars
            )
        ]

        new_candidates = []

        # =====================================================================
        # NEW BULL FVG AGAINST PREVIOUS BEAR FVGs
        # =====================================================================

        if bool(
            out.iloc[i][
                "fvg_bull"
            ]
        ):

            bull = {
                "idx":
                    i,

                "low":
                    float(
                        out.iloc[i][
                            "fvg_bull_low"
                        ]
                    ),

                "high":
                    float(
                        out.iloc[i][
                            "fvg_bull_high"
                        ]
                    ),
            }

            for bear in bear_zones:

                overlap_low = max(
                    bull["low"],
                    bear["low"],
                )

                overlap_high = min(
                    bull["high"],
                    bear["high"],
                )

                size = (
                    overlap_high
                    - overlap_low
                )

                if size <= 0:
                    continue

                size_atr = (
                    size / atr[i]
                    if (
                        np.isfinite(atr[i])
                        and atr[i] > 0
                    )
                    else np.nan
                )

                if (
                    params.min_overlap_atr > 0
                    and (
                        not np.isfinite(
                            size_atr
                        )
                        or (
                            size_atr
                            < params.min_overlap_atr
                        )
                    )
                ):
                    continue

                pair = (
                    bull["idx"],
                    bear["idx"],
                )

                if (
                    params.consume_pair
                    and pair in used_pairs
                ):
                    continue

                new_candidates.append(
                    {
                        "bull_idx":
                            bull["idx"],

                        "bear_idx":
                            bear["idx"],

                        "low":
                            overlap_low,

                        "high":
                            overlap_high,

                        "size":
                            size,

                        "size_atr":
                            size_atr,
                    }
                )

            bull_zones.append(
                bull
            )

        # =====================================================================
        # NEW BEAR FVG AGAINST PREVIOUS BULL FVGs
        # =====================================================================

        if bool(
            out.iloc[i][
                "fvg_bear"
            ]
        ):

            bear = {
                "idx":
                    i,

                "low":
                    float(
                        out.iloc[i][
                            "fvg_bear_low"
                        ]
                    ),

                "high":
                    float(
                        out.iloc[i][
                            "fvg_bear_high"
                        ]
                    ),
            }

            for bull in bull_zones:

                # Prevent comparing two FVGs born on same row
                # through the append above if both somehow exist.
                if bull["idx"] == i:
                    continue

                overlap_low = max(
                    bull["low"],
                    bear["low"],
                )

                overlap_high = min(
                    bull["high"],
                    bear["high"],
                )

                size = (
                    overlap_high
                    - overlap_low
                )

                if size <= 0:
                    continue

                size_atr = (
                    size / atr[i]
                    if (
                        np.isfinite(atr[i])
                        and atr[i] > 0
                    )
                    else np.nan
                )

                if (
                    params.min_overlap_atr > 0
                    and (
                        not np.isfinite(
                            size_atr
                        )
                        or (
                            size_atr
                            < params.min_overlap_atr
                        )
                    )
                ):
                    continue

                pair = (
                    bull["idx"],
                    bear["idx"],
                )

                if (
                    params.consume_pair
                    and pair in used_pairs
                ):
                    continue

                new_candidates.append(
                    {
                        "bull_idx":
                            bull["idx"],

                        "bear_idx":
                            bear["idx"],

                        "low":
                            overlap_low,

                        "high":
                            overlap_high,

                        "size":
                            size,

                        "size_atr":
                            size_atr,
                    }
                )

            bear_zones.append(
                bear
            )

        if not new_candidates:
            continue

        # If several BPRs appear on same candle,
        # expose the largest overlap on this row.
        best = max(
            new_candidates,
            key=lambda x: x["size"],
        )

        pair = (
            best["bull_idx"],
            best["bear_idx"],
        )

        used_pairs.add(
            pair
        )

        label = out.index[i]

        out.at[
            label,
            "bpr",
        ] = True

        out.at[
            label,
            "bpr_low",
        ] = best["low"]

        out.at[
            label,
            "bpr_high",
        ] = best["high"]

        out.at[
            label,
            "bpr_mid",
        ] = (
            best["low"]
            + best["high"]
        ) / 2.0

        out.at[
            label,
            "bpr_size",
        ] = best["size"]

        out.at[
            label,
            "bpr_size_atr",
        ] = best["size_atr"]

        out.at[
            label,
            "bpr_bull_source_idx",
        ] = best[
            "bull_idx"
        ]

        out.at[
            label,
            "bpr_bear_source_idx",
        ] = best[
            "bear_idx"
        ]

        out.at[
            label,
            "bpr_confirmation_idx",
        ] = i

        out.at[
            label,
            "bpr_source_gap_bars",
        ] = abs(
            best["bull_idx"]
            - best["bear_idx"]
        )

    return out
