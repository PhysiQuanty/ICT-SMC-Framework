from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NestedFVGParams:
    """
    Nested FVG v0.1.

    Une nouvelle FVG est dite nested si elle est contenue
    dans une FVG antérieure de même direction.

    containment_ratio:
        fraction minimale de la nouvelle FVG située à l'intérieur
        de la FVG parent.

        1.0 = containment complet.
        0.5 = au moins 50%.

    max_parent_age_bars:
        âge maximal du parent.
    """

    containment_ratio: float = 1.0

    max_parent_age_bars: int = 100

    def __post_init__(self) -> None:

        if not (
            0 < self.containment_ratio <= 1
        ):
            raise ValueError(
                "containment_ratio doit être dans ]0,1]"
            )

        if self.max_parent_age_bars < 1:
            raise ValueError(
                "max_parent_age_bars doit être >= 1"
            )


def _containment(
    child_low: float,
    child_high: float,
    parent_low: float,
    parent_high: float,
) -> float:

    child_size = (
        child_high
        - child_low
    )

    if child_size <= 0:
        return 0.0

    overlap = max(
        0.0,
        min(
            child_high,
            parent_high,
        )
        - max(
            child_low,
            parent_low,
        ),
    )

    return (
        overlap
        / child_size
    )


def detect_nested_fvg(
    df: pd.DataFrame,
    params: NestedFVGParams = NestedFVGParams(),
) -> pd.DataFrame:

    required = {
        "fvg_bull",
        "fvg_bear",

        "fvg_bull_low",
        "fvg_bull_high",

        "fvg_bear_low",
        "fvg_bear_high",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_nested_fvg(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    for side in (
        "nested_fvg_bull",
        "nested_fvg_bear",
    ):

        out[side] = False

        out[f"{side}_low"] = np.nan
        out[f"{side}_high"] = np.nan
        out[f"{side}_mid"] = np.nan

        out[
            f"{side}_containment_ratio"
        ] = np.nan

        out[
            f"{side}_parent_size_ratio"
        ] = np.nan

        for suffix in (
            "parent_idx",
            "confirmation_idx",
            "parent_age_bars",
        ):

            out[f"{side}_{suffix}"] = pd.Series(
                pd.array(
                    [pd.NA] * n,
                    dtype="Int64",
                ),
                index=out.index,
            )

    bull_history = []
    bear_history = []

    for i in range(n):

        # Drop old parents.
        bull_history = [
            z
            for z in bull_history
            if (
                i - z["idx"]
                <= params.max_parent_age_bars
            )
        ]

        bear_history = [
            z
            for z in bear_history
            if (
                i - z["idx"]
                <= params.max_parent_age_bars
            )
        ]

        row = out.iloc[i]
        label = out.index[i]

        # =====================================================================
        # BULL CHILD
        # =====================================================================

        if bool(
            row["fvg_bull"]
        ):

            child_low = float(
                row[
                    "fvg_bull_low"
                ]
            )

            child_high = float(
                row[
                    "fvg_bull_high"
                ]
            )

            child_size = (
                child_high
                - child_low
            )

            candidates = []

            for parent in bull_history:

                ratio = _containment(
                    child_low,
                    child_high,
                    parent["low"],
                    parent["high"],
                )

                if (
                    ratio
                    >= params.containment_ratio
                ):

                    candidates.append(
                        (
                            ratio,
                            parent,
                        )
                    )

            if candidates:

                # Prefer most recent parent,
                # then highest containment.
                ratio, parent = max(
                    candidates,
                    key=lambda x: (
                        x[1]["idx"],
                        x[0],
                    ),
                )

                parent_size = (
                    parent["high"]
                    - parent["low"]
                )

                out.at[
                    label,
                    "nested_fvg_bull",
                ] = True

                out.at[
                    label,
                    "nested_fvg_bull_low",
                ] = child_low

                out.at[
                    label,
                    "nested_fvg_bull_high",
                ] = child_high

                out.at[
                    label,
                    "nested_fvg_bull_mid",
                ] = (
                    child_low
                    + child_high
                ) / 2.0

                out.at[
                    label,
                    "nested_fvg_bull_containment_ratio",
                ] = ratio

                out.at[
                    label,
                    "nested_fvg_bull_parent_size_ratio",
                ] = (
                    child_size / parent_size
                    if parent_size > 0
                    else np.nan
                )

                out.at[
                    label,
                    "nested_fvg_bull_parent_idx",
                ] = parent[
                    "idx"
                ]

                out.at[
                    label,
                    "nested_fvg_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "nested_fvg_bull_parent_age_bars",
                ] = (
                    i
                    - parent["idx"]
                )

            bull_history.append(
                {
                    "idx":
                        i,

                    "low":
                        child_low,

                    "high":
                        child_high,
                }
            )

        # =====================================================================
        # BEAR CHILD
        # =====================================================================

        if bool(
            row["fvg_bear"]
        ):

            child_low = float(
                row[
                    "fvg_bear_low"
                ]
            )

            child_high = float(
                row[
                    "fvg_bear_high"
                ]
            )

            child_size = (
                child_high
                - child_low
            )

            candidates = []

            for parent in bear_history:

                ratio = _containment(
                    child_low,
                    child_high,
                    parent["low"],
                    parent["high"],
                )

                if (
                    ratio
                    >= params.containment_ratio
                ):

                    candidates.append(
                        (
                            ratio,
                            parent,
                        )
                    )

            if candidates:

                ratio, parent = max(
                    candidates,
                    key=lambda x: (
                        x[1]["idx"],
                        x[0],
                    ),
                )

                parent_size = (
                    parent["high"]
                    - parent["low"]
                )

                out.at[
                    label,
                    "nested_fvg_bear",
                ] = True

                out.at[
                    label,
                    "nested_fvg_bear_low",
                ] = child_low

                out.at[
                    label,
                    "nested_fvg_bear_high",
                ] = child_high

                out.at[
                    label,
                    "nested_fvg_bear_mid",
                ] = (
                    child_low
                    + child_high
                ) / 2.0

                out.at[
                    label,
                    "nested_fvg_bear_containment_ratio",
                ] = ratio

                out.at[
                    label,
                    "nested_fvg_bear_parent_size_ratio",
                ] = (
                    child_size / parent_size
                    if parent_size > 0
                    else np.nan
                )

                out.at[
                    label,
                    "nested_fvg_bear_parent_idx",
                ] = parent[
                    "idx"
                ]

                out.at[
                    label,
                    "nested_fvg_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "nested_fvg_bear_parent_age_bars",
                ] = (
                    i
                    - parent["idx"]
                )

            bear_history.append(
                {
                    "idx":
                        i,

                    "low":
                        child_low,

                    "high":
                        child_high,
                }
            )

    return out
