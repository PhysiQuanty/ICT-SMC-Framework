from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExternalInternalLiquidityParams:
    """
    External / Internal Liquidity v0.1.

    External liquidity:
        niveaux situés aux bornes ou hors du Dealing Range courant.

    Internal liquidity:
        niveaux situés strictement à l'intérieur du Dealing Range.

    Sources optionnelles:
        - dealing range high/low
        - PDH/PDL
        - PWH/PWL
        - Asian H/L
        - equal highs/lows
        - FVG midpoints

    La classification est purement géométrique.
    Aucun intent / aucune stratégie.
    """

    include_previous_day: bool = True
    include_previous_week: bool = True
    include_asian: bool = True
    include_equal_high_low: bool = True
    include_fvg: bool = True

    boundary_tolerance_fraction: float = 0.0

    def __post_init__(self) -> None:

        if self.boundary_tolerance_fraction < 0:
            raise ValueError(
                "boundary_tolerance_fraction doit être >= 0"
            )


def detect_external_internal_liquidity(
    df: pd.DataFrame,
    params: ExternalInternalLiquidityParams = (
        ExternalInternalLiquidityParams()
    ),
) -> pd.DataFrame:

    required = {
        "dealing_range_active",
        "dealing_range_low",
        "dealing_range_high",
        "dealing_range_mid",
        "dealing_range_size",
    }

    optional_requirements = []

    if params.include_previous_day:

        optional_requirements += [
            "previous_day_high",
            "previous_day_low",
        ]

    if params.include_previous_week:

        optional_requirements += [
            "previous_week_high",
            "previous_week_low",
        ]

    if params.include_asian:

        optional_requirements += [
            "asian_high",
            "asian_low",
        ]

    if params.include_equal_high_low:

        optional_requirements += [
            "equal_high",
            "equal_low",
            "equal_high_level",
            "equal_low_level",
        ]

    if params.include_fvg:

        optional_requirements += [
            "fvg_bull",
            "fvg_bear",
            "fvg_bull_mid",
            "fvg_bear_mid",
        ]

    required.update(
        optional_requirements
    )

    missing = required.difference(
        df.columns
    )

    if missing:

        raise ValueError(
            "detect_external_internal_liquidity(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    active = out[
        "dealing_range_active"
    ].to_numpy(dtype=bool)

    range_low = out[
        "dealing_range_low"
    ].to_numpy(dtype=np.float64)

    range_high = out[
        "dealing_range_high"
    ].to_numpy(dtype=np.float64)

    range_mid = out[
        "dealing_range_mid"
    ].to_numpy(dtype=np.float64)

    range_size = out[
        "dealing_range_size"
    ].to_numpy(dtype=np.float64)

    # =========================================================================
    # CORE EXTERNAL LEVELS
    # =========================================================================

    out["external_liquidity_high"] = np.where(
        active,
        range_high,
        np.nan,
    )

    out["external_liquidity_low"] = np.where(
        active,
        range_low,
        np.nan,
    )

    out["external_liquidity_mid"] = np.where(
        active,
        range_mid,
        np.nan,
    )

    # =========================================================================
    # COUNTS
    # =========================================================================

    out["external_liquidity_above_count"] = 0
    out["external_liquidity_below_count"] = 0

    out["internal_liquidity_count"] = 0

    out["nearest_external_above"] = np.nan
    out["nearest_external_below"] = np.nan

    out["nearest_internal_above"] = np.nan
    out["nearest_internal_below"] = np.nan

    out["nearest_external_above_type"] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out["nearest_external_below_type"] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out["nearest_internal_above_type"] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out["nearest_internal_below_type"] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    close = out[
        "close"
    ].to_numpy(dtype=np.float64)

    # =========================================================================
    # ROW-BY-ROW CLASSIFICATION
    # =========================================================================

    for i in range(n):

        if not active[i]:
            continue

        lo = range_low[i]
        hi = range_high[i]
        size = range_size[i]

        if (
            not np.isfinite(lo)
            or not np.isfinite(hi)
            or not np.isfinite(size)
            or size <= 0
        ):
            continue

        tolerance = (
            size
            * params.boundary_tolerance_fraction
        )

        internal_low = (
            lo + tolerance
        )

        internal_high = (
            hi - tolerance
        )

        price = close[i]

        candidates: list[
            tuple[
                str,
                float,
            ]
        ] = []

        # Dealing-range boundaries are always external.
        candidates.append(
            (
                "dealing_range_high",
                hi,
            )
        )

        candidates.append(
            (
                "dealing_range_low",
                lo,
            )
        )

        if params.include_previous_day:

            pdh = out.iloc[i][
                "previous_day_high"
            ]

            pdl = out.iloc[i][
                "previous_day_low"
            ]

            if pd.notna(pdh):

                candidates.append(
                    (
                        "previous_day_high",
                        float(pdh),
                    )
                )

            if pd.notna(pdl):

                candidates.append(
                    (
                        "previous_day_low",
                        float(pdl),
                    )
                )

        if params.include_previous_week:

            pwh = out.iloc[i][
                "previous_week_high"
            ]

            pwl = out.iloc[i][
                "previous_week_low"
            ]

            if pd.notna(pwh):

                candidates.append(
                    (
                        "previous_week_high",
                        float(pwh),
                    )
                )

            if pd.notna(pwl):

                candidates.append(
                    (
                        "previous_week_low",
                        float(pwl),
                    )
                )

        if params.include_asian:

            ah = out.iloc[i][
                "asian_high"
            ]

            al = out.iloc[i][
                "asian_low"
            ]

            if pd.notna(ah):

                candidates.append(
                    (
                        "asian_high",
                        float(ah),
                    )
                )

            if pd.notna(al):

                candidates.append(
                    (
                        "asian_low",
                        float(al),
                    )
                )

        if params.include_equal_high_low:

            if bool(
                out.iloc[i][
                    "equal_high"
                ]
            ):

                level = out.iloc[i][
                    "equal_high_level"
                ]

                if pd.notna(level):

                    candidates.append(
                        (
                            "equal_high",
                            float(level),
                        )
                    )

            if bool(
                out.iloc[i][
                    "equal_low"
                ]
            ):

                level = out.iloc[i][
                    "equal_low_level"
                ]

                if pd.notna(level):

                    candidates.append(
                        (
                            "equal_low",
                            float(level),
                        )
                    )

        if params.include_fvg:

            if bool(
                out.iloc[i][
                    "fvg_bull"
                ]
            ):

                level = out.iloc[i][
                    "fvg_bull_mid"
                ]

                if pd.notna(level):

                    candidates.append(
                        (
                            "fvg_bull_mid",
                            float(level),
                        )
                    )

            if bool(
                out.iloc[i][
                    "fvg_bear"
                ]
            ):

                level = out.iloc[i][
                    "fvg_bear_mid"
                ]

                if pd.notna(level):

                    candidates.append(
                        (
                            "fvg_bear_mid",
                            float(level),
                        )
                    )

        external_above = []
        external_below = []

        internal_above = []
        internal_below = []

        for name, level in candidates:

            if not np.isfinite(level):
                continue

            # Range boundaries are always external.
            if name in {
                "dealing_range_high",
                "dealing_range_low",
            }:

                is_external = True

            else:

                is_internal = (
                    level > internal_low
                    and level < internal_high
                )

                is_external = (
                    not is_internal
                )

            if is_external:

                if level > price:

                    external_above.append(
                        (
                            level,
                            name,
                        )
                    )

                elif level < price:

                    external_below.append(
                        (
                            level,
                            name,
                        )
                    )

            else:

                if level > price:

                    internal_above.append(
                        (
                            level,
                            name,
                        )
                    )

                elif level < price:

                    internal_below.append(
                        (
                            level,
                            name,
                        )
                    )

        label = out.index[i]

        out.at[
            label,
            "external_liquidity_above_count",
        ] = len(
            external_above
        )

        out.at[
            label,
            "external_liquidity_below_count",
        ] = len(
            external_below
        )

        out.at[
            label,
            "internal_liquidity_count",
        ] = (
            len(internal_above)
            + len(internal_below)
        )

        if external_above:

            level, name = min(
                external_above,
                key=lambda x: x[0],
            )

            out.at[
                label,
                "nearest_external_above",
            ] = level

            out.at[
                label,
                "nearest_external_above_type",
            ] = name

        if external_below:

            level, name = max(
                external_below,
                key=lambda x: x[0],
            )

            out.at[
                label,
                "nearest_external_below",
            ] = level

            out.at[
                label,
                "nearest_external_below_type",
            ] = name

        if internal_above:

            level, name = min(
                internal_above,
                key=lambda x: x[0],
            )

            out.at[
                label,
                "nearest_internal_above",
            ] = level

            out.at[
                label,
                "nearest_internal_above_type",
            ] = name

        if internal_below:

            level, name = max(
                internal_below,
                key=lambda x: x[0],
            )

            out.at[
                label,
                "nearest_internal_below",
            ] = level

            out.at[
                label,
                "nearest_internal_below_type",
            ] = name

    return out
