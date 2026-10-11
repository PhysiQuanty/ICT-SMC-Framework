from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HTFPDArrayParams:
    """
    HTF PD Array context v0.1.

    Consomme les zones HTF déjà calculées.

    Sources supportées:
        FVG
        Order Block
        OTE

    On cherche si le prix courant HTF se trouve dans une zone,
    et on expose ensuite cette information au LTF via alignement causal.

    priority:
        ordre utilisé quand plusieurs zones se chevauchent.

    max_age_bars:
        âge maximal d'une zone HTF.
    """

    sources: tuple[str, ...] = (
        "order_block",
        "fvg",
        "ote",
    )

    priority: tuple[str, ...] = (
        "order_block",
        "fvg",
        "ote",
    )

    max_age_bars: int = 50

    def __post_init__(self) -> None:

        allowed = {
            "order_block",
            "fvg",
            "ote",
        }

        unknown = (
            set(self.sources)
            - allowed
        )

        if unknown:
            raise ValueError(
                f"sources inconnues: {sorted(unknown)}"
            )

        if self.max_age_bars < 1:
            raise ValueError(
                "max_age_bars doit être >= 1"
            )


def detect_htf_pd_array(
    df: pd.DataFrame,
    params: HTFPDArrayParams = HTFPDArrayParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",
    }

    if "order_block" in params.sources:

        required.update(
            {
                "ob_bull",
                "ob_bear",

                "ob_bull_low",
                "ob_bull_high",

                "ob_bear_low",
                "ob_bear_high",
            }
        )

    if "fvg" in params.sources:

        required.update(
            {
                "fvg_bull",
                "fvg_bear",

                "fvg_bull_low",
                "fvg_bull_high",

                "fvg_bear_low",
                "fvg_bear_high",
            }
        )

    if "ote" in params.sources:

        required.update(
            {
                "ote_bull_active",
                "ote_bear_active",

                "ote_bull_low",
                "ote_bull_high",

                "ote_bear_low",
                "ote_bear_high",
            }
        )

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_htf_pd_array(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    close = out[
        "close"
    ].to_numpy(
        dtype=np.float64,
    )

    out["htf_pd_array_bull"] = False
    out["htf_pd_array_bear"] = False

    out["htf_pd_array_type"] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out["htf_pd_array_low"] = np.nan
    out["htf_pd_array_high"] = np.nan
    out["htf_pd_array_mid"] = np.nan

    out["htf_pd_array_position"] = np.nan

    out["htf_pd_array_source_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["htf_pd_array_age_bars"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    bull_zones = []
    bear_zones = []

    # =========================================================================
    # PRIORITY MAP
    # =========================================================================

    priority = {
        name:
            i
        for i, name in enumerate(
            params.priority
        )
    }

    def add_zone(
        container: list,
        zone_type: str,
        idx: int,
        low: float,
        high: float,
    ) -> None:

        if (
            not np.isfinite(low)
            or not np.isfinite(high)
            or high <= low
        ):
            return

        container.append(
            {
                "type":
                    zone_type,

                "idx":
                    idx,

                "low":
                    float(low),

                "high":
                    float(high),
            }
        )

    for i in range(n):

        # =====================================================================
        # EXPIRE
        # =====================================================================

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

        row = out.iloc[i]

        # =====================================================================
        # ADD NEW HTF PD ARRAYS
        # =====================================================================

        if "order_block" in params.sources:

            if bool(
                row["ob_bull"]
            ):

                add_zone(
                    bull_zones,
                    "order_block",
                    i,
                    row[
                        "ob_bull_low"
                    ],
                    row[
                        "ob_bull_high"
                    ],
                )

            if bool(
                row["ob_bear"]
            ):

                add_zone(
                    bear_zones,
                    "order_block",
                    i,
                    row[
                        "ob_bear_low"
                    ],
                    row[
                        "ob_bear_high"
                    ],
                )

        if "fvg" in params.sources:

            if bool(
                row["fvg_bull"]
            ):

                add_zone(
                    bull_zones,
                    "fvg",
                    i,
                    row[
                        "fvg_bull_low"
                    ],
                    row[
                        "fvg_bull_high"
                    ],
                )

            if bool(
                row["fvg_bear"]
            ):

                add_zone(
                    bear_zones,
                    "fvg",
                    i,
                    row[
                        "fvg_bear_low"
                    ],
                    row[
                        "fvg_bear_high"
                    ],
                )

        # OTE is already persistent, so only add on source-range changes.
        if "ote" in params.sources:

            if bool(
                row[
                    "ote_bull_active"
                ]
            ):

                source = row.get(
                    "ote_bull_source_range_idx",
                    pd.NA,
                )

                source = (
                    int(source)
                    if pd.notna(source)
                    else i
                )

                exists = any(
                    (
                        z["type"] == "ote"
                        and z["idx"] == source
                    )
                    for z in bull_zones
                )

                if not exists:

                    add_zone(
                        bull_zones,
                        "ote",
                        source,
                        row[
                            "ote_bull_low"
                        ],
                        row[
                            "ote_bull_high"
                        ],
                    )

            if bool(
                row[
                    "ote_bear_active"
                ]
            ):

                source = row.get(
                    "ote_bear_source_range_idx",
                    pd.NA,
                )

                source = (
                    int(source)
                    if pd.notna(source)
                    else i
                )

                exists = any(
                    (
                        z["type"] == "ote"
                        and z["idx"] == source
                    )
                    for z in bear_zones
                )

                if not exists:

                    add_zone(
                        bear_zones,
                        "ote",
                        source,
                        row[
                            "ote_bear_low"
                        ],
                        row[
                            "ote_bear_high"
                        ],
                    )

        # =====================================================================
        # PRICE INSIDE ZONE
        # =====================================================================

        price = close[i]

        candidates = []

        for side, zones in (
            ("bull", bull_zones),
            ("bear", bear_zones),
        ):

            for zone in zones:

                if (
                    zone["low"]
                    <= price
                    <= zone["high"]
                ):

                    candidates.append(
                        {
                            **zone,
                            "side":
                                side,
                        }
                    )

        if not candidates:
            continue

        # Prefer explicit type priority, then latest zone.
        selected = min(
            candidates,
            key=lambda z: (
                priority.get(
                    z["type"],
                    999,
                ),
                -z["idx"],
            ),
        )

        label = out.index[i]

        low_value = selected[
            "low"
        ]

        high_value = selected[
            "high"
        ]

        size = (
            high_value
            - low_value
        )

        out.at[
            label,
            "htf_pd_array_bull",
        ] = (
            selected["side"]
            == "bull"
        )

        out.at[
            label,
            "htf_pd_array_bear",
        ] = (
            selected["side"]
            == "bear"
        )

        out.at[
            label,
            "htf_pd_array_type",
        ] = selected[
            "type"
        ]

        out.at[
            label,
            "htf_pd_array_low",
        ] = low_value

        out.at[
            label,
            "htf_pd_array_high",
        ] = high_value

        out.at[
            label,
            "htf_pd_array_mid",
        ] = (
            low_value
            + high_value
        ) / 2.0

        out.at[
            label,
            "htf_pd_array_position",
        ] = (
            (
                price
                - low_value
            )
            / size
            if size > 0
            else np.nan
        )

        out.at[
            label,
            "htf_pd_array_source_idx",
        ] = selected[
            "idx"
        ]

        out.at[
            label,
            "htf_pd_array_age_bars",
        ] = (
            i
            - selected[
                "idx"
            ]
        )

    return out
