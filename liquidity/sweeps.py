from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SweepParams:
    """
    Liquidity Sweep.

    Bearish liquidity sweep:
        wick au-dessus d'une BSL active
        puis clôture sous le niveau.

    Bullish liquidity sweep:
        wick sous une SSL active
        puis clôture au-dessus du niveau.

    min_depth_atr:
        profondeur minimale du dépassement.

    require_close_back_inside:
        si False:
            toute prise du niveau peut être considérée sweep.
        si True:
            prix doit clôturer de nouveau derrière le niveau.
    """

    min_depth_atr: float = 0.0

    atr_column: str = "atr"

    require_close_back_inside: bool = True

    def __post_init__(self) -> None:

        if self.min_depth_atr < 0:
            raise ValueError(
                "min_depth_atr doit être >= 0"
            )


def detect_sweeps(
    df: pd.DataFrame,
    params: SweepParams = SweepParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",

        "bsl_price",
        "ssl_price",

        "bsl_active",
        "ssl_active",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_sweeps(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()

    n = len(out)

    out["sweep_bear"] = False
    out["sweep_bull"] = False

    out["sweep_bear_level"] = np.nan
    out["sweep_bull_level"] = np.nan

    out["sweep_bear_depth"] = np.nan
    out["sweep_bull_depth"] = np.nan

    out["sweep_bear_depth_atr"] = np.nan
    out["sweep_bull_depth_atr"] = np.nan

    out["sweep_bear_close_back"] = np.nan
    out["sweep_bull_close_back"] = np.nan

    out["sweep_bear_close_back_atr"] = np.nan
    out["sweep_bull_close_back_atr"] = np.nan

    for prefix in (
        "sweep_bear",
        "sweep_bull",
    ):

        out[
            f"{prefix}_source_confirmation_idx"
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
            f"{prefix}_source_age_bars"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    high = out["high"].to_numpy(
        dtype=np.float64,
    )

    low = out["low"].to_numpy(
        dtype=np.float64,
    )

    close = out["close"].to_numpy(
        dtype=np.float64,
    )

    bsl_price = out[
        "bsl_price"
    ].to_numpy(
        dtype=np.float64,
    )

    ssl_price = out[
        "ssl_price"
    ].to_numpy(
        dtype=np.float64,
    )

    bsl_active = out[
        "bsl_active"
    ].to_numpy(
        dtype=bool,
    )

    ssl_active = out[
        "ssl_active"
    ].to_numpy(
        dtype=bool,
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

    for i in range(1, n):

        label = out.index[i]

        # =====================================================================
        # BEARISH SWEEP OF PREVIOUS BSL
        # =====================================================================

        if (
            bsl_active[i - 1]
            and np.isfinite(
                bsl_price[i - 1]
            )
        ):

            level = (
                bsl_price[i - 1]
            )

            depth = (
                high[i] - level
            )

            depth_atr = (
                depth / atr[i]
                if (
                    np.isfinite(atr[i])
                    and atr[i] > 0
                )
                else np.nan
            )

            crossed = (
                depth > 0
            )

            threshold = (
                params.min_depth_atr == 0
                or (
                    np.isfinite(depth_atr)
                    and (
                        depth_atr
                        >= params.min_depth_atr
                    )
                )
            )

            close_back = (
                level - close[i]
            )

            rejection_ok = (
                close[i] < level
                if params.require_close_back_inside
                else True
            )

            if (
                crossed
                and threshold
                and rejection_ok
            ):

                out.at[
                    label,
                    "sweep_bear",
                ] = True

                out.at[
                    label,
                    "sweep_bear_level",
                ] = level

                out.at[
                    label,
                    "sweep_bear_depth",
                ] = depth

                out.at[
                    label,
                    "sweep_bear_depth_atr",
                ] = depth_atr

                out.at[
                    label,
                    "sweep_bear_close_back",
                ] = close_back

                out.at[
                    label,
                    "sweep_bear_close_back_atr",
                ] = (
                    close_back / atr[i]
                    if (
                        np.isfinite(atr[i])
                        and atr[i] > 0
                    )
                    else np.nan
                )

                source_idx = (
                    out.iloc[i - 1][
                        "bsl_confirmation_idx"
                    ]
                )

                if pd.notna(source_idx):

                    source_idx = int(
                        source_idx
                    )

                    out.at[
                        label,
                        "sweep_bear_source_confirmation_idx",
                    ] = source_idx

                    out.at[
                        label,
                        "sweep_bear_source_age_bars",
                    ] = (
                        i - source_idx
                    )

                out.at[
                    label,
                    "sweep_bear_confirmation_idx",
                ] = i

        # =====================================================================
        # BULLISH SWEEP OF PREVIOUS SSL
        # =====================================================================

        if (
            ssl_active[i - 1]
            and np.isfinite(
                ssl_price[i - 1]
            )
        ):

            level = (
                ssl_price[i - 1]
            )

            depth = (
                level - low[i]
            )

            depth_atr = (
                depth / atr[i]
                if (
                    np.isfinite(atr[i])
                    and atr[i] > 0
                )
                else np.nan
            )

            crossed = (
                depth > 0
            )

            threshold = (
                params.min_depth_atr == 0
                or (
                    np.isfinite(depth_atr)
                    and (
                        depth_atr
                        >= params.min_depth_atr
                    )
                )
            )

            close_back = (
                close[i] - level
            )

            rejection_ok = (
                close[i] > level
                if params.require_close_back_inside
                else True
            )

            if (
                crossed
                and threshold
                and rejection_ok
            ):

                out.at[
                    label,
                    "sweep_bull",
                ] = True

                out.at[
                    label,
                    "sweep_bull_level",
                ] = level

                out.at[
                    label,
                    "sweep_bull_depth",
                ] = depth

                out.at[
                    label,
                    "sweep_bull_depth_atr",
                ] = depth_atr

                out.at[
                    label,
                    "sweep_bull_close_back",
                ] = close_back

                out.at[
                    label,
                    "sweep_bull_close_back_atr",
                ] = (
                    close_back / atr[i]
                    if (
                        np.isfinite(atr[i])
                        and atr[i] > 0
                    )
                    else np.nan
                )

                source_idx = (
                    out.iloc[i - 1][
                        "ssl_confirmation_idx"
                    ]
                )

                if pd.notna(source_idx):

                    source_idx = int(
                        source_idx
                    )

                    out.at[
                        label,
                        "sweep_bull_source_confirmation_idx",
                    ] = source_idx

                    out.at[
                        label,
                        "sweep_bull_source_age_bars",
                    ] = (
                        i - source_idx
                    )

                out.at[
                    label,
                    "sweep_bull_confirmation_idx",
                ] = i

    return out
