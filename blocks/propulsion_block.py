from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PropulsionBlockParams:
    """
    Propulsion Block proxy.

    On cherche une petite bougie directionnelle immédiatement
    avant une continuation structurelle avec displacement.

    Bullish:
        bougie source bullish
        puis, dans max_lag:
            BOS bullish + displacement bullish.

    Bearish:
        symétrique.

    max_source_body_atr:
        la bougie source doit rester relativement compacte.

    min_confirmation_displacement_atr:
        displacement minimal sur la confirmation.
    """

    max_lag: int = 3

    max_source_body_atr: float = 1.0

    min_confirmation_displacement_atr: float = 1.0

    atr_column: str = "atr"

    def __post_init__(self) -> None:

        if self.max_lag < 1:
            raise ValueError(
                "max_lag doit être >= 1"
            )

        if self.max_source_body_atr < 0:
            raise ValueError(
                "max_source_body_atr doit être >= 0"
            )


def detect_propulsion_blocks(
    df: pd.DataFrame,
    params: PropulsionBlockParams = PropulsionBlockParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "high",
        "low",
        "close",

        "bos_bull",
        "bos_bear",

        "displacement_bull",
        "displacement_bear",

        "displacement_body_atr",

        params.atr_column,
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_propulsion_blocks(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    for side in (
        "propulsion_bull",
        "propulsion_bear",
    ):

        out[side] = False

        out[f"{side}_low"] = np.nan
        out[f"{side}_high"] = np.nan
        out[f"{side}_mid"] = np.nan

        out[f"{side}_source_body_atr"] = np.nan
        out[f"{side}_confirmation_displacement_atr"] = np.nan

        for suffix in (
            "origin_idx",
            "confirmation_idx",
            "confirmation_lag",
        ):

            out[f"{side}_{suffix}"] = pd.Series(
                pd.array(
                    [pd.NA] * n,
                    dtype="Int64",
                ),
                index=out.index,
            )

    open_ = out["open"].to_numpy(
        dtype=np.float64,
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

    atr = out[
        params.atr_column
    ].to_numpy(
        dtype=np.float64,
    )

    bos_bull = out[
        "bos_bull"
    ].to_numpy(
        dtype=bool,
    )

    bos_bear = out[
        "bos_bear"
    ].to_numpy(
        dtype=bool,
    )

    disp_bull = out[
        "displacement_bull"
    ].to_numpy(
        dtype=bool,
    )

    disp_bear = out[
        "displacement_bear"
    ].to_numpy(
        dtype=bool,
    )

    disp_body_atr = out[
        "displacement_body_atr"
    ].to_numpy(
        dtype=np.float64,
    )

    for i in range(n):

        # =====================================================================
        # BULL
        # =====================================================================

        if (
            bos_bull[i]
            and disp_bull[i]
            and np.isfinite(
                disp_body_atr[i]
            )
            and (
                disp_body_atr[i]
                >= params.min_confirmation_displacement_atr
            )
        ):

            source = None

            for j in range(
                i - 1,
                max(
                    -1,
                    i - params.max_lag - 1,
                ),
                -1,
            ):

                if close[j] <= open_[j]:
                    continue

                body = (
                    close[j]
                    - open_[j]
                )

                body_atr = (
                    body / atr[j]
                    if (
                        np.isfinite(
                            atr[j]
                        )
                        and atr[j] > 0
                    )
                    else np.nan
                )

                if (
                    np.isfinite(
                        body_atr
                    )
                    and (
                        body_atr
                        <= params.max_source_body_atr
                    )
                ):

                    source = (
                        j,
                        body_atr,
                    )

                    break

            if source is not None:

                j, body_atr = source
                label = out.index[i]

                zone_low = min(
                    open_[j],
                    close[j],
                )

                zone_high = max(
                    open_[j],
                    close[j],
                )

                out.at[
                    label,
                    "propulsion_bull",
                ] = True

                out.at[
                    label,
                    "propulsion_bull_low",
                ] = zone_low

                out.at[
                    label,
                    "propulsion_bull_high",
                ] = zone_high

                out.at[
                    label,
                    "propulsion_bull_mid",
                ] = (
                    zone_low
                    + zone_high
                ) / 2.0

                out.at[
                    label,
                    "propulsion_bull_source_body_atr",
                ] = body_atr

                out.at[
                    label,
                    "propulsion_bull_confirmation_displacement_atr",
                ] = disp_body_atr[i]

                out.at[
                    label,
                    "propulsion_bull_origin_idx",
                ] = j

                out.at[
                    label,
                    "propulsion_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "propulsion_bull_confirmation_lag",
                ] = i - j

        # =====================================================================
        # BEAR
        # =====================================================================

        if (
            bos_bear[i]
            and disp_bear[i]
            and np.isfinite(
                disp_body_atr[i]
            )
            and (
                disp_body_atr[i]
                >= params.min_confirmation_displacement_atr
            )
        ):

            source = None

            for j in range(
                i - 1,
                max(
                    -1,
                    i - params.max_lag - 1,
                ),
                -1,
            ):

                if close[j] >= open_[j]:
                    continue

                body = (
                    open_[j]
                    - close[j]
                )

                body_atr = (
                    body / atr[j]
                    if (
                        np.isfinite(
                            atr[j]
                        )
                        and atr[j] > 0
                    )
                    else np.nan
                )

                if (
                    np.isfinite(
                        body_atr
                    )
                    and (
                        body_atr
                        <= params.max_source_body_atr
                    )
                ):

                    source = (
                        j,
                        body_atr,
                    )

                    break

            if source is not None:

                j, body_atr = source
                label = out.index[i]

                zone_low = min(
                    open_[j],
                    close[j],
                )

                zone_high = max(
                    open_[j],
                    close[j],
                )

                out.at[
                    label,
                    "propulsion_bear",
                ] = True

                out.at[
                    label,
                    "propulsion_bear_low",
                ] = zone_low

                out.at[
                    label,
                    "propulsion_bear_high",
                ] = zone_high

                out.at[
                    label,
                    "propulsion_bear_mid",
                ] = (
                    zone_low
                    + zone_high
                ) / 2.0

                out.at[
                    label,
                    "propulsion_bear_source_body_atr",
                ] = body_atr

                out.at[
                    label,
                    "propulsion_bear_confirmation_displacement_atr",
                ] = disp_body_atr[i]

                out.at[
                    label,
                    "propulsion_bear_origin_idx",
                ] = j

                out.at[
                    label,
                    "propulsion_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "propulsion_bear_confirmation_lag",
                ] = i - j

    return out
