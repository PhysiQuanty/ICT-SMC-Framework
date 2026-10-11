from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LiquidityVoidParams:
    """
    Proxy OHLC de Liquidity Void.

    On ne prétend pas observer directement la profondeur du carnet.

    Proxy:
        séquence de bougies directionnelles de forte amplitude
        avec peu de retracement / overlap.

    min_bars:
        nombre minimum de bougies dans la séquence.

    max_bars:
        longueur maximale analysée.

    min_total_range_atr:
        déplacement total minimum en ATR.

    max_overlap_ratio:
        overlap maximal moyen entre bougies successives.

    same_direction_required:
        impose que les bougies aient toutes la même direction.
    """

    min_bars: int = 2
    max_bars: int = 5

    min_total_range_atr: float = 2.0

    max_overlap_ratio: float = 0.35

    atr_column: str = "atr"

    same_direction_required: bool = True

    def __post_init__(self) -> None:

        if self.min_bars < 2:
            raise ValueError(
                "min_bars doit être >= 2"
            )

        if self.max_bars < self.min_bars:
            raise ValueError(
                "max_bars doit être >= min_bars"
            )

        if self.min_total_range_atr < 0:
            raise ValueError(
                "min_total_range_atr doit être >= 0"
            )

        if not 0 <= self.max_overlap_ratio <= 1:
            raise ValueError(
                "max_overlap_ratio doit être entre 0 et 1"
            )


def _overlap_ratio(
    high_a: float,
    low_a: float,
    high_b: float,
    low_b: float,
) -> float:

    overlap = max(
        0.0,
        min(high_a, high_b)
        - max(low_a, low_b),
    )

    range_a = high_a - low_a
    range_b = high_b - low_b

    denom = min(
        range_a,
        range_b,
    )

    if denom <= 0:
        return 0.0

    return overlap / denom


def detect_liquidity_void(
    df: pd.DataFrame,
    params: LiquidityVoidParams = LiquidityVoidParams(),
) -> pd.DataFrame:

    required = {
        "open",
        "high",
        "low",
        "close",
        params.atr_column,
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"detect_liquidity_void(): colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["liquidity_void_bull"] = False
    out["liquidity_void_bear"] = False

    out["liquidity_void_bars"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["liquidity_void_range"] = np.nan
    out["liquidity_void_range_atr"] = np.nan
    out["liquidity_void_avg_overlap"] = np.nan

    out["liquidity_void_origin_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["liquidity_void_confirmation_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    open_ = out["open"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)
    atr = out[params.atr_column].to_numpy(dtype=np.float64)

    for i in range(n):

        best = None

        for length in range(
            params.min_bars,
            params.max_bars + 1,
        ):

            start = (
                i - length + 1
            )

            if start < 0:
                continue

            opens = open_[start:i + 1]
            closes = close[start:i + 1]

            bull = np.all(
                closes > opens
            )

            bear = np.all(
                closes < opens
            )

            if params.same_direction_required:
                if not bull and not bear:
                    continue

            displacement = (
                close[i] - open_[start]
            )

            direction = (
                "bull"
                if displacement > 0
                else "bear"
            )

            total_range = abs(
                displacement
            )

            range_atr = (
                total_range / atr[i]
                if (
                    np.isfinite(atr[i])
                    and atr[i] > 0
                )
                else np.nan
            )

            if (
                not np.isfinite(range_atr)
                or range_atr < params.min_total_range_atr
            ):
                continue

            overlaps = []

            for j in range(
                start + 1,
                i + 1,
            ):

                overlaps.append(
                    _overlap_ratio(
                        high[j - 1],
                        low[j - 1],
                        high[j],
                        low[j],
                    )
                )

            avg_overlap = (
                float(np.mean(overlaps))
                if overlaps
                else 0.0
            )

            if (
                avg_overlap
                > params.max_overlap_ratio
            ):
                continue

            candidate = {
                "direction":
                    direction,

                "length":
                    length,

                "start":
                    start,

                "range":
                    total_range,

                "range_atr":
                    range_atr,

                "avg_overlap":
                    avg_overlap,
            }

            if (
                best is None
                or candidate["range_atr"]
                > best["range_atr"]
            ):
                best = candidate

        if best is None:
            continue

        label = out.index[i]

        if best["direction"] == "bull":
            out.at[
                label,
                "liquidity_void_bull",
            ] = True

        else:
            out.at[
                label,
                "liquidity_void_bear",
            ] = True

        out.at[
            label,
            "liquidity_void_bars",
        ] = best["length"]

        out.at[
            label,
            "liquidity_void_range",
        ] = best["range"]

        out.at[
            label,
            "liquidity_void_range_atr",
        ] = best["range_atr"]

        out.at[
            label,
            "liquidity_void_avg_overlap",
        ] = best["avg_overlap"]

        out.at[
            label,
            "liquidity_void_origin_idx",
        ] = best["start"]

        out.at[
            label,
            "liquidity_void_confirmation_idx",
        ] = i

    return out
