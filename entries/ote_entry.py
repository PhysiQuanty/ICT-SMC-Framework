from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OTEEntryParams:
    """
    OTE Entry candidate v0.1.

    Consomme les zones OTE existantes.

    entry_mode:
        "touch_close"
            confirmation au premier contact;
            prix candidat = close de la bougie de contact.

        "mid"
            confirmation quand le midpoint OTE est atteint;
            prix candidat = midpoint OTE.

        "near"
            confirmation au premier contact de la zone;
            prix candidat = borne la plus proche du mouvement.

            bull -> ote_high
            bear -> ote_low

    first_touch_only:
        une seule entrée par dealing range.

    min_age_bars:
        évite l'entrée sur la bougie de création du range.
    """

    entry_mode: str = "touch_close"

    first_touch_only: bool = True
    min_age_bars: int = 1

    def __post_init__(self) -> None:

        if self.entry_mode not in {
            "touch_close",
            "mid",
            "near",
        }:
            raise ValueError(
                "entry_mode doit être touch_close, mid ou near"
            )

        if self.min_age_bars < 1:
            raise ValueError(
                "min_age_bars doit être >= 1"
            )


def detect_ote_entries(
    df: pd.DataFrame,
    params: OTEEntryParams = OTEEntryParams(),
) -> pd.DataFrame:

    required = {
        "high",
        "low",
        "close",

        "ote_bull_active",
        "ote_bear_active",

        "ote_bull_low",
        "ote_bull_high",
        "ote_bull_mid",

        "ote_bear_low",
        "ote_bear_high",
        "ote_bear_mid",

        "ote_bull_source_range_idx",
        "ote_bear_source_range_idx",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "detect_ote_entries(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    high = out["high"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)

    out["entry_ote_bull"] = False
    out["entry_ote_bear"] = False

    out["entry_ote_bull_price"] = np.nan
    out["entry_ote_bear_price"] = np.nan

    out["entry_ote_bull_zone_low"] = np.nan
    out["entry_ote_bull_zone_high"] = np.nan
    out["entry_ote_bull_zone_mid"] = np.nan

    out["entry_ote_bear_zone_low"] = np.nan
    out["entry_ote_bear_zone_high"] = np.nan
    out["entry_ote_bear_zone_mid"] = np.nan

    for side in (
        "entry_ote_bull",
        "entry_ote_bear",
    ):
        for suffix in (
            "source_idx",
            "confirmation_idx",
            "age_bars",
        ):
            out[f"{side}_{suffix}"] = pd.Series(
                pd.array(
                    [pd.NA] * n,
                    dtype="Int64",
                ),
                index=out.index,
            )

    consumed_bull = set()
    consumed_bear = set()

    for i in range(n):

        label = out.index[i]
        row = out.iloc[i]

        # =====================================================================
        # BULL
        # =====================================================================

        if bool(row["ote_bull_active"]):

            source = row[
                "ote_bull_source_range_idx"
            ]

            if pd.notna(source):

                source = int(source)
                age = i - source

                if age >= params.min_age_bars:

                    zone_low = float(
                        row["ote_bull_low"]
                    )

                    zone_high = float(
                        row["ote_bull_high"]
                    )

                    zone_mid = float(
                        row["ote_bull_mid"]
                    )

                    if params.entry_mode == "mid":

                        touched = (
                            low[i] <= zone_mid
                            and high[i] >= zone_mid
                        )

                        entry_price = zone_mid

                    else:

                        touched = (
                            low[i] <= zone_high
                            and high[i] >= zone_low
                        )

                        if params.entry_mode == "near":
                            entry_price = zone_high
                        else:
                            entry_price = close[i]

                    allowed = (
                        not params.first_touch_only
                        or source not in consumed_bull
                    )

                    if touched and allowed:

                        out.at[
                            label,
                            "entry_ote_bull",
                        ] = True

                        out.at[
                            label,
                            "entry_ote_bull_price",
                        ] = entry_price

                        out.at[
                            label,
                            "entry_ote_bull_zone_low",
                        ] = zone_low

                        out.at[
                            label,
                            "entry_ote_bull_zone_high",
                        ] = zone_high

                        out.at[
                            label,
                            "entry_ote_bull_zone_mid",
                        ] = zone_mid

                        out.at[
                            label,
                            "entry_ote_bull_source_idx",
                        ] = source

                        out.at[
                            label,
                            "entry_ote_bull_confirmation_idx",
                        ] = i

                        out.at[
                            label,
                            "entry_ote_bull_age_bars",
                        ] = age

                        consumed_bull.add(source)

        # =====================================================================
        # BEAR
        # =====================================================================

        if bool(row["ote_bear_active"]):

            source = row[
                "ote_bear_source_range_idx"
            ]

            if pd.notna(source):

                source = int(source)
                age = i - source

                if age >= params.min_age_bars:

                    zone_low = float(
                        row["ote_bear_low"]
                    )

                    zone_high = float(
                        row["ote_bear_high"]
                    )

                    zone_mid = float(
                        row["ote_bear_mid"]
                    )

                    if params.entry_mode == "mid":

                        touched = (
                            low[i] <= zone_mid
                            and high[i] >= zone_mid
                        )

                        entry_price = zone_mid

                    else:

                        touched = (
                            high[i] >= zone_low
                            and low[i] <= zone_high
                        )

                        if params.entry_mode == "near":
                            entry_price = zone_low
                        else:
                            entry_price = close[i]

                    allowed = (
                        not params.first_touch_only
                        or source not in consumed_bear
                    )

                    if touched and allowed:

                        out.at[
                            label,
                            "entry_ote_bear",
                        ] = True

                        out.at[
                            label,
                            "entry_ote_bear_price",
                        ] = entry_price

                        out.at[
                            label,
                            "entry_ote_bear_zone_low",
                        ] = zone_low

                        out.at[
                            label,
                            "entry_ote_bear_zone_high",
                        ] = zone_high

                        out.at[
                            label,
                            "entry_ote_bear_zone_mid",
                        ] = zone_mid

                        out.at[
                            label,
                            "entry_ote_bear_source_idx",
                        ] = source

                        out.at[
                            label,
                            "entry_ote_bear_confirmation_idx",
                        ] = i

                        out.at[
                            label,
                            "entry_ote_bear_age_bars",
                        ] = age

                        consumed_bear.add(source)

    return out
