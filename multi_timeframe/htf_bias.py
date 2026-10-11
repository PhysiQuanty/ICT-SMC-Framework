from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HTFBiasParams:
    """
    HTF Bias v0.1.

    Bias construit à partir d'événements structurels HTF.

    Priorité par défaut:

        MSS
          >
        BOS
          >
        Dealing Range direction

    Le bias persiste jusqu'à ce qu'une information structurelle
    de sens opposé apparaisse.

    Aucun prix futur.
    """

    use_mss: bool = True
    use_bos: bool = True
    use_dealing_range: bool = True

    neutral_until_signal: bool = True


def detect_htf_bias(
    df: pd.DataFrame,
    params: HTFBiasParams = HTFBiasParams(),
) -> pd.DataFrame:

    required = set()

    if params.use_mss:

        required.update(
            {
                "mss_bull",
                "mss_bear",
            }
        )

    if params.use_bos:

        required.update(
            {
                "bos_bull",
                "bos_bear",
            }
        )

    if params.use_dealing_range:

        required.add(
            "dealing_range_direction"
        )

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_htf_bias(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    out["htf_bias_bull"] = False
    out["htf_bias_bear"] = False
    out["htf_bias_neutral"] = True

    out["htf_bias_direction"] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out["htf_bias_source"] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    out["htf_bias_strength"] = np.nan

    out["htf_bias_last_change_idx"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    out["htf_bias_age_bars"] = pd.Series(
        pd.array(
            [pd.NA] * n,
            dtype="Int64",
        ),
        index=out.index,
    )

    bias = None
    source = None
    strength = np.nan

    last_change_idx = None

    for i in range(n):

        new_bias = None
        new_source = None
        new_strength = None

        # =====================================================================
        # MSS = strongest structural transition
        # =====================================================================

        if params.use_mss:

            if bool(
                out.iloc[i][
                    "mss_bull"
                ]
            ):

                new_bias = "bull"
                new_source = "mss"
                new_strength = 3.0

            elif bool(
                out.iloc[i][
                    "mss_bear"
                ]
            ):

                new_bias = "bear"
                new_source = "mss"
                new_strength = 3.0

        # =====================================================================
        # BOS
        # =====================================================================

        if (
            new_bias is None
            and params.use_bos
        ):

            if bool(
                out.iloc[i][
                    "bos_bull"
                ]
            ):

                new_bias = "bull"
                new_source = "bos"
                new_strength = 2.0

            elif bool(
                out.iloc[i][
                    "bos_bear"
                ]
            ):

                new_bias = "bear"
                new_source = "bos"
                new_strength = 2.0

        # =====================================================================
        # DEALING RANGE
        # =====================================================================

        if (
            new_bias is None
            and params.use_dealing_range
        ):

            direction = out.iloc[i][
                "dealing_range_direction"
            ]

            if direction == "bull":

                if (
                    bias is None
                    or not params.neutral_until_signal
                ):

                    new_bias = "bull"
                    new_source = "dealing_range"
                    new_strength = 1.0

            elif direction == "bear":

                if (
                    bias is None
                    or not params.neutral_until_signal
                ):

                    new_bias = "bear"
                    new_source = "dealing_range"
                    new_strength = 1.0

        # =====================================================================
        # CHANGE STATE
        # =====================================================================

        if new_bias is not None:

            changed = (
                bias != new_bias
            )

            bias = new_bias
            source = new_source
            strength = float(
                new_strength
            )

            if changed:

                last_change_idx = i

        # =====================================================================
        # OUTPUT STATE
        # =====================================================================

        label = out.index[i]

        if bias is None:

            out.at[
                label,
                "htf_bias_neutral",
            ] = True

            continue

        out.at[
            label,
            "htf_bias_direction",
        ] = bias

        out.at[
            label,
            "htf_bias_source",
        ] = source

        out.at[
            label,
            "htf_bias_strength",
        ] = strength

        out.at[
            label,
            "htf_bias_bull",
        ] = (
            bias == "bull"
        )

        out.at[
            label,
            "htf_bias_bear",
        ] = (
            bias == "bear"
        )

        out.at[
            label,
            "htf_bias_neutral",
        ] = False

        if last_change_idx is not None:

            out.at[
                label,
                "htf_bias_last_change_idx",
            ] = last_change_idx

            out.at[
                label,
                "htf_bias_age_bars",
            ] = (
                i
                - last_change_idx
            )

    return out
