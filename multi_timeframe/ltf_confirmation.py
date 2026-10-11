from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LTFConfirmationParams:
    """
    HTF context + LTF confirmation v0.1.

    Confirmation LTF supportée:

        MSS
        displacement
        FVG

    require_htf_pd_array:
        True:
            une zone HTF directionnellement cohérente
            doit être active.

        False:
            HTF bias seul suffit.

    max_confirmation_lag:
        permet:

            HTF context
                ↓
            LTF confirmation dans N bars

        0 = confirmation même bougie LTF.

    consume_context:
        un contexte produit au maximum une confirmation.
    """

    confirmation: str = "mss"

    require_htf_pd_array: bool = False

    max_confirmation_lag: int = 0

    consume_context: bool = False

    htf_prefix: str = "htf_"

    def __post_init__(self) -> None:

        if self.confirmation not in {
            "mss",
            "displacement",
            "fvg",
        }:

            raise ValueError(
                "confirmation doit être "
                "mss, displacement ou fvg"
            )

        if self.max_confirmation_lag < 0:

            raise ValueError(
                "max_confirmation_lag doit être >= 0"
            )


def detect_ltf_confirmation(
    df: pd.DataFrame,
    params: LTFConfirmationParams = LTFConfirmationParams(),
) -> pd.DataFrame:

    prefix = params.htf_prefix

    bias_bull_col = (
        f"{prefix}htf_bias_bull"
    )

    bias_bear_col = (
        f"{prefix}htf_bias_bear"
    )

    pd_bull_col = (
        f"{prefix}htf_pd_array_bull"
    )

    pd_bear_col = (
        f"{prefix}htf_pd_array_bear"
    )

    required = {
        bias_bull_col,
        bias_bear_col,
    }

    if params.require_htf_pd_array:

        required.update(
            {
                pd_bull_col,
                pd_bear_col,
            }
        )

    if params.confirmation == "mss":

        required.update(
            {
                "mss_bull",
                "mss_bear",
            }
        )

        bull_col = "mss_bull"
        bear_col = "mss_bear"

    elif params.confirmation == "displacement":

        required.update(
            {
                "displacement_bull",
                "displacement_bear",
            }
        )

        bull_col = (
            "displacement_bull"
        )

        bear_col = (
            "displacement_bear"
        )

    else:

        required.update(
            {
                "fvg_bull",
                "fvg_bear",
            }
        )

        bull_col = "fvg_bull"
        bear_col = "fvg_bear"

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "detect_ltf_confirmation(): "
            f"colonnes manquantes: {sorted(missing)}"
        )

    out = df.copy()
    n = len(out)

    bias_bull = out[
        bias_bull_col
    ].fillna(False).to_numpy(
        dtype=bool
    )

    bias_bear = out[
        bias_bear_col
    ].fillna(False).to_numpy(
        dtype=bool
    )

    if params.require_htf_pd_array:

        pd_bull = out[
            pd_bull_col
        ].fillna(False).to_numpy(
            dtype=bool
        )

        pd_bear = out[
            pd_bear_col
        ].fillna(False).to_numpy(
            dtype=bool
        )

    else:

        pd_bull = np.ones(
            n,
            dtype=bool,
        )

        pd_bear = np.ones(
            n,
            dtype=bool,
        )

    ltf_bull = out[
        bull_col
    ].to_numpy(dtype=bool)

    ltf_bear = out[
        bear_col
    ].to_numpy(dtype=bool)

    out[
        "mtf_confirmation_bull"
    ] = False

    out[
        "mtf_confirmation_bear"
    ] = False

    out[
        "mtf_confirmation_type"
    ] = pd.Series(
        [None] * n,
        dtype="object",
        index=out.index,
    )

    for side in (
        "mtf_confirmation_bull",
        "mtf_confirmation_bear",
    ):

        out[
            f"{side}_context_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{side}_confirmation_idx"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

        out[
            f"{side}_lag"
        ] = pd.Series(
            pd.array(
                [pd.NA] * n,
                dtype="Int64",
            ),
            index=out.index,
        )

    pending_bull = None
    pending_bear = None

    for i in range(n):

        bull_context = (
            bias_bull[i]
            and pd_bull[i]
        )

        bear_context = (
            bias_bear[i]
            and pd_bear[i]
        )

        # =====================================================================
        # SAME-BAR MODE
        # =====================================================================

        if params.max_confirmation_lag == 0:

            label = out.index[i]

            if (
                bull_context
                and ltf_bull[i]
            ):

                out.at[
                    label,
                    "mtf_confirmation_bull",
                ] = True

                out.at[
                    label,
                    "mtf_confirmation_type",
                ] = params.confirmation

                out.at[
                    label,
                    "mtf_confirmation_bull_context_idx",
                ] = i

                out.at[
                    label,
                    "mtf_confirmation_bull_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "mtf_confirmation_bull_lag",
                ] = 0

            if (
                bear_context
                and ltf_bear[i]
            ):

                out.at[
                    label,
                    "mtf_confirmation_bear",
                ] = True

                out.at[
                    label,
                    "mtf_confirmation_type",
                ] = params.confirmation

                out.at[
                    label,
                    "mtf_confirmation_bear_context_idx",
                ] = i

                out.at[
                    label,
                    "mtf_confirmation_bear_confirmation_idx",
                ] = i

                out.at[
                    label,
                    "mtf_confirmation_bear_lag",
                ] = 0

            continue

        # =====================================================================
        # SEQUENTIAL MODE
        # =====================================================================

        if bull_context:

            pending_bull = i

        if bear_context:

            pending_bear = i

        if (
            pending_bull is not None
            and (
                i - pending_bull
                > params.max_confirmation_lag
            )
        ):

            pending_bull = None

        if (
            pending_bear is not None
            and (
                i - pending_bear
                > params.max_confirmation_lag
            )
        ):

            pending_bear = None

        # ---------------------------------------------------------------------
        # BULL
        # ---------------------------------------------------------------------

        if (
            pending_bull is not None
            and ltf_bull[i]
        ):

            label = out.index[i]

            out.at[
                label,
                "mtf_confirmation_bull",
            ] = True

            out.at[
                label,
                "mtf_confirmation_type",
            ] = params.confirmation

            out.at[
                label,
                "mtf_confirmation_bull_context_idx",
            ] = pending_bull

            out.at[
                label,
                "mtf_confirmation_bull_confirmation_idx",
            ] = i

            out.at[
                label,
                "mtf_confirmation_bull_lag",
            ] = (
                i
                - pending_bull
            )

            if params.consume_context:
                pending_bull = None

        # ---------------------------------------------------------------------
        # BEAR
        # ---------------------------------------------------------------------

        if (
            pending_bear is not None
            and ltf_bear[i]
        ):

            label = out.index[i]

            out.at[
                label,
                "mtf_confirmation_bear",
            ] = True

            out.at[
                label,
                "mtf_confirmation_type",
            ] = params.confirmation

            out.at[
                label,
                "mtf_confirmation_bear_context_idx",
            ] = pending_bear

            out.at[
                label,
                "mtf_confirmation_bear_confirmation_idx",
            ] = i

            out.at[
                label,
                "mtf_confirmation_bear_lag",
            ] = (
                i
                - pending_bear
            )

            if params.consume_context:
                pending_bear = None

    return out
