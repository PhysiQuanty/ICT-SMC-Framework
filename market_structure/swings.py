from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwingParams:
    """
    Paramètres du détecteur de Swing High / Swing Low.

    left:
        Nombre de bougies à gauche du pivot.

    right:
        Nombre de bougies à droite nécessaires pour confirmer le pivot.

    strict:
        Si True:
            Swing High -> high[i] strictement supérieur aux highs voisins
            Swing Low  -> low[i] strictement inférieur aux lows voisins

        Si False:
            >= / <= sont autorisés.

    IMPORTANT CAUSALITÉ:
        Un swing géométriquement situé à l'index i n'est connu
        qu'à l'index i + right.
    """

    left: int = 3
    right: int = 3
    strict: bool = True

    def __post_init__(self) -> None:
        if self.left < 1:
            raise ValueError("SwingParams.left doit être >= 1")

        if self.right < 1:
            raise ValueError("SwingParams.right doit être >= 1")


def _validate_input(df: pd.DataFrame) -> None:
    required = {"high", "low"}

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes OHLCV manquantes pour detect_swings(): {sorted(missing)}"
        )

    if len(df) == 0:
        return

    if df["high"].isna().any():
        raise ValueError("La colonne 'high' contient des NaN")

    if df["low"].isna().any():
        raise ValueError("La colonne 'low' contient des NaN")


def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """
    Essaie de trouver automatiquement une colonne temporelle standard.
    Le détecteur n'impose aucun nom particulier au DataFrame.
    """

    candidates = (
        "timestamp",
        "open_time",
        "datetime",
        "date",
        "time",
    )

    for column in candidates:
        if column in df.columns:
            return column

    return None


def _is_swing_high(
    highs: np.ndarray,
    i: int,
    left: int,
    right: int,
    strict: bool,
) -> bool:
    pivot = highs[i]

    left_values = highs[i - left : i]
    right_values = highs[i + 1 : i + right + 1]

    if strict:
        return bool(
            np.all(pivot > left_values)
            and np.all(pivot > right_values)
        )

    return bool(
        np.all(pivot >= left_values)
        and np.all(pivot >= right_values)
    )


def _is_swing_low(
    lows: np.ndarray,
    i: int,
    left: int,
    right: int,
    strict: bool,
) -> bool:
    pivot = lows[i]

    left_values = lows[i - left : i]
    right_values = lows[i + 1 : i + right + 1]

    if strict:
        return bool(
            np.all(pivot < left_values)
            and np.all(pivot < right_values)
        )

    return bool(
        np.all(pivot <= left_values)
        and np.all(pivot <= right_values)
    )


def detect_swings(
    df: pd.DataFrame,
    params: SwingParams = SwingParams(),
) -> pd.DataFrame:
    """
    Détecte les Swing High / Swing Low de manière strictement causale.

    ------------------------------------------------------------------
    CONTRAT FEATURE
    ------------------------------------------------------------------

    Entrées:
        DataFrame contenant au minimum:
            high
            low

    Sorties ajoutées:
        swing_high
        swing_low

        swing_high_price
        swing_low_price

        swing_high_origin_idx
        swing_low_origin_idx

        swing_high_confirmation_idx
        swing_low_confirmation_idx

        swing_high_origin_time
        swing_low_origin_time

        swing_high_confirmation_time
        swing_low_confirmation_time

        swing_high_strength_left
        swing_high_strength_right

        swing_low_strength_left
        swing_low_strength_right

    ------------------------------------------------------------------
    CAUSALITÉ
    ------------------------------------------------------------------

    Exemple:
        left=3
        right=3

        pivot géométrique:
            index i

        information disponible:
            index i + 3

    La ligne i ne reçoit donc PAS immédiatement swing_high=True.

    Le signal de confirmation est écrit à la ligne i + right.

    Cela permet à une stratégie d'utiliser directement les colonnes
    swing_high / swing_low sans regarder accidentellement le futur.

    Les colonnes *_origin_idx permettent néanmoins de savoir où
    géométriquement le swing s'est formé.

    ------------------------------------------------------------------
    IMPORTANT
    ------------------------------------------------------------------

    Cette fonction:
        - ne contient aucune stratégie
        - ne connaît aucun timeframe
        - ne connaît ni HTF ni LTF
        - ne fait aucune entrée/sortie de trade
        - retourne seulement une primitive de market structure
    """

    _validate_input(df)

    out = df.copy()

    n = len(out)

    # ---------------------------------------------------------------
    # Colonnes booléennes:
    # elles représentent la CONFIRMATION du swing, pas son origine.
    # ---------------------------------------------------------------

    out["swing_high"] = False
    out["swing_low"] = False

    # ---------------------------------------------------------------
    # Prix du pivot confirmé
    # ---------------------------------------------------------------

    out["swing_high_price"] = np.nan
    out["swing_low_price"] = np.nan

    # ---------------------------------------------------------------
    # Indices
    # ---------------------------------------------------------------

    out["swing_high_origin_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["swing_low_origin_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["swing_high_confirmation_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["swing_low_confirmation_idx"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    # ---------------------------------------------------------------
    # Strength
    #
    # Pour cette première définition:
    # strength_left/right = nombre de bougies utilisées pour confirmer.
    #
    # Plus tard on pourra ajouter une véritable mesure continue de
    # prominence / excursion / persistence sans casser cette API.
    # ---------------------------------------------------------------

    out["swing_high_strength_left"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["swing_high_strength_right"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["swing_low_strength_left"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    out["swing_low_strength_right"] = pd.Series(
        pd.array([pd.NA] * n, dtype="Int64"),
        index=out.index,
    )

    timestamp_column = _find_timestamp_column(out)

    datetime_dtype = "datetime64[ns, UTC]"

    out["swing_high_origin_time"] = pd.Series(
        pd.array([pd.NaT] * n, dtype=datetime_dtype),
        index=out.index,
    )

    out["swing_low_origin_time"] = pd.Series(
        pd.array([pd.NaT] * n, dtype=datetime_dtype),
        index=out.index,
    )

    out["swing_high_confirmation_time"] = pd.Series(
        pd.array([pd.NaT] * n, dtype=datetime_dtype),
        index=out.index,
    )

    out["swing_low_confirmation_time"] = pd.Series(
        pd.array([pd.NaT] * n, dtype=datetime_dtype),
        index=out.index,
    )

    if n == 0:
        return out

    left = params.left
    right = params.right

    if n < left + right + 1:
        return out

    highs = out["high"].to_numpy(dtype=np.float64, copy=False)
    lows = out["low"].to_numpy(dtype=np.float64, copy=False)

    # On utilise des positions entières internes indépendantes du
    # DataFrame.index utilisateur.
    #
    # Cela garantit que origin_idx / confirmation_idx représentent
    # toujours des positions chronologiques 0..N-1.
    for origin_idx in range(left, n - right):

        confirmation_idx = origin_idx + right

        is_high = _is_swing_high(
            highs=highs,
            i=origin_idx,
            left=left,
            right=right,
            strict=params.strict,
        )

        is_low = _is_swing_low(
            lows=lows,
            i=origin_idx,
            left=left,
            right=right,
            strict=params.strict,
        )

        confirmation_label = out.index[confirmation_idx]

        if is_high:
            out.at[confirmation_label, "swing_high"] = True

            out.at[
                confirmation_label,
                "swing_high_price",
            ] = highs[origin_idx]

            out.at[
                confirmation_label,
                "swing_high_origin_idx",
            ] = origin_idx

            out.at[
                confirmation_label,
                "swing_high_confirmation_idx",
            ] = confirmation_idx

            out.at[
                confirmation_label,
                "swing_high_strength_left",
            ] = left

            out.at[
                confirmation_label,
                "swing_high_strength_right",
            ] = right

            if timestamp_column is not None:
                out.at[
                    confirmation_label,
                    "swing_high_origin_time",
                ] = pd.to_datetime(
                    out.iloc[origin_idx][timestamp_column],
                    unit="ms",
                    utc=True,
                )

                out.at[
                    confirmation_label,
                    "swing_high_confirmation_time",
                ] = pd.to_datetime(
                    out.iloc[confirmation_idx][timestamp_column],
                    unit="ms",
                    utc=True,
                )

        if is_low:
            out.at[confirmation_label, "swing_low"] = True

            out.at[
                confirmation_label,
                "swing_low_price",
            ] = lows[origin_idx]

            out.at[
                confirmation_label,
                "swing_low_origin_idx",
            ] = origin_idx

            out.at[
                confirmation_label,
                "swing_low_confirmation_idx",
            ] = confirmation_idx

            out.at[
                confirmation_label,
                "swing_low_strength_left",
            ] = left

            out.at[
                confirmation_label,
                "swing_low_strength_right",
            ] = right

            if timestamp_column is not None:
                out.at[
                    confirmation_label,
                    "swing_low_origin_time",
                ] = pd.to_datetime(
                    out.iloc[origin_idx][timestamp_column],
                    unit="ms",
                    utc=True,
                )

                out.at[
                    confirmation_label,
                    "swing_low_confirmation_time",
                ] = pd.to_datetime(
                    out.iloc[confirmation_idx][timestamp_column],
                    unit="ms",
                    utc=True,
                )

    return out
