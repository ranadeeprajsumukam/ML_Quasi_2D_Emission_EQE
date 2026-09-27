"""End-to-end inference pipeline for emission photon energy prediction.

Provides :class:`EmissionPredictor` which loads a saved model bundle and
predicts emission photon energy (eV) from user-friendly chemical inputs:
  - IUPAC name or SMILES of the primary organic spacer
  - Optionally a secondary spacer (for mixed-spacer systems)
  - Solvent choice
  - Precursor composition ratios

Designed so that researchers can use the model without understanding the
internal feature-engineering pipeline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from ml_quasit.iupac_to_smiles import MASTER_SMILES_MAP, cir_convert
from ml_quasit.rdkit_descriptor_features import DESC_NAMES, get_2d_descriptors
from ml_quasit.model_serialization import ModelBundle, load_model_bundle

warnings.filterwarnings("ignore")

# ── Canonical lists for user-facing validation ──────────────────────────

KNOWN_SOLVENTS = ["DMF", "DMSO", "DMF:DMSO", "NMP"]

HALIDE_SALTS = [
    "CL_PRIMARY_ORGANIC_HALIDE",
    "BR_PRIMARY_ORGANIC_HALIDE",
    "I_PRIMARY_ORGANIC_HALIDE",
    "CL_SECONDARY_ORGANIC_HALIDE",
    "BR_SECONDARY_ORGANIC_HALIDE",
    "I_SECONDARY_ORGANIC_HALIDE",
]

PRECURSOR_RATIO_COLUMNS = [
    "PbCl2",
    "PbBr2",
    "PbI2",
    "CsCl_TO_Pb",
    "CsBr_TO_Pb",
    "CsI_TO_Pb",
    "RbBr_TO_Pb",
    "RbI_TO_Pb",
    "FABr_TO_Pb",
    "FAI_TO_Pb",
    "MABr_TO_Pb",
    "MAI_TO_Pb",
]


@dataclass
class PredictionInput:
    """User-facing input for a single prediction.

    Only ``primary_spacer`` and ``solvent`` are required.
    Everything else has sensible defaults for the most common
    single-spacer PEA-based perovskite system.
    """

    # ── Required ────────────────────────────────────────────────
    primary_spacer: str
    """IUPAC name or SMILES of the primary organic spacer cation."""

    solvent: str
    """Solvent used for perovskite film deposition."""

    # ── Spacer details ──────────────────────────────────────────
    primary_halide: str = "Br"
    """Counter-ion halide for the primary spacer: 'Cl', 'Br', or 'I'."""

    primary_spacer_fraction: float = 1.0
    """Molar fraction of primary spacer (0–1). Default 1.0 (single-spacer)."""

    secondary_spacer: str | None = None
    """IUPAC name or SMILES of the secondary spacer (if mixed)."""

    secondary_halide: str = "Br"
    """Counter-ion halide for the secondary spacer: 'Cl', 'Br', or 'I'."""

    secondary_spacer_fraction: float = 0.0
    """Molar fraction of secondary spacer (0–1)."""

    spacer_to_pb_ratio: float = 2.0
    """Total spacer-to-Pb molar ratio (typically 1.6–3.0 for <n>=3–5)."""

    # ── Precursor composition (molar ratios to Pb) ──────────────
    PbCl2: float = 0.0
    PbBr2: float = 1.0
    PbI2: float = 0.0
    CsCl_TO_Pb: float = 0.0
    CsBr_TO_Pb: float = 1.0
    CsI_TO_Pb: float = 0.0
    RbBr_TO_Pb: float = 0.0
    RbI_TO_Pb: float = 0.0
    FABr_TO_Pb: float = 0.0
    FAI_TO_Pb: float = 0.0
    MABr_TO_Pb: float = 0.0
    MAI_TO_Pb: float = 0.0


@dataclass
class PredictionResult:
    """Holds the prediction output and supporting information."""

    predicted_ev: float
    """Predicted emission photon energy in eV."""

    predicted_nm: float
    """Predicted emission wavelength in nm (converted from eV)."""

    model_name: str
    """Name of the model used for prediction."""

    confidence_note: str
    """Qualitative confidence note based on input similarity to training data."""

    all_model_predictions: dict[str, float]
    """Predictions from all available models in the bundle."""

    feature_values: dict[str, float] | None = None
    """The computed feature values fed into the model (for debugging)."""


def _resolve_smiles(name_or_smiles: str) -> str | None:
    """Convert an IUPAC name or SMILES to canonical SMILES.

    Tries these strategies in order:
    1. Direct lookup in the hardcoded MASTER_SMILES_MAP
    2. Try parsing as SMILES via RDKit
    3. IUPAC → SMILES via CACTUS NCI
    """
    # 1. Check master map (IUPAC name match)
    key = name_or_smiles.strip()
    if key.lower() in {k.lower() for k in MASTER_SMILES_MAP}:
        for k, v in MASTER_SMILES_MAP.items():
            if k.lower() == key.lower():
                return v

    # 2. Try parsing as SMILES directly
    mol = Chem.MolFromSmiles(key)
    if mol is not None:
        return Chem.MolToSmiles(mol)  # canonicalize

    # 3. Fall back to CACTUS
    result = cir_convert(key)
    if result and "Could not find" not in result:
        return result

    return None


def _halide_flags(halide: str) -> tuple[int, int, int]:
    """Return (Cl, Br, I) binary flags from a halide string."""
    h = halide.strip().upper()
    if h == "CL":
        return (1, 0, 0)
    elif h == "BR":
        return (0, 1, 0)
    elif h == "I":
        return (0, 0, 1)
    else:
        raise ValueError(
            f"Invalid halide '{halide}'. Must be one of: 'Cl', 'Br', 'I'"
        )


def _validate_input(inp: PredictionInput, bundle: ModelBundle) -> list[str]:
    """Validate user input and return a list of warnings (empty = all OK)."""
    warnings_list: list[str] = []

    # Validate solvent
    if inp.solvent not in KNOWN_SOLVENTS:
        warnings_list.append(
            f"Unknown solvent '{inp.solvent}'. "
            f"Training data used: {KNOWN_SOLVENTS}. "
            f"Prediction may be unreliable."
        )

    # Validate fractions
    if not (0.0 <= inp.primary_spacer_fraction <= 1.0):
        raise ValueError("primary_spacer_fraction must be between 0 and 1.")
    if not (0.0 <= inp.secondary_spacer_fraction <= 1.0):
        raise ValueError("secondary_spacer_fraction must be between 0 and 1.")

    # Check if spacer was seen in training
    if inp.primary_spacer not in bundle.known_primary_spacers:
        smiles = _resolve_smiles(inp.primary_spacer)
        if smiles is None:
            raise ValueError(
                f"Cannot resolve '{inp.primary_spacer}' to a valid SMILES. "
                f"Please provide a valid IUPAC name or SMILES string."
            )
        warnings_list.append(
            f"Primary spacer '{inp.primary_spacer}' was not in the training data. "
            f"Prediction is an extrapolation and may be less reliable."
        )

    # Mixed spacer consistency
    if inp.secondary_spacer and inp.secondary_spacer_fraction == 0.0:
        warnings_list.append(
            "Secondary spacer provided but fraction is 0.0. "
            "Setting fraction to a non-zero value is recommended."
        )

    if inp.secondary_spacer is None and inp.secondary_spacer_fraction > 0.0:
        raise ValueError(
            "secondary_spacer_fraction > 0 but no secondary_spacer provided."
        )

    # Validate spacer-to-Pb ratio range
    if inp.spacer_to_pb_ratio < 0.5 or inp.spacer_to_pb_ratio > 10.0:
        warnings_list.append(
            f"spacer_to_pb_ratio={inp.spacer_to_pb_ratio} is outside the "
            f"typical range (0.5–10.0). Prediction may be unreliable."
        )

    return warnings_list


def _build_feature_row(
    inp: PredictionInput, bundle: ModelBundle
) -> pd.DataFrame:
    """Convert user input into the feature vector expected by the model.

    Replicates the exact preprocessing from the training pipeline:
    1. Resolve SMILES
    2. Compute RDKit 2D descriptors (with optional AddHs)
    3. One-hot encode solvent
    4. Assemble all columns in the same order as training
    """
    # ── Resolve SMILES ──────────────────────────────────────────
    primary_smiles = _resolve_smiles(inp.primary_spacer)
    if primary_smiles is None:
        raise ValueError(
            f"Cannot resolve primary spacer '{inp.primary_spacer}' to SMILES."
        )

    secondary_smiles = None
    if inp.secondary_spacer:
        secondary_smiles = _resolve_smiles(inp.secondary_spacer)
        if secondary_smiles is None:
            raise ValueError(
                f"Cannot resolve secondary spacer '{inp.secondary_spacer}' to SMILES."
            )

    # ── RDKit descriptors ───────────────────────────────────────
    pri_desc = get_2d_descriptors(primary_smiles, add_hydrogens=bundle.add_hydrogens)
    sec_desc = get_2d_descriptors(secondary_smiles, add_hydrogens=bundle.add_hydrogens)

    # ── Halide flags ────────────────────────────────────────────
    pri_cl, pri_br, pri_i = _halide_flags(inp.primary_halide)
    sec_cl, sec_br, sec_i = _halide_flags(inp.secondary_halide)

    # ── Build raw row dict ──────────────────────────────────────
    row: dict[str, Any] = {
        "IS_MIXED_SPACERS_SPACER": 1 if inp.secondary_spacer else 0,
        "CL_PRIMARY_ORGANIC_HALIDE": pri_cl,
        "BR_PRIMARY_ORGANIC_HALIDE": pri_br,
        "I_PRIMARY_ORGANIC_HALIDE": pri_i,
        "PRIMARY_SPACER_FRACTION": inp.primary_spacer_fraction,
        "CL_SECONDARY_ORGANIC_HALIDE": sec_cl,
        "BR_SECONDARY_ORGANIC_HALIDE": sec_br,
        "I_SECONDARY_ORGANIC_HALIDE": sec_i,
        "SECONDARY_SPACER_FRACTION": inp.secondary_spacer_fraction,
        "SPACER_TO_PB_RATIO": inp.spacer_to_pb_ratio,
    }

    # Precursor ratios
    for col in PRECURSOR_RATIO_COLUMNS:
        row[col] = getattr(inp, col, 0.0)

    # Solvent one-hot (match training categories)
    for solvent in bundle.solvent_categories:
        col_name = f"SOLVENT_{solvent}"
        row[col_name] = 1.0 if inp.solvent == solvent else 0.0

    # Primary descriptors with "Pri_" prefix
    for i, name in enumerate(DESC_NAMES):
        val = pri_desc[i] if pri_desc[i] is not None else 0.0
        row[f"Pri_{name}"] = val

    # Secondary descriptors with "Sec_" prefix
    for i, name in enumerate(DESC_NAMES):
        val = sec_desc[i] if sec_desc[i] is not None else 0.0
        row[f"Sec_{name}"] = val

    # Build DataFrame with exactly the selected features
    df = pd.DataFrame([row])

    # Ensure all expected columns exist (fill missing with 0)
    for col in bundle.selected_features:
        if col not in df.columns:
            df[col] = 0.0

    # Select only the features the model expects, in the right order
    df = df[bundle.selected_features]

    # Fill any remaining NaN with 0
    df = df.fillna(0)

    return df


class EmissionPredictor:
    """High-level predictor for emission photon energy.

    Usage::

        predictor = EmissionPredictor("models/emission_photon_energy_add_hydrogens_tuned_select_from_model")
        result = predictor.predict(
            PredictionInput(
                primary_spacer="2-phenylethan-1-amine",
                solvent="DMSO",
                PbBr2=1.0,
                CsBr_TO_Pb=1.0,
            )
        )
        print(f"Predicted: {result.predicted_ev:.3f} eV ({result.predicted_nm:.0f} nm)")
    """

    def __init__(self, model_path: str | Path):
        self.bundle = load_model_bundle(model_path)
        self._best_model = self.bundle.models[self.bundle.best_model_name]

    @property
    def model_names(self) -> list[str]:
        return list(self.bundle.models.keys())

    @property
    def best_model_name(self) -> str:
        return self.bundle.best_model_name

    @property
    def known_primary_spacers(self) -> list[str]:
        return self.bundle.known_primary_spacers

    @property
    def known_secondary_spacers(self) -> list[str]:
        return self.bundle.known_secondary_spacers

    def predict(
        self,
        inp: PredictionInput,
        *,
        model_name: str | None = None,
        return_features: bool = False,
    ) -> PredictionResult:
        """Predict emission photon energy for a given input.

        Parameters
        ----------
        inp : PredictionInput
            Chemical and processing inputs.
        model_name : str, optional
            Use a specific model from the bundle. Defaults to best.
        return_features : bool
            If True, include computed feature values in the result.
        """
        # Validate
        input_warnings = _validate_input(inp, self.bundle)

        # Build feature row
        feature_df = _build_feature_row(inp, self.bundle)

        # Scale if scaler exists
        if self.bundle.scaler is not None:
            feature_scaled = self.bundle.scaler.transform(feature_df)
        else:
            feature_scaled = feature_df.values

        # Predict with all models
        all_preds: dict[str, float] = {}
        for name, model in self.bundle.models.items():
            pred = model.predict(feature_scaled)
            all_preds[name] = float(pred[0])

        # Pick the requested or best model
        use_model = model_name or self.bundle.best_model_name
        if use_model not in all_preds:
            raise ValueError(
                f"Model '{use_model}' not found. "
                f"Available: {list(all_preds.keys())}"
            )
        predicted_ev = all_preds[use_model]

        # Convert eV → nm: E(eV) = 1240 / λ(nm)
        predicted_nm = 1240.0 / predicted_ev if predicted_ev > 0 else float("nan")

        # Confidence note
        if input_warnings:
            confidence = "[WARNING] Extrapolation: " + "; ".join(input_warnings)
        else:
            confidence = "[OK] Input is within the training domain."

        result = PredictionResult(
            predicted_ev=predicted_ev,
            predicted_nm=predicted_nm,
            model_name=use_model,
            confidence_note=confidence,
            all_model_predictions=all_preds,
        )

        if return_features:
            result.feature_values = feature_df.iloc[0].to_dict()

        return result

    def predict_batch(
        self,
        inputs: list[PredictionInput],
        *,
        model_name: str | None = None,
    ) -> pd.DataFrame:
        """Predict emission for a batch of inputs.

        Returns a DataFrame with one row per input, including predicted
        eV, nm, and all model predictions.
        """
        results = []
        for i, inp in enumerate(inputs):
            try:
                res = self.predict(inp, model_name=model_name)
                results.append(
                    {
                        "index": i,
                        "primary_spacer": inp.primary_spacer,
                        "solvent": inp.solvent,
                        "predicted_ev": res.predicted_ev,
                        "predicted_nm": res.predicted_nm,
                        "model": res.model_name,
                        "confidence": res.confidence_note,
                        **{
                            f"pred_{k}": v
                            for k, v in res.all_model_predictions.items()
                        },
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "index": i,
                        "primary_spacer": inp.primary_spacer,
                        "solvent": inp.solvent,
                        "predicted_ev": float("nan"),
                        "predicted_nm": float("nan"),
                        "model": "ERROR",
                        "confidence": str(e),
                    }
                )
        return pd.DataFrame(results)
