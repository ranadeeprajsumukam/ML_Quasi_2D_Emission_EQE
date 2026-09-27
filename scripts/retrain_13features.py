"""
Retrain using the exact 13-feature set from the original notebook
(Final_RFECV_eV_clean_H_add_opty_woRFECV.ipynb).

The 13 features were determined by running SelectFromModel once in the
notebook. We lock them here to guarantee full reproducibility and
consistency between the paper metrics and the deployed web app.

Run from repo root:
    python scripts/retrain_13features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from ml_quasit.experiment_config import load_config
from ml_quasit.rdkit_descriptor_features import build_feature_matrix
from ml_quasit.excel_dataset_loader import load_raw_dataset
from ml_quasit.regression_model_trainer import train_emission_models
from ml_quasit.model_serialization import ModelBundle, save_model_bundle

# ── Configuration ──────────────────────────────────────────────────────────────
CONFIG_PATH = REPO_ROOT / "configs" / "emission_photon_energy" / \
              "add_hydrogens_tuned_select_from_model.yaml"

# The exact 13 features confirmed from the original notebook
# (Final_RFECV_eV_clean_H_add_opty_woRFECV.ipynb, cell 9 output)
NOTEBOOK_13_FEATURES = [
    "CL_PRIMARY_ORGANIC_HALIDE",
    "BR_PRIMARY_ORGANIC_HALIDE",
    "SPACER_TO_PB_RATIO",
    "PbCl2",
    "PbBr2",
    "CsBr_TO_Pb",
    "CsI_TO_Pb",
    "FABr_TO_Pb",
    "FAI_TO_Pb",
    "SOLVENT_DMF",
    "Pri_VSA_EState4",
    "Sec_HallKierAlpha",
    "Sec_PEOE_VSA6",
]

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    config = load_config(str(CONFIG_PATH))

    print(f"\n=== Retraining with locked 13-feature set ===\n")

    # 1. Load raw data (for known spacers list)
    raw_dataset = load_raw_dataset(config)
    known_primary = sorted(
        raw_dataset["PRIMARY_ORGANIC_SPACER_IUPAC"].dropna().unique().tolist()
    )
    known_secondary = sorted(
        raw_dataset["SECONDARY_ORGANIC_SPACER_IUPAC"].dropna().unique().tolist()
    )

    # 2. Build full feature matrix (same as production pipeline)
    dataset_final = build_feature_matrix(config)
    print(f"Full feature matrix shape: {dataset_final.shape}")

    # 3. Capture solvent categories
    solvent_categories = [
        col.replace("SOLVENT_", "")
        for col in dataset_final.columns
        if col.startswith("SOLVENT_")
    ]
    all_feature_cols = [
        c for c in dataset_final.columns if c != config.target_column
    ]

    # 4. Same 80/20 stratified split as notebook
    X = dataset_final.drop(columns=[config.target_column] + list(config.drop_columns),
                           errors="ignore")
    y = dataset_final[config.target_column]

    stratify_col = (X["IS_MIXED_SPACERS_SPACER"] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col,
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # 5. Verify all 13 features exist in the feature matrix
    missing = [f for f in NOTEBOOK_13_FEATURES if f not in X_train.columns]
    if missing:
        raise ValueError(
            f"These features from the notebook are not in the current feature matrix: {missing}\n"
            "The dataset or preprocessing may have changed."
        )

    # 6. Select exactly the 13 notebook features
    X_train_final = X_train[NOTEBOOK_13_FEATURES].copy()
    X_test_final  = X_test[NOTEBOOK_13_FEATURES].copy()
    print(f"\nLocked to {len(NOTEBOOK_13_FEATURES)} notebook features:")
    for f in NOTEBOOK_13_FEATURES:
        print(f"  - {f}")

    # 7. Train all models with hyperparameter tuning (same as notebook)
    print("\nStarting hyperparameter tuning...")
    models, benchmark, scaler = train_emission_models(
        X_train_final, X_test_final,
        y_train, y_test,
        config,
    )

    print("\n--- Benchmark Results (13-feature model) ---")
    print(benchmark.to_string(index=False))

    # 8. Save as model bundle — app picks this up automatically
    best_model_name = benchmark.iloc[0]["Model"]
    bundle = ModelBundle(
        models=models,
        scaler=scaler,
        selected_features=NOTEBOOK_13_FEATURES,
        solvent_categories=solvent_categories,
        add_hydrogens=config.add_hydrogens,
        best_model_name=best_model_name,
        benchmark=benchmark,
        known_primary_spacers=known_primary,
        known_secondary_spacers=known_secondary,
        all_feature_columns=all_feature_cols,
        metadata={
            "note": "13-feature model locked from original notebook "
                    "(Final_RFECV_eV_clean_H_add_opty_woRFECV.ipynb). "
                    "SelectFromModel was re-run in the notebook and selected "
                    "13 features; those are hardcoded here for reproducibility."
        },
    )
    save_model_bundle(bundle, config)
    print(f"\nBest model: {best_model_name}")
    print(f"Features:   {len(NOTEBOOK_13_FEATURES)}")
    print("\nDone. App will use this bundle on next load.")


if __name__ == "__main__":
    main()
