"""End-to-end ML pipeline orchestration."""

from __future__ import annotations

import warnings

import pandas as pd

from ml_quasit.experiment_config import ExperimentConfig, load_config
from ml_quasit.train_test_feature_selection import (
    SplitData,
    select_features,
    train_test_split_data,
)
from ml_quasit.rdkit_descriptor_features import build_feature_matrix
from ml_quasit.regression_model_trainer import train_emission_models, train_eqe_models
from ml_quasit.experiment_results_plots import (
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_rfecv_curve,
    plot_shap_emission,
    plot_shap_eqe,
)
from ml_quasit.model_serialization import ModelBundle, save_model_bundle

warnings.filterwarnings("ignore")


def _feature_importances(selector, support) -> tuple:
    """Extract feature importances from various selector types.

    For RFECV: use ``selector.estimator_.feature_importances_``
    For SelectFromModel (returns the fitted RF): use ``feature_importances_[support]``
    """
    if hasattr(selector, "feature_importances_"):
        importances = selector.feature_importances_
        # If importances cover all features (pre-selection), subset to selected
        if len(importances) > support.sum() and len(importances) == len(support):
            return importances[support]
        return importances
    elif hasattr(selector, "estimator_") and hasattr(
        selector.estimator_, "feature_importances_"
    ):
        return selector.estimator_.feature_importances_
    raise AttributeError("Cannot extract feature importances from selector.")



def _extract_solvent_categories(dataset_final: pd.DataFrame) -> list[str]:
    """Extract solvent category names from one-hot encoded columns."""
    return [
        col.replace("SOLVENT_", "")
        for col in dataset_final.columns
        if col.startswith("SOLVENT_")
    ]


def _extract_known_spacers(dataset_final: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Extract known spacer IUPAC names from the raw dataset.

    These are preserved before the IUPAC columns are dropped so that
    the inference pipeline can warn when an unseen spacer is used.
    """
    from ml_quasit.excel_dataset_loader import load_raw_dataset
    # We read from the columns before they were dropped
    primary = []
    secondary = []
    if "PRIMARY_ORGANIC_SPACER_IUPAC" in dataset_final.columns:
        primary = sorted(
            dataset_final["PRIMARY_ORGANIC_SPACER_IUPAC"].dropna().unique().tolist()
        )
    if "SECONDARY_ORGANIC_SPACER_IUPAC" in dataset_final.columns:
        secondary = sorted(
            dataset_final["SECONDARY_ORGANIC_SPACER_IUPAC"].dropna().unique().tolist()
        )
    return primary, secondary


def run_experiment(
    config: ExperimentConfig,
    *,
    save_bundle: bool = True,
) -> pd.DataFrame:
    """Execute the full workflow for one experiment configuration.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration.
    save_bundle : bool
        If True (default), save the trained model bundle to ``models/``
        for later inference.
    """
    print(f"\n=== Running pipeline: {config.name} ===\n")

    # ── Featurization ───────────────────────────────────────────
    # Load raw dataset first to capture spacer IUPAC names
    from ml_quasit.excel_dataset_loader import load_raw_dataset
    raw_dataset = load_raw_dataset(config)
    known_primary = sorted(
        raw_dataset["PRIMARY_ORGANIC_SPACER_IUPAC"].dropna().unique().tolist()
    )
    known_secondary = sorted(
        raw_dataset["SECONDARY_ORGANIC_SPACER_IUPAC"].dropna().unique().tolist()
    )

    dataset_final = build_feature_matrix(config)
    print(f"Feature matrix shape: {dataset_final.shape}")

    # Capture solvent categories from one-hot columns
    solvent_categories = _extract_solvent_categories(dataset_final)

    # Capture all feature columns before selection
    all_feature_cols = [
        c for c in dataset_final.columns if c != config.target_column
    ]

    split = train_test_split_data(dataset_final, config)
    X_train_final, X_test_final, selector, support = select_features(split, config)

    if config.feature_selection == "rfecv" and hasattr(selector, "cv_results_"):
        plot_rfecv_curve(selector, config)

    plot_correlation_heatmap(X_train_final, config)
    plot_feature_importance(
        _feature_importances(selector, support),
        X_train_final.columns,
        config,
    )

    scaler = None
    if config.task == "eqe":
        models, benchmark = train_eqe_models(
            X_train_final,
            X_test_final,
            split.y_train,
            split.y_test,
            config,
            split.X_train_char,
            split.X_test_char,
        )
        if not config.eqe_catboost_architecture_only:
            train_map = {}
            string_cols = X_train_final.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()
            cat_features = [c for c in X_train_final.columns if c in string_cols]
            num_features = [c for c in X_train_final.columns if c not in string_cols]
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="median")
            X_num = pd.DataFrame(
                imputer.fit_transform(X_train_final[num_features]),
                columns=num_features,
            )
            X_cat = X_train_final.copy()
            for col in cat_features:
                X_cat[col] = X_cat[col].fillna("Unknown").astype(str)
            for name in models:
                train_map[name] = X_cat if "CatBoost" in name else X_num
            plot_shap_eqe(models, train_map, config)
        else:
            X_full = pd.concat(
                [X_train_final.fillna(0), split.X_train_char], axis=1
            )
            for col in split.X_train_char.columns:
                X_full[col] = X_full[col].astype(str)
            plot_shap_eqe(
                models, {"CatBoost Architecture": X_full}, config
            )
    else:
        models, benchmark, scaler = train_emission_models(
            X_train_final,
            X_test_final,
            split.y_train,
            split.y_test,
            config,
        )
        if scaler is not None:
            plot_shap_emission(
                models, X_train_final, X_test_final, scaler, config
            )

    out_csv = config.results_dir / "benchmark_results.csv"
    benchmark.to_csv(out_csv, index=False)
    print("\n--- Benchmark Results ---")
    print(benchmark.to_string(index=False))
    print(f"\nResults saved to: {config.results_dir}")

    # ── Save model bundle ───────────────────────────────────────
    if save_bundle:
        # Identify best model (first row in benchmark, sorted by RMSE)
        best_model_name = benchmark.iloc[0]["Model"]

        bundle = ModelBundle(
            models=models,
            scaler=scaler,
            selected_features=X_train_final.columns.tolist(),
            solvent_categories=solvent_categories,
            add_hydrogens=config.add_hydrogens,
            best_model_name=best_model_name,
            benchmark=benchmark,
            known_primary_spacers=known_primary,
            known_secondary_spacers=known_secondary,
            all_feature_columns=all_feature_cols,
        )
        save_model_bundle(bundle, config)

    return benchmark


def main(config_path: str) -> None:
    """CLI entry: python -m ml_quasit.run_experiment configs/foo.yaml"""
    config = load_config(config_path)
    run_experiment(config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ml_quasit.run_experiment <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
