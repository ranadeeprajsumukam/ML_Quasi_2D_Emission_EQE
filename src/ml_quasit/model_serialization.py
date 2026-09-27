"""Save and load trained model bundles for inference.

A model bundle captures everything needed to reproduce predictions
without retraining: fitted models, scaler, feature list, solvent
encoder categories, and metadata about training configuration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ml_quasit.experiment_config import ExperimentConfig, REPO_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = REPO_ROOT / "models"


@dataclass
class ModelBundle:
    """Everything needed to make predictions with a trained pipeline."""

    # Fitted sklearn/catboost/xgboost models keyed by name
    models: dict[str, Any]

    # StandardScaler fitted on training features (emission tasks)
    scaler: Any  # StandardScaler | None

    # Ordered list of feature column names the models expect
    selected_features: list[str]

    # Solvent categories seen during training (for one-hot encoding)
    solvent_categories: list[str]

    # Whether hydrogens were added during RDKit featurization
    add_hydrogens: bool

    # Name of the best model (lowest RMSE)
    best_model_name: str

    # Benchmark results from training evaluation
    benchmark: pd.DataFrame

    # Known primary spacer IUPAC names from training data
    known_primary_spacers: list[str]

    # Known secondary spacer IUPAC names from training data
    known_secondary_spacers: list[str]

    # All columns present before feature selection (for validation)
    all_feature_columns: list[str] = field(default_factory=list)

    # Training metadata
    metadata: dict[str, Any] = field(default_factory=dict)


def save_model_bundle(
    bundle: ModelBundle,
    config: ExperimentConfig,
    *,
    tag: str | None = None,
) -> Path:
    """Persist a model bundle to disk.

    Saves to ``models/<config.name>/`` with an optional tag suffix.
    Returns the path to the saved ``.joblib`` file.
    """
    out_dir = MODELS_DIR / config.name
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{tag}" if tag else ""
    bundle_path = out_dir / f"bundle_{timestamp}{suffix}.joblib"

    # Enrich metadata
    bundle.metadata.update(
        {
            "config_name": config.name,
            "task": config.task,
            "target_column": config.target_column,
            "add_hydrogens": config.add_hydrogens,
            "feature_selection": config.feature_selection,
            "hyperparameter_tuning": config.hyperparameter_tuning,
            "correlation_threshold": config.correlation_threshold,
            "saved_at": timestamp,
            "n_features": len(bundle.selected_features),
            "n_models": len(bundle.models),
            "best_model": bundle.best_model_name,
        }
    )

    joblib.dump(bundle, bundle_path, compress=3)
    logger.info("Model bundle saved to %s", bundle_path)

    # Also save a human-readable metadata sidecar
    meta_path = bundle_path.with_suffix(".json")
    serializable_meta = {
        k: (v.tolist() if isinstance(v, np.ndarray) else v)
        for k, v in bundle.metadata.items()
    }
    serializable_meta["selected_features"] = bundle.selected_features
    serializable_meta["best_model"] = bundle.best_model_name
    serializable_meta["solvent_categories"] = bundle.solvent_categories
    serializable_meta["known_primary_spacers"] = bundle.known_primary_spacers
    serializable_meta["known_secondary_spacers"] = bundle.known_secondary_spacers
    serializable_meta["benchmark"] = bundle.benchmark.to_dict(orient="records")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable_meta, f, indent=2, default=str)

    # Symlink (or copy on Windows) as "latest"
    latest_path = out_dir / "latest.joblib"
    if latest_path.exists():
        latest_path.unlink()
    try:
        latest_path.symlink_to(bundle_path.name)
    except OSError:
        # Windows may not support symlinks without admin; just copy
        import shutil
        shutil.copy2(bundle_path, latest_path)

    latest_meta = out_dir / "latest.json"
    if latest_meta.exists():
        latest_meta.unlink()
    try:
        latest_meta.symlink_to(meta_path.name)
    except OSError:
        import shutil
        shutil.copy2(meta_path, latest_meta)

    print(f"[OK] Model bundle saved to: {bundle_path}")
    print(f"  Best model: {bundle.best_model_name}")
    print(f"  Features: {len(bundle.selected_features)}")
    return bundle_path


def load_model_bundle(path: str | Path) -> ModelBundle:
    """Load a previously saved model bundle.

    Parameters
    ----------
    path : str or Path
        Path to a ``.joblib`` bundle file, or a directory containing
        ``latest.joblib``.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "latest.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    bundle: ModelBundle = joblib.load(path)
    logger.info(
        "Loaded model bundle from %s (%d models, %d features)",
        path,
        len(bundle.models),
        len(bundle.selected_features),
    )
    return bundle


def find_latest_bundle(config_name: str | None = None) -> Path | None:
    """Find the most recent model bundle.

    If *config_name* is given, look in ``models/<config_name>/``.
    Otherwise search all subdirectories of ``models/``.
    """
    if config_name:
        latest = MODELS_DIR / config_name / "latest.joblib"
        return latest if latest.exists() else None

    # Search all experiment subdirs for the newest bundle
    candidates: list[tuple[float, Path]] = []
    if MODELS_DIR.exists():
        for sub in MODELS_DIR.iterdir():
            latest = sub / "latest.joblib"
            if latest.exists():
                candidates.append((latest.stat().st_mtime, latest))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]
