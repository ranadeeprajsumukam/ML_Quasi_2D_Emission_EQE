"""Command-line interface for emission photon energy prediction.

Usage examples
--------------
Single prediction (interactive):
    python scripts/predict.py

Single prediction (command-line args):
    python scripts/predict.py \\
        --primary-spacer "2-phenylethan-1-amine" \\
        --solvent DMSO \\
        --PbBr2 1.0 --CsBr 1.0

Batch prediction from CSV:
    python scripts/predict.py --batch inputs.csv --output predictions.csv

List available models:
    python scripts/predict.py --list-models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for standalone execution
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd

from ml_quasit.inference_pipeline import (
    EmissionPredictor,
    PredictionInput,
    PredictionResult,
    KNOWN_SOLVENTS,
    PRECURSOR_RATIO_COLUMNS,
)
from ml_quasit.iupac_to_smiles import MASTER_SMILES_MAP
from ml_quasit.model_serialization import MODELS_DIR, find_latest_bundle


def _find_model(model_arg: str | None) -> Path:
    """Resolve the model path from CLI argument or find latest."""
    if model_arg:
        p = Path(model_arg)
        if p.exists():
            return p
        # Try relative to models dir
        p = MODELS_DIR / model_arg
        if p.exists():
            return p
        # Try as config name
        p = MODELS_DIR / model_arg / "latest.joblib"
        if p.exists():
            return p
        print(f"Error: Model not found at '{model_arg}'")
        sys.exit(1)

    latest = find_latest_bundle()
    if latest is None:
        print(
            "Error: No trained model found. Run training first:\n"
            "  python scripts/run_experiment.py "
            "--config configs/emission_photon_energy/"
            "add_hydrogens_tuned_select_from_model.yaml"
        )
        sys.exit(1)
    return latest


def _print_header():
    print("=" * 60)
    print("  Quasi-2D Perovskite Emission Energy Predictor")
    print("=" * 60)


def _interactive_predict(predictor: EmissionPredictor):
    """Run an interactive prediction session."""
    _print_header()

    # Show known spacers
    print("\n📋 Known primary spacers (from training data):")
    for i, s in enumerate(predictor.known_primary_spacers, 1):
        print(f"   {i:2d}. {s}")

    print(f"\n📋 Available solvents: {', '.join(KNOWN_SOLVENTS)}")

    # Primary spacer
    print("\n── Primary Spacer ──")
    spacer_input = input(
        "Enter IUPAC name, SMILES, or number from list above: "
    ).strip()
    if spacer_input.isdigit():
        idx = int(spacer_input) - 1
        if 0 <= idx < len(predictor.known_primary_spacers):
            primary_spacer = predictor.known_primary_spacers[idx]
        else:
            print("Invalid number.")
            return
    else:
        primary_spacer = spacer_input

    # Primary halide
    halide = (
        input("Primary spacer halide [Cl/Br/I] (default: Br): ").strip() or "Br"
    )

    # Solvent
    solvent = input(f"Solvent [{'/'.join(KNOWN_SOLVENTS)}] (default: DMSO): ").strip()
    if not solvent:
        solvent = "DMSO"

    # Mixed spacers?
    is_mixed = input("Mixed spacer system? [y/N]: ").strip().lower()
    secondary_spacer = None
    secondary_halide = "Br"
    primary_fraction = 1.0
    secondary_fraction = 0.0

    if is_mixed == "y":
        print("\n📋 Known secondary spacers:")
        for i, s in enumerate(predictor.known_secondary_spacers, 1):
            print(f"   {i:2d}. {s}")
        sec_input = input(
            "Enter secondary spacer (IUPAC/SMILES/number): "
        ).strip()
        if sec_input.isdigit():
            idx = int(sec_input) - 1
            if 0 <= idx < len(predictor.known_secondary_spacers):
                secondary_spacer = predictor.known_secondary_spacers[idx]
            else:
                print("Invalid number.")
                return
        else:
            secondary_spacer = sec_input

        secondary_halide = (
            input("Secondary halide [Cl/Br/I] (default: Br): ").strip() or "Br"
        )
        primary_fraction = float(
            input("Primary spacer molar fraction (default: 0.8): ").strip() or "0.8"
        )
        secondary_fraction = float(
            input("Secondary spacer molar fraction (default: 0.2): ").strip() or "0.2"
        )

    # Spacer-to-Pb ratio
    spacer_to_pb = float(
        input("Spacer-to-Pb molar ratio (default: 2.0): ").strip() or "2.0"
    )

    # Precursor composition
    print("\n── Precursor Composition (molar ratios to Pb) ──")
    print("Press Enter to use default (0.0). Common: PbBr2=1.0, CsBr=1.0")
    precursors = {}
    PRECURSOR_DISPLAY = {
        "PbCl2": "PbCl₂",
        "PbBr2": "PbBr₂",
        "PbI2": "PbI₂",
        "CsCl_TO_Pb": "CsCl/Pb",
        "CsBr_TO_Pb": "CsBr/Pb",
        "CsI_TO_Pb": "CsI/Pb",
        "RbBr_TO_Pb": "RbBr/Pb",
        "RbI_TO_Pb": "RbI/Pb",
        "FABr_TO_Pb": "FABr/Pb",
        "FAI_TO_Pb": "FAI/Pb",
        "MABr_TO_Pb": "MABr/Pb",
        "MAI_TO_Pb": "MAI/Pb",
    }
    DEFAULTS = {"PbBr2": 1.0, "CsBr_TO_Pb": 1.0}

    for col in PRECURSOR_RATIO_COLUMNS:
        default = DEFAULTS.get(col, 0.0)
        display = PRECURSOR_DISPLAY.get(col, col)
        val = input(f"  {display} (default: {default}): ").strip()
        precursors[col] = float(val) if val else default

    # Build input
    inp = PredictionInput(
        primary_spacer=primary_spacer,
        solvent=solvent,
        primary_halide=halide,
        primary_spacer_fraction=primary_fraction,
        secondary_spacer=secondary_spacer,
        secondary_halide=secondary_halide,
        secondary_spacer_fraction=secondary_fraction,
        spacer_to_pb_ratio=spacer_to_pb,
        **precursors,
    )

    print("\n⏳ Computing molecular descriptors and predicting...")
    result = predictor.predict(inp, return_features=False)
    _print_result(result)


def _print_result(result: PredictionResult):
    """Pretty-print a prediction result."""
    print("\n" + "=" * 60)
    print("  PREDICTION RESULTS")
    print("=" * 60)
    print(f"\n  🔬 Model used:       {result.model_name}")
    print(f"  ⚡ Photon Energy:    {result.predicted_ev:.4f} eV")
    print(f"  🌈 Wavelength:       {result.predicted_nm:.1f} nm")

    # Color indication
    nm = result.predicted_nm
    if nm < 400:
        color = "UV/Violet"
    elif nm < 450:
        color = "Violet-Blue"
    elif nm < 495:
        color = "Blue"
    elif nm < 520:
        color = "Blue-Green"
    elif nm < 565:
        color = "Green"
    elif nm < 590:
        color = "Yellow"
    elif nm < 625:
        color = "Orange"
    else:
        color = "Red/NIR"
    print(f"  🎨 Color region:     {color}")

    print(f"\n  {result.confidence_note}")

    print("\n  All model predictions:")
    for name, ev in sorted(
        result.all_model_predictions.items(), key=lambda x: x[1]
    ):
        nm_val = 1240.0 / ev if ev > 0 else float("nan")
        print(f"    {name:20s}: {ev:.4f} eV  ({nm_val:.1f} nm)")
    print()


def _batch_predict(
    predictor: EmissionPredictor,
    csv_path: str,
    output_path: str | None,
    model_name: str | None,
):
    """Run batch predictions from a CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    required = ["primary_spacer", "solvent"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Error: CSV missing required columns: {missing}")
        print(
            "Required columns: primary_spacer, solvent\n"
            "Optional columns: primary_halide, secondary_spacer, "
            "secondary_halide, primary_spacer_fraction, "
            "secondary_spacer_fraction, spacer_to_pb_ratio, "
            + ", ".join(PRECURSOR_RATIO_COLUMNS)
        )
        sys.exit(1)

    inputs = []
    for _, row in df.iterrows():
        kwargs = {
            "primary_spacer": str(row["primary_spacer"]),
            "solvent": str(row["solvent"]),
        }
        # Optional fields
        optional_str = [
            "primary_halide",
            "secondary_spacer",
            "secondary_halide",
        ]
        optional_float = [
            "primary_spacer_fraction",
            "secondary_spacer_fraction",
            "spacer_to_pb_ratio",
        ] + PRECURSOR_RATIO_COLUMNS

        for col in optional_str:
            if col in row and pd.notna(row[col]):
                kwargs[col] = str(row[col])

        for col in optional_float:
            if col in row and pd.notna(row[col]):
                kwargs[col] = float(row[col])

        inputs.append(PredictionInput(**kwargs))

    results_df = predictor.predict_batch(inputs, model_name=model_name)

    out = output_path or csv_path.replace(".csv", "_predictions.csv")
    results_df.to_csv(out, index=False)
    print(f"\n✓ Predictions saved to: {out}")
    print(results_df[["primary_spacer", "solvent", "predicted_ev", "predicted_nm"]].to_string())


def _list_models():
    """List all available model bundles."""
    if not MODELS_DIR.exists():
        print("No models directory found. Train a model first.")
        return

    found = False
    for sub in sorted(MODELS_DIR.iterdir()):
        if sub.is_dir():
            latest = sub / "latest.json"
            if latest.exists():
                import json
                with open(latest) as f:
                    meta = json.load(f)
                found = True
                print(f"\n📦 {sub.name}")
                print(f"   Best model: {meta.get('best_model', 'N/A')}")
                print(f"   Features:   {meta.get('n_features', 'N/A')}")
                print(f"   Saved at:   {meta.get('saved_at', 'N/A')}")
                if "benchmark" in meta:
                    print("   Benchmark:")
                    for row in meta["benchmark"][:3]:
                        print(
                            f"     {row['Model']:20s} "
                            f"R²={row.get('R2 Score', 0):.4f}"
                        )
    if not found:
        print("No trained model bundles found.")


def main():
    parser = argparse.ArgumentParser(
        description="Predict emission photon energy for quasi-2D perovskites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python scripts/predict.py  # interactive mode\n'
            '  python scripts/predict.py --primary-spacer "2-phenylethan-1-amine" '
            "--solvent DMSO\n"
            "  python scripts/predict.py --batch inputs.csv\n"
            "  python scripts/predict.py --list-models\n"
        ),
    )

    parser.add_argument(
        "--model",
        help="Path to model bundle or experiment name (default: latest)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available trained models",
    )

    # Single prediction args
    pred = parser.add_argument_group("single prediction")
    pred.add_argument("--primary-spacer", help="Primary spacer (IUPAC or SMILES)")
    pred.add_argument("--solvent", help="Solvent name")
    pred.add_argument("--primary-halide", default="Br", help="Primary halide (default: Br)")
    pred.add_argument("--secondary-spacer", help="Secondary spacer (optional)")
    pred.add_argument("--secondary-halide", default="Br", help="Secondary halide")
    pred.add_argument("--primary-fraction", type=float, default=1.0)
    pred.add_argument("--secondary-fraction", type=float, default=0.0)
    pred.add_argument("--spacer-to-pb", type=float, default=2.0)
    pred.add_argument("--use-model", help="Name of specific model in bundle")

    # Precursor ratios
    chem = parser.add_argument_group("precursor composition (molar ratios to Pb)")
    chem.add_argument("--PbCl2", type=float, default=0.0)
    chem.add_argument("--PbBr2", type=float, default=1.0)
    chem.add_argument("--PbI2", type=float, default=0.0)
    chem.add_argument("--CsCl", type=float, default=0.0, dest="CsCl_TO_Pb")
    chem.add_argument("--CsBr", type=float, default=1.0, dest="CsBr_TO_Pb")
    chem.add_argument("--CsI", type=float, default=0.0, dest="CsI_TO_Pb")
    chem.add_argument("--RbBr", type=float, default=0.0, dest="RbBr_TO_Pb")
    chem.add_argument("--RbI", type=float, default=0.0, dest="RbI_TO_Pb")
    chem.add_argument("--FABr", type=float, default=0.0, dest="FABr_TO_Pb")
    chem.add_argument("--FAI", type=float, default=0.0, dest="FAI_TO_Pb")
    chem.add_argument("--MABr", type=float, default=0.0, dest="MABr_TO_Pb")
    chem.add_argument("--MAI", type=float, default=0.0, dest="MAI_TO_Pb")

    # Batch prediction
    batch = parser.add_argument_group("batch prediction")
    batch.add_argument("--batch", help="Path to CSV file with batch inputs")
    batch.add_argument("--output", help="Output CSV path (default: input_predictions.csv)")

    args = parser.parse_args()

    if args.list_models:
        _list_models()
        return

    model_path = _find_model(args.model)
    print(f"Loading model from: {model_path}")
    predictor = EmissionPredictor(model_path)

    if args.batch:
        _batch_predict(predictor, args.batch, args.output, args.use_model)
    elif args.primary_spacer and args.solvent:
        # Direct CLI prediction
        precursors = {
            col: getattr(args, col, 0.0) for col in PRECURSOR_RATIO_COLUMNS
        }
        inp = PredictionInput(
            primary_spacer=args.primary_spacer,
            solvent=args.solvent,
            primary_halide=args.primary_halide,
            primary_spacer_fraction=args.primary_fraction,
            secondary_spacer=args.secondary_spacer,
            secondary_halide=args.secondary_halide,
            secondary_spacer_fraction=args.secondary_fraction,
            spacer_to_pb_ratio=args.spacer_to_pb,
            **precursors,
        )
        result = predictor.predict(inp)
        _print_result(result)
    else:
        # Interactive mode
        _interactive_predict(predictor)


if __name__ == "__main__":
    main()
