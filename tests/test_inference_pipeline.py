"""Unit tests for the inference pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


class TestIUPACResolution:
    """Test SMILES resolution from IUPAC names and SMILES strings."""

    def test_known_iupac_resolves(self):
        from ml_quasit.inference_pipeline import _resolve_smiles

        result = _resolve_smiles("2-phenylethan-1-amine")
        assert result is not None
        assert "N" in result  # Should contain nitrogen

    def test_direct_smiles_passes_through(self):
        from ml_quasit.inference_pipeline import _resolve_smiles

        result = _resolve_smiles("NCCc1ccccc1")
        assert result is not None
        # RDKit should canonicalize it
        from rdkit import Chem

        mol = Chem.MolFromSmiles(result)
        assert mol is not None

    def test_invalid_input_returns_none(self):
        from ml_quasit.inference_pipeline import _resolve_smiles

        result = _resolve_smiles("this_is_not_a_real_molecule_xyz123")
        assert result is None

    def test_case_insensitive_lookup(self):
        from ml_quasit.inference_pipeline import _resolve_smiles

        result1 = _resolve_smiles("Ethanamine")
        result2 = _resolve_smiles("ethanamine")
        # Both should resolve
        assert result1 is not None
        assert result2 is not None


class TestHalideFlags:
    """Test halide flag generation."""

    def test_bromine(self):
        from ml_quasit.inference_pipeline import _halide_flags

        assert _halide_flags("Br") == (0, 1, 0)

    def test_chlorine(self):
        from ml_quasit.inference_pipeline import _halide_flags

        assert _halide_flags("Cl") == (1, 0, 0)

    def test_iodine(self):
        from ml_quasit.inference_pipeline import _halide_flags

        assert _halide_flags("I") == (0, 0, 1)

    def test_case_insensitive(self):
        from ml_quasit.inference_pipeline import _halide_flags

        assert _halide_flags("br") == (0, 1, 0)
        assert _halide_flags("CL") == (1, 0, 0)

    def test_invalid_halide_raises(self):
        from ml_quasit.inference_pipeline import _halide_flags

        with pytest.raises(ValueError, match="Invalid halide"):
            _halide_flags("F")


class TestPredictionInput:
    """Test PredictionInput validation."""

    def test_minimal_input(self):
        from ml_quasit.inference_pipeline import PredictionInput

        inp = PredictionInput(
            primary_spacer="2-phenylethan-1-amine",
            solvent="DMSO",
        )
        assert inp.primary_spacer_fraction == 1.0
        assert inp.secondary_spacer is None
        assert inp.PbBr2 == 1.0

    def test_mixed_spacer_input(self):
        from ml_quasit.inference_pipeline import PredictionInput

        inp = PredictionInput(
            primary_spacer="2-phenylethan-1-amine",
            solvent="DMF",
            secondary_spacer="butan-1-amine",
            primary_spacer_fraction=0.8,
            secondary_spacer_fraction=0.2,
        )
        assert inp.secondary_spacer is not None


class TestRDKitDescriptors:
    """Test RDKit descriptor computation."""

    def test_valid_smiles_produces_descriptors(self):
        from ml_quasit.rdkit_descriptor_features import get_2d_descriptors, DESC_NAMES

        desc = get_2d_descriptors("NCCc1ccccc1", add_hydrogens=True)
        assert len(desc) == len(DESC_NAMES)
        # At least some values should be non-None
        non_none = [d for d in desc if d is not None]
        assert len(non_none) > 100  # There are ~200 descriptors

    def test_invalid_smiles_returns_nones(self):
        from ml_quasit.rdkit_descriptor_features import get_2d_descriptors, DESC_NAMES

        desc = get_2d_descriptors("NOT_VALID", add_hydrogens=False)
        assert len(desc) == len(DESC_NAMES)
        assert all(d is None for d in desc)

    def test_add_hydrogens_changes_descriptors(self):
        from ml_quasit.rdkit_descriptor_features import get_2d_descriptors

        desc_no_h = get_2d_descriptors("NCCc1ccccc1", add_hydrogens=False)
        desc_h = get_2d_descriptors("NCCc1ccccc1", add_hydrogens=True)
        # At least some descriptors should differ
        diffs = sum(
            1
            for a, b in zip(desc_no_h, desc_h)
            if a is not None and b is not None and a != b
        )
        assert diffs > 0


class TestModelSerialization:
    """Test model bundle save/load round-trip."""

    def test_model_bundle_creation(self):
        from ml_quasit.model_serialization import ModelBundle

        bundle = ModelBundle(
            models={"test": None},
            scaler=None,
            selected_features=["feat1", "feat2"],
            solvent_categories=["DMF", "DMSO"],
            add_hydrogens=True,
            best_model_name="test",
            benchmark=pd.DataFrame({"Model": ["test"], "R2 Score": [0.95]}),
            known_primary_spacers=["spacer1"],
            known_secondary_spacers=["spacer2"],
        )
        assert bundle.best_model_name == "test"
        assert len(bundle.selected_features) == 2

    def test_save_load_roundtrip(self, tmp_path):
        """Test that a bundle can be saved and loaded."""
        from ml_quasit.model_serialization import ModelBundle
        import joblib

        bundle = ModelBundle(
            models={"dummy": "dummy_model"},
            scaler=None,
            selected_features=["f1", "f2", "f3"],
            solvent_categories=["DMF", "DMSO"],
            add_hydrogens=False,
            best_model_name="dummy",
            benchmark=pd.DataFrame(
                {"Model": ["dummy"], "R2 Score": [0.9], "RMSE (eV)": [0.05]}
            ),
            known_primary_spacers=["spacer_a"],
            known_secondary_spacers=[],
        )

        path = tmp_path / "test_bundle.joblib"
        joblib.dump(bundle, path)

        loaded = joblib.load(path)
        assert loaded.best_model_name == "dummy"
        assert loaded.selected_features == ["f1", "f2", "f3"]
        assert loaded.add_hydrogens is False


class TestInputValidation:
    """Test input validation logic."""

    def test_invalid_fraction_raises(self):
        from ml_quasit.inference_pipeline import PredictionInput, _validate_input
        from ml_quasit.model_serialization import ModelBundle

        bundle = ModelBundle(
            models={},
            scaler=None,
            selected_features=[],
            solvent_categories=["DMSO"],
            add_hydrogens=False,
            best_model_name="test",
            benchmark=pd.DataFrame(),
            known_primary_spacers=["2-phenylethan-1-amine"],
            known_secondary_spacers=[],
        )

        inp = PredictionInput(
            primary_spacer="2-phenylethan-1-amine",
            solvent="DMSO",
            primary_spacer_fraction=1.5,  # Invalid
        )

        with pytest.raises(ValueError, match="between 0 and 1"):
            _validate_input(inp, bundle)

    def test_missing_secondary_spacer_raises(self):
        from ml_quasit.inference_pipeline import PredictionInput, _validate_input
        from ml_quasit.model_serialization import ModelBundle

        bundle = ModelBundle(
            models={},
            scaler=None,
            selected_features=[],
            solvent_categories=["DMSO"],
            add_hydrogens=False,
            best_model_name="test",
            benchmark=pd.DataFrame(),
            known_primary_spacers=["2-phenylethan-1-amine"],
            known_secondary_spacers=[],
        )

        inp = PredictionInput(
            primary_spacer="2-phenylethan-1-amine",
            solvent="DMSO",
            secondary_spacer_fraction=0.3,  # No secondary spacer specified
        )

        with pytest.raises(ValueError, match="no secondary_spacer"):
            _validate_input(inp, bundle)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
