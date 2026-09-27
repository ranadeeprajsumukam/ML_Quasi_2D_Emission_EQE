# Models Directory

Trained model bundles are saved here by `run_experiment.py`.

Each experiment creates a subdirectory containing:
- `bundle_<timestamp>.joblib` — Full model bundle (models, scaler, features, metadata)
- `bundle_<timestamp>.json` — Human-readable metadata sidecar
- `latest.joblib` — Symlink/copy to the most recent bundle
- `latest.json` — Symlink/copy to the most recent metadata

## Usage

```python
from ml_quasit import EmissionPredictor, PredictionInput

predictor = EmissionPredictor("models/emission_photon_energy_add_hydrogens_tuned_select_from_model")
result = predictor.predict(
    PredictionInput(
        primary_spacer="2-phenylethan-1-amine",
        solvent="DMSO",
    )
)
print(f"{result.predicted_ev:.4f} eV ({result.predicted_nm:.1f} nm)")
```
