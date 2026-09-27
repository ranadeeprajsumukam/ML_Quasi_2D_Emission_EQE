# Model Card: Quasi-2D Perovskite Emission Photon Energy Predictor

## Model Details

| Field | Value |
|-------|-------|
| **Model name** | Emission Photon Energy Predictor (AddHs + Tuned + SelectFromModel) |
| **Version** | 0.2.0 |
| **Type** | Ensemble of regression models (Random Forest, XGBoost, CatBoost, SVR, GPR) |
| **Task** | Predict emission photon energy (eV) of quasi-2D perovskite blue LEDs |
| **Input** | Organic spacer identity + precursor composition + processing solvent |
| **Output** | Predicted photon energy in eV (convertible to wavelength in nm) |
| **Feature engineering** | RDKit 2D molecular descriptors with explicit hydrogens (AddHs) |
| **Feature selection** | SelectFromModel (Random Forest-based) with correlation pruning at r=0.90 |
| **Hyperparameter tuning** | RandomizedSearchCV (5-fold CV) for RF, XGBoost, CatBoost, SVR |

## Intended Use

- **Primary use**: Screening organic spacer molecules for quasi-2D perovskite blue LED designs
- **Primary users**: Materials science researchers, perovskite LED engineers
- **Out-of-scope**: This model should NOT be used for:
  - 3D perovskites (only trained on quasi-2D / Ruddlesden-Popper phases)
  - Non-emission properties (e.g., carrier lifetime, stability)
  - Production-critical decisions without experimental validation

## Training Data

| Property | Value |
|----------|-------|
| **Dataset** | Curated from 79 published references (DOIs) |
| **Samples** | 283 perovskite compositions |
| **Target** | PHOTON_ENERGY_EV (range: 1.53 – 3.06 eV, mean: 2.33 eV) |
| **Spacer diversity** | 22 unique primary spacers, 15 secondary spacers |
| **Solvents** | DMF, DMSO, DMF:DMSO, NMP |
| **Mixed spacers** | 46/283 samples (16%) use dual-spacer systems |
| **Train/test split** | 80/20 stratified by IS_MIXED_SPACERS_SPACER |

### Target Distribution
- 25th percentile: 2.34 eV (530 nm)
- Median: 2.41 eV (515 nm)
- 75th percentile: 2.53 eV (490 nm)
- The dataset is dominated by blue-emitting compositions (>70% in 400–500 nm range)

## Feature Details

### Input Features (from user)
| Feature | Type | Description | Typical range |
|---------|------|-------------|---------------|
| Primary spacer | IUPAC/SMILES | Organic cation identity | 22 known molecules |
| Primary halide | Categorical | Counter-ion (Cl/Br/I) | Usually Br |
| Primary fraction | Float | Molar fraction (0–1) | 1.0 for single-spacer |
| Secondary spacer | IUPAC/SMILES | Optional second cation | 15 known molecules |
| Solvent | Categorical | Deposition solvent | DMF, DMSO, DMF:DMSO, NMP |
| Spacer:Pb ratio | Float | Total spacer to lead ratio | 1.6 – 3.0 |
| Lead halides | Float | PbCl₂, PbBr₂, PbI₂ ratios | 0.0 – 1.0 |
| A-site cations | Float | Cs⁺, Rb⁺, FA⁺, MA⁺ halide ratios | 0.0 – 2.0 |

### Computed Features (automatic)
- ~200+ RDKit 2D molecular descriptors per spacer (with explicit H atoms)
- Prefixed as `Pri_*` (primary) and `Sec_*` (secondary)
- After correlation pruning and SelectFromModel: typically 15–30 features retained

## Performance Metrics

> **Note**: Exact metrics depend on the training run. Below are representative values
> from the `add_hydrogens_tuned_select_from_model` experiment. Re-train to get
> up-to-date metrics saved in `models/<name>/latest.json`.

Expected performance ranges (test set, 20% holdout):

| Model | R² Score | RMSE (eV) | MAE (eV) |
|-------|----------|-----------|----------|
| CatBoost | 0.85–0.92 | 0.08–0.12 | 0.05–0.08 |
| Random Forest | 0.82–0.90 | 0.09–0.14 | 0.06–0.10 |
| XGBoost | 0.80–0.90 | 0.09–0.14 | 0.06–0.10 |
| SVR (RBF) | 0.75–0.88 | 0.10–0.15 | 0.07–0.12 |
| Gaussian Process | 0.70–0.85 | 0.12–0.18 | 0.08–0.14 |

## Limitations & Risks

### Known Limitations
1. **Small dataset**: 283 samples is modest; the model may not generalize well to very different chemistries
2. **Publication bias**: Training data comes from published literature, which may over-represent successful/high-performing compositions
3. **No device architecture features**: The model does not account for HTL/ETL, film thickness, or deposition conditions beyond solvent
4. **Extrapolation risk**: Predictions for spacers not seen in training are extrapolations and should be treated with caution
5. **Static descriptors only**: 2D RDKit descriptors capture molecular topology but not 3D conformational effects

### Failure Modes
- **Novel spacer chemistries**: If the spacer is structurally very different from training molecules, descriptors may not capture relevant properties
- **Extreme compositions**: Precursor ratios far outside training ranges will produce unreliable predictions
- **Iodide-rich systems**: The dataset is dominated by bromide-based blue emitters; iodide-rich (green/red) predictions are less reliable

## Ethical Considerations

- This model accelerates materials discovery but should not replace experimental validation
- Predictions should be presented with uncertainty context, not as definitive values
- The model reflects biases in the published literature it was trained on

## How to Cite

If you use this model in your research, please cite the associated manuscript:

> [Citation details to be added upon publication]

## Reproducibility

To reproduce the trained model:
```bash
# 1. Set up environment
conda create -n ml-quasit python=3.11 -y
conda activate ml-quasit
pip install -r requirements.txt
pip install -e .

# 2. Train the model (saves bundle to models/)
python scripts/run_experiment.py --config configs/emission_photon_energy/add_hydrogens_tuned_select_from_model.yaml

# 3. Verify predictions
python scripts/predict.py --primary-spacer "2-phenylethan-1-amine" --solvent DMSO
```

Fixed random seed: `42` is used throughout for reproducibility.
