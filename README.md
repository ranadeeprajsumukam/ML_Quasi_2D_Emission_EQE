# Cheminformatics and Machine Learning for Quasi-2D Perovskites: Emission Energy Prediction and the Limits of EQE Modelling

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mlquasi2demissioneqe-8yvtmbep6kul3izkcuafzs.streamlit.app/)

Official repository containing the codebase, machine learning pipelines, trained models, Jupyter notebooks, and interactive web application for **"Cheminformatics and Machine Learning for Quasi-2D Perovskites: Emission Energy Prediction and the Limits of EQE Modelling"**.

---

## 🌐 Live Interactive Web Application

An interactive web application powered by the champion 13-feature SVR model is publicly hosted on Streamlit Cloud:

👉 **[https://mlquasi2demissioneqe-8yvtmbep6kul3izkcuafzs.streamlit.app/](https://mlquasi2demissioneqe-8yvtmbep6kul3izkcuafzs.streamlit.app/)**

The application supports:
- **Instant SMILES & 2D chemical structure rendering** via RDKit from IUPAC names.
- **Real-time emission photon energy (eV) and peak wavelength (nm)** prediction with visual color swath mapping.
- **Precursor and solvent stoichiometry customization** (e.g., PbBr₂, PbCl₂, A-site cations, DMF/DMSO).
- **Batch CSV screening** with downloadable result tables.

---

## 📁 Repository Structure

```
ML_Quasi_2D_Emission_EQE/
├── app/                              # Streamlit web application
│   ├── streamlit_app.py              # Application source code
│   └── assets/                       # UI assets and logos
├── configs/                          # Experiment configuration YAMLs
│   ├── emission_photon_energy/       # Photon energy (eV) configs
│   ├── emission_wavelength_nm/       # Wavelength (nm) configs
│   └── eqe_percent/                  # EQE (%) configs
├── data/                             # Dataset schema & guidelines
│   └── README.md                     # Data format and sheet specifications
├── models/                           # Serialized model bundles and metadata
│   ├── README.md
│   └── emission_photon_energy_add_hydrogens_tuned_select_from_model/
│       ├── latest.joblib             # Champion model bundle
│       └── latest.json               # Metadata sidecar
├── notebooks/                        # Jupyter notebooks for full pipeline walkthroughs
│   ├── emission_photon_energy/       # Emission energy models (RFECV, SelectFromModel, tuning)
│   ├── emission_wavelength_nm/       # Emission peak wavelength models
│   └── eqe_percent/                  # Device-level EQE models
├── scripts/                          # Execution and reproduction CLI scripts
│   ├── predict.py                    # Standalone CLI prediction tool
│   ├── retrain_13features.py         # Retrain the champion 13-feature SVR model
│   ├── generate_parity_and_cv.py     # Generate parity plots & 10-fold CV metrics
│   └── run_experiment.py             # Orchestrate experiments from YAML configs
├── src/ml_quasit/                    # Core Python package
│   ├── excel_dataset_loader.py       # Data extraction and preprocessing
│   ├── experiment_config.py          # YAML configuration schemas
│   ├── experiment_results_plots.py   # Publication figure plotting
│   ├── inference_pipeline.py         # End-to-end inference engine
│   ├── iupac_to_smiles.py            # IUPAC to SMILES resolution
│   ├── model_serialization.py        # Model artifact loading and export
│   ├── rdkit_descriptor_features.py  # 2D molecular descriptor generation
│   ├── regression_model_trainer.py   # Regressor training & hyperparameter tuning
│   └── train_test_feature_selection.py # Feature filtering & selection algorithms
├── tests/                            # Unit test suite
├── Dockerfile                        # Docker container specification
├── MODEL_CARD.md                     # Model Card specifications
├── pyproject.toml                    # Package build configuration
└── requirements.txt                  # Python dependencies
```

---

## ⚙️ Installation & Setup

### Option 1: Conda Environment (Recommended)

```bash
git clone https://github.com/ranadeeprajsumukam/ML_Quasi_2D_Emission_EQE.git
cd ML_Quasi_2D_Emission_EQE

conda create -n ml-quasit python=3.11 -y
conda activate ml-quasit

pip install -r requirements.txt
pip install -e .
```

### Option 2: Docker Container

```bash
docker build -t perovskite-predictor .
docker run -p 8501:8501 perovskite-predictor
```
Then navigate to `http://localhost:8501` in your browser.

---

## 🚀 Quickstart & Usage

### 1. Launch Web Application Locally

```bash
streamlit run app/streamlit_app.py
```

### 2. Command-Line Prediction

Run single predictions directly from the command line:

```bash
python scripts/predict.py \
    --primary-spacer "2-phenylethan-1-amine" \
    --solvent DMSO \
    --PbBr2 1.0 \
    --CsBr 1.0
```

Interactive guided mode:
```bash
python scripts/predict.py
```

Batch prediction from CSV:
```bash
python scripts/predict.py --batch inputs.csv --output predictions.csv
```

### 3. Reproduce Manuscript Figures & Metrics

To generate the parity plot and run 10-fold cross-validation with 95% bootstrap confidence intervals:

```bash
python scripts/generate_parity_and_cv.py
```

To retrain the champion 13-feature SVR model:

```bash
python scripts/retrain_13features.py
```

---

## 📊 Summary of Experiments & Notebooks

| Notebook / Config | Target Property | Feature Selection Strategy | Hyperparameter Tuning |
|---|---|---|---|
| `emission_photon_energy/baseline_rfecv` | PHOTON_ENERGY_EV | RFECV (31 features) | Baseline |
| `emission_photon_energy/tuned_rfecv` | PHOTON_ENERGY_EV | RFECV (31 features) | RandomizedSearchCV |
| `emission_photon_energy/tuned_select_from_model` | PHOTON_ENERGY_EV | SelectFromModel (13 features) | RandomizedSearchCV |
| `emission_photon_energy/add_hydrogens_baseline_rfecv` | PHOTON_ENERGY_EV | RFECV + AddHs | Baseline |
| `emission_photon_energy/add_hydrogens_tuned_rfecv` | PHOTON_ENERGY_EV | RFECV + AddHs | RandomizedSearchCV |
| **`emission_photon_energy/add_hydrogens_tuned_select_from_model`** *(Champion)* | **PHOTON_ENERGY_EV** | **SelectFromModel + AddHs (13 features)** | **RandomizedSearchCV (SVR R² = 0.953)** |
| `emission_wavelength_nm/select_from_model_baseline` | EMISSION_PEAK_NM | SelectFromModel | Baseline |
| `emission_wavelength_nm/rfecv_baseline` | EMISSION_PEAK_NM | RFECV | Baseline |
| `emission_wavelength_nm/rfecv_tuned` | EMISSION_PEAK_NM | RFECV | RandomizedSearchCV |
| `eqe_percent/rfe_20_features_tuned` | EQE_percent | RFE (20 features) | RandomizedSearchCV |
| `eqe_percent/catboost_only` | EQE_percent | Categorical HTL/ETL Encoding | CatBoost Regressor |

To run all configured experiments from the CLI:

```bash
python scripts/run_experiment.py --all
```

---

## 🧪 Testing

To execute the test suite:

```bash
pytest tests/ -v
```

---

## 📄 Citation

If you use this codebase, models, or dataset in your research, please cite:

```bibtex
@article{sumukam2026quasi2d,
  title={Cheminformatics and Machine Learning for Quasi-2D Perovskites: Emission Energy Prediction and the Limits of EQE Modelling},
  author={Sumukam, Ranadeep Raj and Savu, Ramu Naidu and Rompivalasa, Santosh and Banavoth, Murali},
  journal={Journal of the American Chemical Society (Under Review)},
  year={2026}
}
```

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).
