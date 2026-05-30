# Supporting Information

## Machine-Learning Prediction of Quasi-2D Perovskite LED Photon Energy and EQE

**Companion to main text:** `MAIN_Manuscript.md`

---

## Overview

This Supporting Information (SI) collects the numerical detail behind the main manuscript: spectral distributions, complete descriptor lists, every emission notebook we ran (primary and ablation), wavelength-targeted models, EQE benchmarks, and reproduction notes. The **primary scientific conclusions** rest on two notebooks only—`add_hydrogens_tuned_select_from_model.ipynb` (thirteen descriptors, SelectFromModel) and `add_hydrogens_tuned_rfecv.ipynb` (thirty-one descriptors, RFECV)—both using explicit-hydrogen RDKit graphs and hyperparameter-tuned regressors. Everything else in the SI is included so reviewers (and future readers in your group) can see what changes when hydrogens are omitted, when tuning is turned off, or when the target is wavelength instead of photon energy.

If you are tracing a number back to its origin, use **Table S11** for a one-page comparison of the best SVR runs, then open the notebook named in the table caption. Feature names for the thirteen- and thirty-one-variable models are in **Tables S3 and S4**.

---

## Table of Contents

| Section | Topic |
|---------|--------|
| S1 | Dataset statistics and spectral distribution |
| S2 | Emission wavelength (nm) models |
| S3 | Feature counts: 13 vs 31 vs 39 (AddHs vs no AddHs) |
| S4 | Complete feature lists (primary analysis) |
| S5 | Emission *E*<sub>ph</sub> — all notebook benchmarks |
| S6 | SHAP and figure index |
| S7 | EQE models and fabrication variables |
| S8 | Hyperparameters, configs, and reproduction |
| S9 | Bibliography with full paper titles |
| S10 | SHAP–mechanism table (SVR, 13 features) |

---

## S1. Dataset Statistics

The emission corpus is drawn from `data/SF_DATA_EMISSION_PEAK.xlsx`, sheet `SF_DATA_EMISSION_PEAK_NEW`. Each row is a literature PeLED report with extracted precursor stoichiometry, spacer IUPAC names, solvent, and a reported photon energy (eV) and/or peak wavelength (nm). We did not restrict the compilation to a single colour target; the intent was to cover the visible range of quasi-2D devices that appear in the peer-reviewed record.

| Statistic | Value |
|-----------|-------|
| Records (*N*) | 283 |
| Training | 226 (80%) |
| Testing | 57 (20%) |
| Stratification key | `IS_MIXED_SPACERS_SPACER` (> 0 vs. 0) |
| *E*<sub>ph</sub> (eV) | 1.53 – 3.06 |
| λ (nm) | 405 – 810 |
| Mixed-spacer count | 46 (16.3%) |
| Unique primary IUPAC | 22 |

### S1.1 Wavelength bin distribution

The histogram is skewed toward green–yellow emission, which reflects the historical focus of the field rather than a sampling bias introduced by us. Violet/blue and red/NIR entries are still numerous enough that a single global model must learn halide and stoichiometry effects across a wide band.

| Bin (nm) | Label | *n* | Fraction |
|----------|-------|-----|----------|
| 400 – 480 | Violet / blue | 35 | 12.4% |
| 480 – 550 | Green / yellow | 195 | 69.0% |
| 550 – 650 | Orange / red | 6 | 2.1% |
| 650 – 900 | Deep red / NIR | 47 | 16.6% |

### S1.2 Matrix dimensions (AddHs primary path)

| Stage | Shape | Notebook output |
|-------|-------|-----------------|
| Raw Excel | 283 × 37 | `(283, 37)` |
| After AddHs RDKit + solvent OHE | 283 × 462 | `(283, 462)` |
| Unique SMILES processed | 30 | `Found 30 unique molecules. Calculating 217 descriptors...` |
| After \|*r*\| > 0.90 on training fold | 289 columns retained | **173** columns dropped |

---

## S2. Emission Wavelength (nm) Models

The main manuscript discusses photon energy because it maps directly onto confinement physics. Wavelength is reported here for completeness and for comparison with prior composition-only ML work.[9] Models in this section were built from `notebooks/emission_wavelength_nm/rfecv_tuned.ipynb` **without** the AddHs protocol used in the primary analysis; RFECV retained **28** features after dropping **172** correlated columns.

Gaussian process regression performs best on the fifty-seven-record test fold (*R*<sup>2</sup> = 0.958, RMSE ≈ 21 nm). **Important:** the notebook’s printed table header incorrectly labels RMSE as “(eV)”; the numerical values are in nanometres (21.0 is plausible for nm; it is not plausible for eV). We corrected the unit in Table S2 below.

### Table S1. Wavelength pipeline summary

| Step | Value |
|------|-------|
| Correlated features dropped | 172 |
| RFECV optimal features | 28 |
| Train / test shapes | (226, 28) / (57, 28) |

### Table S2. Tuned wavelength benchmark (test set, *n* = 57)

| Model | Test *R*<sup>2</sup> | RMSE (nm) | MAE (nm) |
|-------|---------------|-----------|----------|
| Gaussian Process | **0.958** | **21.0** | **12.5** |
| CatBoost | 0.942 | 24.6 | 14.3 |
| XGBoost | 0.941 | 24.9 | 14.1 |
| Random Forest | 0.816 | 43.8 | 20.0 |
| SVR (RBF) | 0.806 | 45.0 | 25.2 |

**Suggested SI figures:** histogram of λ (Fig. S1); parity plot for GPR (Fig. S2); RFECV curve at 28 features (Fig. S3); SHAP beeswarms per model (Figs. S4–S8).

---

## S3. Feature Counts: Thirteen, Thirty-One, and Thirty-Nine

A common point of confusion in our internal drafts was the RFECV optimum of **39** features, which appears only when explicit hydrogens are **not** added before RDKit descriptor calculation (`baseline_rfecv.ipynb`, `tuned_rfecv.ipynb`). The **published primary analysis** uses AddHs throughout and finds RFECV optima of **31** features, with **13** features from SelectFromModel.

### Table S3. Notebook comparison

| Notebook | AddHs? | Tuned? | \|*r*\| dropped | RFECV optimum | SelectFromModel |
|----------|--------|--------|----------------|---------------|-----------------|
| **`add_hydrogens_tuned_rfecv`** | **Yes** | **Yes** | **173** | **31** | — |
| **`add_hydrogens_tuned_select_from_model`** | **Yes** | **Yes** | **173** | — | **13** |
| `add_hydrogens_baseline_rfecv` | Yes | No | 173 | 31 | — |
| `baseline_rfecv` | No | No | 172 | 39 | — |
| `tuned_rfecv` | No | Yes | 172 | 39 | — |
| `tuned_select_from_model` | No | Yes | 172 | — | 13 |

### Table S4. Sample-to-feature ratios (*N*<sub>train</sub> = 226)

| Feature set | *p* | *N*/*p* |
|-------------|-----|--------|
| SelectFromModel, AddHs (**primary**) | 13 | **17.38** |
| RFECV, AddHs (**primary**) | 31 | **7.29** |
| RFECV, no AddHs (ablation) | 39 | 5.79 |

---

## S4. Complete Feature Lists (AddHs Primary Analysis)

### Table S5. Thirteen features — `add_hydrogens_tuned_select_from_model.ipynb`

```
CL_PRIMARY_ORGANIC_HALIDE
BR_PRIMARY_ORGANIC_HALIDE
SPACER_TO_PB_RATIO
PbCl2
PbBr2
CsBr_TO_Pb
CsI_TO_Pb
FABr_TO_Pb
FAI_TO_Pb
SOLVENT_DMF
Pri_VSA_EState4
Sec_HallKierAlpha
Sec_PEOE_VSA6
```

Shapes: (226, 13) train; (57, 13) test.

**Note:** The no-AddHs SelectFromModel notebook swaps `Sec_PEOE_VSA6` for `Sec_EState_VSA8`; all compositional terms are identical.

### Table S6. Thirty-one features — `add_hydrogens_tuned_rfecv.ipynb`

```
BR_PRIMARY_ORGANIC_HALIDE
PRIMARY_SPACER_FRACTION
SPACER_TO_PB_RATIO
PbCl2
PbBr2
CsBr_TO_Pb
CsI_TO_Pb
FABr_TO_Pb
FAI_TO_Pb
SOLVENT_DMF
SOLVENT_DMSO
Pri_qed
Pri_SPS
Pri_BCUT2D_MWHI
Pri_BCUT2D_MWLOW
Pri_BCUT2D_LOGPLOW
Pri_BCUT2D_MRHI
Pri_Kappa3
Pri_PEOE_VSA10
Pri_PEOE_VSA7
Pri_SMR_VSA1
Pri_VSA_EState4
Pri_VSA_EState7
Pri_VSA_EState8
Pri_NumHeteroatoms
Pri_NumRotatableBonds
Pri_Phi
Sec_BCUT2D_MWHI
Sec_HallKierAlpha
Sec_Kappa3
Sec_PEOE_VSA6
```

Shapes: (226, 31) train; (57, 31) test.

---

## S5. Emission Photon Energy — All Notebook Benchmarks

All values below are **held-out test** metrics (*n* = 57) transcribed from notebook stdout. Rounding in the main text (three decimal places) does not change ranking.

### Table S7. **PRIMARY** — AddHs, tuned, SelectFromModel (13 features)

| Rank | Model | *R*<sup>2</sup> | RMSE (eV) | MAE (eV) |
|------|-------|------|-----------|----------|
| 1 | **SVR (RBF)** | **0.952715** | **0.074335** | **0.047942** |
| 2 | Gaussian Process | 0.920381 | 0.096459 | 0.060961 |
| 3 | CatBoost | 0.874856 | 0.120931 | 0.062522 |
| 4 | XGBoost | 0.756337 | 0.168744 | 0.070986 |
| 5 | Random Forest | 0.747938 | 0.171628 | 0.083796 |

### Table S8. **PRIMARY** — AddHs, tuned, RFECV (31 features)

| Rank | Model | *R*<sup>2</sup> | RMSE (eV) | MAE (eV) |
|------|-------|------|-----------|----------|
| 1 | **SVR (RBF)** | **0.919559** | **0.096955** | **0.053890** |
| 2 | Gaussian Process | 0.876814 | 0.119981 | 0.072528 |
| 3 | CatBoost | 0.861979 | 0.127001 | 0.066114 |
| 4 | XGBoost | 0.746257 | 0.172199 | 0.072514 |
| 5 | Random Forest | 0.690493 | 0.190182 | 0.086730 |

### Table S9. Ablation — AddHs, **untuned**, RFECV (31 features)

Hyperparameter tuning matters: compare Table S8 with Table S9.

| Rank | Model | *R*<sup>2</sup> | RMSE (eV) | MAE (eV) |
|------|-------|------|-----------|----------|
| 1 | Gaussian Process | 0.876814 | 0.119981 | 0.072528 |
| 2 | CatBoost | 0.857190 | 0.129185 | 0.069607 |
| 3 | SVR (RBF) | 0.819187 | 0.145361 | 0.073809 |
| 4 | XGBoost | 0.809737 | 0.149111 | 0.068635 |
| 5 | Random Forest | 0.699186 | 0.187492 | 0.085629 |

### Table S10. Ablation — no AddHs, tuned, SelectFromModel (13 features)

| Rank | Model | *R*<sup>2</sup> | RMSE (eV) | MAE (eV) |
|------|-------|------|-----------|----------|
| 1 | SVR (RBF) | 0.943457 | 0.081287 | 0.053091 |
| 2 | Gaussian Process | 0.928067 | 0.091685 | 0.059866 |
| 3 | CatBoost | 0.878484 | 0.119165 | 0.063835 |
| 4 | XGBoost | 0.862872 | 0.126589 | 0.070007 |
| 5 | Random Forest | 0.728165 | 0.178232 | 0.090018 |

### Table S11. Ablation — no AddHs, tuned, RFECV (39 features)

| Rank | Model | *R*<sup>2</sup> | RMSE (eV) | MAE (eV) |
|------|-------|------|-----------|----------|
| 1 | SVR (RBF) | 0.923568 | 0.094509 | 0.054670 |
| 2 | Gaussian Process | 0.909226 | 0.102995 | 0.058032 |
| 3 | CatBoost | 0.889680 | 0.113543 | 0.059378 |
| 4 | XGBoost | 0.880244 | 0.118299 | 0.065432 |
| 5 | Random Forest | 0.751882 | 0.170279 | 0.083300 |

### Table S12. Ablation — no AddHs, untuned, RFECV (39 features)

| Rank | Model | *R*<sup>2</sup> | RMSE (eV) | MAE (eV) |
|------|-------|------|-----------|----------|
| 1 | Gaussian Process | 0.909226 | 0.102995 | 0.058032 |
| 2 | CatBoost | 0.895562 | 0.110475 | 0.058297 |
| 3 | SVR (RBF) | 0.881301 | 0.117776 | 0.062248 |
| 4 | XGBoost | 0.837960 | 0.137608 | 0.065738 |
| 5 | Random Forest | 0.736578 | 0.175452 | 0.083928 |

### Table S13. Head-to-head — best SVR only

| Notebook | Features | AddHs | Tuned | SVR *R*<sup>2</sup> | SVR RMSE (eV) |
|----------|----------|-------|-------|---------------|---------------|
| **add_hydrogens_tuned_select_from_model** | **13** | **Yes** | **Yes** | **0.953** | **0.074** |
| add_hydrogens_tuned_rfecv | 31 | Yes | Yes | 0.920 | 0.097 |
| tuned_select_from_model | 13 | No | Yes | 0.943 | 0.081 |
| tuned_rfecv | 39 | No | Yes | 0.924 | 0.095 |
| add_hydrogens_baseline_rfecv | 31 | Yes | No | 0.819 | 0.145 |

---

## S6. SHAP and Figure Index

| Figure ID | Description | Source |
|-----------|-------------|--------|
| Main Fig. 1 | Workflow schematic | Author |
| Main Fig. 2 | RFECV curve, 31 features (AddHs) | `add_hydrogens_tuned_rfecv` |
| Main Fig. 3 | Correlation heatmap, 13 features | `add_hydrogens_tuned_select_from_model` |
| Main Fig. 4 | Feature importance, 13 features | same |
| Main Fig. 5 | SHAP beeswarm, SVR, 13 features | same |
| Main Fig. 6 | Parity *E*<sub>ph</sub>, test set | Generate from model predictions |
| Main Fig. 7 | EQE vs *E*<sub>ph</sub> control schematic | Author |
| Main Fig. 8 | SHAP, CatBoost architecture (EQE) | `catboost_only` |
| Fig. S1 | λ histogram | Data export |
| Figs. S2–S8 | Wavelength parity, RFECV, SHAP | `emission_wavelength_nm/*` |

**SHAP (brief).** SHAP values partition each prediction as *f*(**x**) = *φ*<sub>0</sub> + Σ*φ*<sub>*j*</sub> (main text Table 2 and ref. [12]). The primary emission model uses **Kernel SHAP** on tuned SVR–RBF; tree models in auxiliary runs use Tree SHAP.

---

## S7. EQE Dataset and Models

The EQE sheet (`EQE_CLEANED`) contains **266** records with the same compositional columns as the emission set, plus HTL, ETL, additive, and coarse thermal fields. Even with twenty features from recursive feature elimination and full hyperparameter tuning, tree models barely exceed *R*<sup>2</sup> ≈ 0.5 on the test fold. The best result we obtained (**Table S15**) encodes HTL/ETL/additive as categorical variables in CatBoost (*R*<sup>2</sup> = 0.570), which supports the main-text argument that stack identity is partially informative but fabrication remains largely unobserved.

### Table S14. EQE dataset summary

| Statistic | Value |
|-----------|-------|
| Records | 266 |
| EQE range (%) | 0.002 – 30.84 |
| Unique HTL | 34 |
| Unique ETL | 13 |

### Table S15. RFE-20 tuned models — `rfe_20_features_tuned.ipynb`

| Model | *R*<sup>2</sup> | RMSE (% EQE) | MAE (% EQE) |
|-------|------|--------------|-------------|
| Tuned XGBoost | 0.512 | 5.01 | 3.77 |
| Tuned Random Forest | 0.478 | 5.18 | 3.88 |
| Tuned CatBoost | 0.443 | 5.35 | 3.75 |

### Table S16. CatBoost architecture model — `catboost_only.ipynb`

| Metric | Value |
|--------|-------|
| *R*<sup>2</sup> | 0.570 |
| RMSE (% EQE) | 4.42 |
| MAE (% EQE) | 3.25 |

Categorical inputs: `Additive`, `Additive ratio`, `HTL`, `ETL`.

### Table S17. Fabrication-related quantities rarely tabulated in literature

| Variable | Why it matters for EQE |
|----------|------------------------|
| Antisolvent drip rate / timing | Crystallization, pinholes |
| Annealing atmosphere (O₂, N₂) | Trap formation |
| Substrate cleaning / wetting | Coverage |
| Active-area definition | Absolute EQE value |
| Sweep direction / pre-bias | Reported efficiency |

---

## S8. Hyperparameters, Configs, and Reproduction

**Tuning (emission).** RandomizedSearchCV, 15 iterations, 5-fold CV, `random_state = 42`; search grids in `src/ml_quasit/regression_model_trainer.py`.

**RFECV / SelectFromModel.** Correlation threshold 0.90; RF selector `max_depth = 5`, `min_samples_leaf = 6`; RFECV `min_features_to_select = 5`; SelectFromModel `threshold = 'mean'`.

**Canonical YAML configs**

| Analysis | Config path |
|----------|-------------|
| 13 features, tuned | `configs/emission_photon_energy/add_hydrogens_tuned_select_from_model.yaml` |
| 31 features, tuned | `configs/emission_photon_energy/add_hydrogens_tuned_rfecv.yaml` |

```bash
conda activate ml-quasit
pip install -e .
python scripts/run_experiment.py --config configs/emission_photon_energy/add_hydrogens_tuned_select_from_model.yaml
```

---

## S10. SHAP Analysis — Tuned SVR–RBF (Thirteen Features, AddHs)

This table is the Supporting Information companion to **main-text Table 2**. It was generated from the canonical thirteen-descriptor list (SI Table S5) after the same correlation filter (173 columns dropped on the training fold), hyperparameter-tuned SVR–RBF (*C* = 50, *γ* = 0.1, *ε* = 0.01), and Kernel SHAP (50 background samples, 250 coalitions per record, *N* = 283). Test-set performance at these settings: *R*<sup>2</sup> = 0.953, RMSE = 0.074 eV, MAE = 0.048 eV (*n*<sub>test</sub> = 57). CSV export: `manuscript/shap_svr13_summary.csv`.

### Table S10. Mean absolute SHAP and feature–SHAP correlation

| Rank | Descriptor | ⟨\|SHAP\|⟩ (eV) | Share (%) | Mean SHAP (eV) | corr(*x*, SHAP) | Mechanism (summary) |
|------|------------|----------------|-----------|----------------|-----------------|---------------------|
| 1 | PbBr2 | 0.0743 | 21.4 | −0.0064 | +0.952 | Inorganic Br; wider-gap wells |
| 2 | BR_PRIMARY_ORGANIC_HALIDE | 0.0592 | 17.1 | +0.0041 | +0.961 | Br on primary spacer |
| 3 | FAI_TO_Pb | 0.0310 | 8.9 | +0.0029 | −0.978 | FA iodide; iodide-rich, redder |
| 4 | FABr_TO_Pb | 0.0276 | 8.0 | −0.0057 | −0.913 | FA bromide ratio |
| 5 | CsI_TO_Pb | 0.0272 | 7.8 | +0.0035 | −0.873 | Cs iodide supply |
| 6 | PbCl2 | 0.0267 | 7.7 | +0.0061 | +0.969 | Inorganic Cl; high-gap bias |
| 7 | Pri_VSA_EState4 | 0.0236 | 6.8 | +0.0032 | −0.792 | Primary spacer polarity/VSA |
| 8 | CsBr_TO_Pb | 0.0227 | 6.5 | +0.0003 | −0.325 | Cs bromide ratio |
| 9 | SOLVENT_DMF | 0.0170 | 4.9 | −0.0037 | −0.261 | DMF vs other solvents |
| 10 | SPACER_TO_PB_RATIO | 0.0128 | 3.7 | +0.0011 | +0.139 | Nominal *n* / confinement lever |
| 11 | Sec_HallKierAlpha | 0.0113 | 3.3 | +0.0011 | −0.340 | Secondary spacer topology |
| 12 | CL_PRIMARY_ORGANIC_HALIDE | 0.0079 | 2.3 | −0.0007 | +0.970 | Cl on primary spacer |
| 13 | Sec_PEOE_VSA6 | 0.0060 | 1.7 | −0.0010 | −0.504 | Secondary spacer charge/VSA |

**Reading the columns.** Positive corr(*x*, SHAP) means that, across the corpus, larger feature values associate with more positive SHAP contributions and therefore **higher** predicted *E*<sub>ph</sub> (bluer). Binary flags are interpreted as 0/1. SHAP base value *φ*<sub>0</sub> = **2.3269 eV**.

**Reproduction (Python).** After `pip install -e .`, fix the thirteen feature names from Table S5, run correlation filtering and SVR tuning as in `add_hydrogens_tuned_select_from_model.yaml`, then apply `shap.KernelExplainer(svr.predict, background)` on standardized `X_train`/`X_test`. The manuscript CSV was produced with this workflow on the repository data freeze.

---

## S9. Bibliography (Numbered as in Main Text; Full Titles for Cross-Referencing)

Search the title in your reference manager or publisher site if the author list is abbreviated in the main manuscript.

**[1]** L. N. Quan et al. **“Perovskites for Next-Generation Optical Sources.”** *Chem. Rev.* **2019**, *119*, 7444–7477.

**[2]** M. Yuan et al. **“Perovskite Energy Funnels for Efficient Light-Emitting Diodes.”** *Nat. Nanotechnol.* **2016**, *11*, 872–877.

**[3]** M. D. Smith et al. **“Mechanisms of Photoluminescence in Quasi-2D Lead-Halide Perovskites.”** *Chem. Rev.* **2019**, *119*, 3104–3139.

**[4]** D. B. Straus, C. R. Kagan. **“Electrons, Excitons, and Phonons in Two-Dimensional Hybrid Perovskites.”** *J. Phys. Chem. Lett.* **2018**, *9*, 1434–1447.

**[5]** Y. Mai et al. **“Machine Learning-Based Screening of Two-Dimensional Perovskite Organic Spacers.”** *Adv. Compos. Hybrid Mater.* **2024**, *7*, 104.

**[6]** F. Lu et al. **“Machine Learning for Perovskite Optoelectronics: A Review.”** *Adv. Photonics* **2024**, *6*, 054001.

**[7]** S. Wang et al. **“From Formability to Bandgap: Machine Learning Accelerates the Discovery and Application of Perovskite Materials.”** *ACS Nano* **2025**, *19*, 29049–29072.

**[8]** Z. Wang, C. Chen et al. **“Data-Driven Design of Halide Perovskites for Efficient Green Light-Emitting Diodes via Machine Learning and DFT.”** *J. Phys. Chem. C* **2025**, *129*, 13542–13557.

**[9]** W. Wang, Y. Li et al. **“Predicting the Photon Energy of Quasi-2D Lead Halide Perovskites from the Precursor Composition through Machine Learning.”** *Nanoscale Adv.* **2022**, *4*, 1632–1638.

**[10]** L. Zhang et al. **“Deep Learning for Additive Screening in Perovskite Light-Emitting Diodes.”** *Angew. Chem. Int. Ed.* **2022**, *61*, e202209337.

**[11]** K. Lin et al. **“Perovskite Light-Emitting Diodes with External Quantum Efficiency Exceeding 20 Per Cent.”** *Nature* **2018**, *562*, 245–248.

**[12]** S. M. Lundberg, S.-I. Lee. **“A Unified Approach to Interpreting Model Predictions.”** *Adv. Neural Inf. Process. Syst.* **2017**, *30*, 4765–4774.

---

*End of Supporting Information. Numerical values match executed notebook stdout in repository `ML_Quasi_2D_Emission_EQE`.*
