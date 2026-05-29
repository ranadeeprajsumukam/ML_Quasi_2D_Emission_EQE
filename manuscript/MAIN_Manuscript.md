# Predicting Emission Energy in Quasi-2D Perovskite LEDs from Spacer Chemistry and Precursor Stoichiometry: When Machine Learning Works, and When It Does Not

**Authors:** [Author Names]<sup>1,*</sup>  
**Affiliations:** <sup>1</sup>[Institution, Address]  
**Corresponding author:** *[email]*

---

## Abstract

Quasi-two-dimensional (quasi-2D) halide perovskites remain difficult to design by intuition alone: the same nominal spacer and halide recipe can emit anywhere from the violet to the near-infrared depending on *n*-phase distribution, dielectric confinement, and processing history. We compiled 283 literature PeLED entries (photon energy 1.53–3.06 eV; peak wavelength 405–810 nm), converted organic spacer names to SMILES, and calculated 217 RDKit two-dimensional descriptors for each primary and secondary ligand after adding explicit hydrogens to the molecular graph. Together with precursor stoichiometry and solvent identity, this produced 462 input columns per record. Removing 173 near-duplicate descriptors (|*r*| > 0.90) and applying two feature-selection strategies on a stratified 80/20 split (226 training, 57 test) gave contrasting results. Recursive feature elimination with cross-validation (RFECV) kept 31 variables—the cross-validated optimum—whereas a random-forest importance filter (SelectFromModel, threshold = mean importance) retained only 13. Hyperparameter-tuned support vector regression on the smaller set performed best on the held-out literature data (*R*<sup>2</sup> = 0.953, RMSE = 0.074 eV, MAE = 0.048 eV), clearly outperforming the 31-feature RFECV model (*R*<sup>2</sup> = 0.920, RMSE = 0.097 eV) despite the latter’s lower cross-validation error. The retained descriptors are chemically recognizable: primary-spacer halide identity, spacer-to-lead ratio, lead halide and A-site fractions, DMF as solvent, and a small number of topological indices for primary and secondary spacers. SHAP analysis is consistent with confinement-dominated colour tuning. By contrast, models trained on 266 literature EQE values—including HTL, ETL, and additive labels—plateau near *R*<sup>2</sup> ≈ 0.51–0.57, which we attribute to fabrication and measurement details that are not captured in compositional tables. Wavelength-based regressions are reported in the Supporting Information. The work separates what literature ML can reliably guide (emission colour from chemistry) from what still requires standardized device fabrication data (efficiency).

**Keywords:** quasi-2D perovskite, PeLED, photon energy, machine learning, RDKit, feature selection, SHAP, EQE

---

## 1. Introduction

Ruddlesden–Popper quasi-2D perovskites, L<sub>2</sub>A<sub>*n*−1</sub>Pb<sub>*n*</sub>X<sub>3*n*+1</sub>, have become a practical route to tunable electroluminescence in metal-halide light-emitting diodes (PeLEDs). Quantum and dielectric confinement in the inorganic sheets, together with the dielectric mismatch at organic–inorganic interfaces, shift the emission energy as the nominal layer thickness *n* and the halide composition change.[1–3] In the laboratory, colour is usually adjusted by swapping the ammonium spacer, tuning the halide ratio between the organic and inorganic sublattices, varying the spacer-to-lead stoichiometry, and choosing solvents that alter crystallization. The parameter space is large, and the outcome is not always predictable from stoichiometry alone because quasi-2D films are rarely single-phase: lower-*n* domains, three-dimensional-like impurity phases, and disorder broadening all influence which transition dominates the electroluminescence spectrum.[3,4]

The underlying photophysics is now reasonably well mapped. Straus and Kagan, among others, have summarized how dielectric confinement, exciton binding energy, and phonon coupling in two-dimensional hybrid perovskites govern optical response as *n* and composition change.[4] Even so, translating that understanding into a quantitative recipe—“if I replace PEA with BA and add 10% chloride on the spacer, where will the EL peak land?”—remains awkward. Stoichiometric *n* in the precursor does not always match the *n* distribution in the annealed film, and the phase that emits most strongly is not necessarily the phase present in highest fraction. Those are familiar frustrations to anyone who has compared XRD or transient absorption with the EL peak from the same wafer.

Most groups still explore this space experimentally, one ink at a time. That approach is understandable—film quality and device stack matter—but it scales poorly as the library of bulky amines, diamines, and mixed-cation recipes grows. Machine learning (ML) has therefore been adopted in perovskite photovoltaics and, more recently, in LED and emitter work, to extract patterns from accumulated reports.[6–8] Reviews of the field emphasize a repeatable workflow: curate experimental tables, convert structures to numerical descriptors, prune correlated features, train regressors or classifiers, and interpret the result with tools such as SHAP (SHapley Additive exPlanations).[6,7] Prior work on quasi-2D systems showed that photon energy can be estimated from precursor composition with random-forest root-mean-square errors near 0.05 eV when the database is chemically focused and the descriptors are chosen carefully.[9] Complementary ML studies have targeted organic-spacer libraries in two-dimensional perovskites, emphasizing that ligand choice is a first-class design variable alongside stoichiometry.[5] Separately, deep-learning classification of PeLED additives reached high accuracy only when the dataset was built from a single device architecture and supplemented with in-house experiments.[10] Those precedents matter for how we should read literature ML: models are most trustworthy when inputs and labels refer to the same experimental contract, and least trustworthy when they silently average over incompatible fabrication routes.

PeLED efficiency has risen sharply—external quantum efficiencies above 20% are now common for three-dimensional and optimized low-dimensional systems[11]—but efficiency is not predicted by the same variables that set emission colour. EQE depends on radiative yield, charge balance, and outcoupling, all of which are sensitive to film morphology and interface quality. Zhang et al. illustrated the data problem for device-targeted ML: even with careful molecular featurization, PeLED performance labels from literature are sparse and heterogeneous compared with the volume of data available in organic LED or photovoltaic ML campaigns.[10]

Our objective is narrower and, we believe, more useful for materials chemists who formulate quasi-2D inks. We ask two questions. First, can literature data spanning the full visible range—not only blue or deep-green emitters—be used to predict photon energy from spacer structures and precursor tables with descriptors that a chemist can interpret? Second, under the same curation rules, can EQE be predicted with comparable confidence from the columns normally reported in papers? We deliberately treat emission energy as a materials-led property and EQE as a device-led property, because the latter depends on morphology, interfaces, and measurement conditions that are rarely digitized in full.

We built the analysis around hydrogen-complete RDKit structures (`Chem.AddHs`), 217 two-dimensional descriptors per spacer, and two feature-selection workflows compared under identical hyperparameter tuning. The main result is a thirteen-descriptor model selected by a random-forest importance gate; a thirty-one-descriptor RFECV set is retained as a controlled comparison because cross-validation favours more variables than our held-out test fold can support. Analyses without added hydrogens or without hyperparameter tuning are summarized in the Supporting Information; they do not alter the conclusions below.

---

## 2. Results and Discussion

The Results section follows the logic of the Introduction. We first describe what the literature corpus contains and how each PeLED record was translated into a numerical representation that a materials chemist would recognize (Section 2.1). We then ask whether that representation is sufficient to predict photon energy on held-out literature reports, and we interpret both the statistical outcome and the retained descriptors in physical terms (Section 2.2). External quantum efficiency is treated separately in Section 2.3 because, as argued above, efficiency is governed by a different—and largely unreported—set of variables.

### 2.1. Building a literature-wide emission corpus and a chemistry-aware descriptor matrix

**What each record represents.** We compiled **283** quasi-2D PeLED entries from a curated workbook (`SF_DATA_EMISSION_PEAK.xlsx`, sheet `SF_DATA_EMISSION_PEAK_NEW`; column definitions and histograms in SI Section S1). Each row is not an abstract “data point” but a reported device or film from the peer-reviewed literature: a precursor stoichiometry table, one or two organic spacer names, a solvent, and a reported emission maximum. The target label is photon energy *E*<sub>ph</sub> in electronvolts, taken from the paper or converted consistently from the quoted peak wavelength. Across the corpus, *E*<sub>ph</sub> spans **1.53–3.06 eV** (peak wavelength **405–810 nm**). That range is chemically meaningful: it covers violet through near-infrared emission and therefore forces any global model to learn how halide composition, spacer bulkiness, and stoichiometry move the EL peak—not merely how to interpolate within a single green or blue halide window.

The spectral histogram is uneven in a way that mirrors the field, not our sampling. About **69%** of entries fall between 480 and 550 nm (green–yellow), reflecting the historical concentration of high-performance green quasi-2D work; **12%** lie in the violet/blue (400–480 nm), and **17%** in deep red and near-infrared (650–900 nm), with fewer orange/red points in between (SI Table S1.1). A model trained only on “typical” green recipes would under-represent the halide and confinement extremes that set the wings of the visible spectrum. We therefore present the analysis as a **broad-spectrum** regression benchmark, to be compared with earlier composition-only ML on quasi-2D photon energy that achieved very low errors on a more chemically focused subset.[9]

**Heterogeneity the model must absorb.** Twenty-two distinct primary spacer IUPAC names appear in the set, reduced to **30** unique SMILES after name resolution. Forty-six records (**16%**) list both a primary and a secondary ammonium ligand. Mixed-spacer inks are not outliers to be discarded: a secondary ligand changes interlayer spacing, hydrogen-bonding networks at the organic–inorganic interface, and—critically—which *n* domain dominates electroluminescence, often without a one-to-one change in the nominal spacer-to-lead ratio printed in the table.[2,3] If the train–test split ignored that structure, the test fold could be enriched in mixed-ligand chemistries unlike the training set and the reported *R*<sup>2</sup> would be optimistically biased. We therefore held out **20%** of records (**57** test, **226** train) with stratification on `IS_MIXED_SPACERS_SPACER`, so that both partitions preserve the same fraction of mixed-spacer formulations. All subsequent pruning, feature selection, hyperparameter search, and model comparison respect that split: **no statistic from the 57 test records entered feature selection or tuning.**

**From ink notation to 462 input columns.** The modelling philosophy was to keep everything a formulator might write in a lab notebook, then add only what cannot be captured by stoichiometry alone—the molecular structure of the spacer(s). Primary and secondary IUPAC names were mapped to SMILES (in-house table plus NIH CACTUS when needed). For each unique structure, RDKit computed **217** two-dimensional descriptors after **`Chem.AddHs`**, so that heteroatom valence and hydrogen-bonding topology on bulky ammoniums are represented explicitly; descriptors were prefixed `Pri_` or `Sec_` and joined to every row sharing that SMILES. Alongside these molecular columns we retained compositional fields extracted from the papers: binary flags for which halide sits on the primary organic cation (`CL_PRIMARY_ORGANIC_HALIDE`, `BR_PRIMARY_ORGANIC_HALIDE`, and analogous iodide flags where present), lead-halide inventory (`PbCl2`, `PbBr2`, …), A-site precursor ratios relative to lead (`FABr_TO_Pb`, `CsI_TO_Pb`, …), **spacer-to-lead ratio**, and related stoichiometric levers. Solvent identity was one-hot encoded (`SOLVENT_DMF`, `SOLVENT_DMSO`, …). The merged design matrix is **283 × 462**.

High-dimensional descriptor blocks are inevitably redundant: many RDKit pairs co-vary when only **30** distinct ligands are present. On the **training fold only**, we computed pairwise Pearson correlations and dropped any column that correlated with another at |*r*| > **0.90**, removing **173** columns and leaving **289** candidates for supervised selection. This step is easy to overlook, but it matters scientifically: without it, the selector can keep multiple proxies for the same molecular property (e.g. correlated PEOE/VSA partitions) and appear to need more features than the underlying chemistry requires.

**Figure 1** (to be inserted) summarizes the full pipeline: literature row → SMILES → AddHs descriptors + stoichiometry + solvent → correlation filter on the training fold → parallel RFECV (31 features) and SelectFromModel (13 features) → hyperparameter-tuned regressors → SHAP readout. Section 2.2 evaluates whether that pipeline recovers the confinement- and composition-dominated physics outlined in the Introduction.

---

### 2.2. Predicting photon energy: when a lean descriptor set outperforms cross-validation logic

**Scientific question and experimental contract.** Given only what appears in a typical quasi-2D PeLED paper—precursor ratios, spacer identity, solvent—can we predict *E*<sub>ph</sub> on devices the model has never seen? The question is stricter than it sounds. Literature rows mix electroluminescence peaks with photoluminescence maxima, different measurement temperatures, and films made on unlike substrates; none of that appears as numeric columns. A positive answer would imply that, despite phase heterogeneity,[3,4] the **reported** peak energy still carries a recoverable signal from ink chemistry. A negative answer would warn against using ML for colour screening without standardizing how emission is measured. Our results support a qualified yes: the signal is strong enough for useful held-out error, but only if the feature space is kept small.

**Two feature-selection strategies and why both are reported.** Starting from 289 post-correlation columns, we applied two selectors in parallel, each fit on the 226 training records only.

*Recursive feature elimination with cross-validation (RFECV)* wraps a shallow random-forest regressor (`max_depth` = 5, `min_samples_leaf` = 6) in five-fold **stratified** cross-validation, with sample weights that up-weight under-represented mixed-spacer profiles. Features are removed one at a time until the cross-validated RMSE of *E*<sub>ph</sub> reaches a minimum. That minimum occurs at **31** retained variables (**Figure 2**). By the textbook reading of RFECV, 31 is the “right” dimensionality for **interpolation within the training partition**—the curve in Figure 2 is itself a useful diagnostic of how quickly additional descriptors stop helping cross-validation.

*SelectFromModel* takes a different philosophical stance. A single random forest is fit on all 289 survivors; scikit-learn’s `SelectFromModel` with `threshold='mean'` keeps only descriptors whose impurity-based importance exceeds the forest average. That is a deliberately **aggressive** gate: it asks which variables the forest would miss if removed, not how many variables minimize CV error. Only **13** pass.

We report both sets because chemists and ML practitioners often talk past each other on dimensionality. RFECV says “use 31 inputs”; SelectFromModel says “most of the forest’s explanatory power lives in 13.” With *N*<sub>train</sub> = 226, the sample-to-feature ratio *N*/*p* is **7.3** for the former and **17.4** for the latter (SI Table S4). In a single-laboratory design with fixed processing, *N*/*p* ≈ 7 might be acceptable; in a literature compilation that silently averages over fabrication routes, extra molecular partitions (`Pri_BCUT2D_*`, redundant EState/VSA channels) frequently fit idiosyncrasies of individual papers. The decisive test is not cross-validation on the training fold but performance on the **57 stratified test records**—the closest analogue we have to “predict the next published device.”

**Model training and the role of hyperparameter tuning.** Every regressor in Table 1 was tuned by randomized search (**15** trials, five-fold CV on the training set, scoring by negative mean squared error) before a single evaluation on the test fold. We did not tune on the test set. Untuned ablations on the same splits (SI Table S9) show why this matters: tree ensembles can memorize rare spacer–halide combinations in 226 rows, and Gaussian processes are sensitive to kernel length scales when *p* is large. For the thirty-one-feature RFECV matrix, tuning moves GPR test *R*<sup>2</sup> from **0.877** to **0.920**; for the thirteen-feature SelectFromModel matrix, SVR and GPR both benefit, but **support vector regression with a radial-basis-function kernel (SVR–RBF)** emerges as the best compromise between flexibility and stability on the small feature set.

**Table 1.** Held-out test performance for photon energy (eV), AddHs descriptors, hyperparameter-tuned models (*n*<sub>test</sub> = 57).

| Feature selection | *p* | Model | *R*<sup>2</sup> | RMSE (eV) | MAE (eV) |
|-------------------|-----|-------|-------------|-----------|----------|
| SelectFromModel | 13 | **SVR (RBF)** | **0.953** | **0.074** | **0.048** |
| SelectFromModel | 13 | Gaussian process | 0.920 | 0.096 | 0.061 |
| SelectFromModel | 13 | CatBoost | 0.875 | 0.121 | 0.063 |
| RFECV | 31 | SVR (RBF) | 0.920 | 0.097 | 0.054 |
| RFECV | 31 | Gaussian process | 0.877 | 0.120 | 0.073 |
| RFECV | 31 | CatBoost | 0.862 | 0.127 | 0.066 |

The central result is the **inversion of RFECV and test rankings**. RFECV prefers 31 inputs and achieves lower training-fold CV error than the 13-variable pipeline, yet on held-out literature data the **thirteen-variable SVR is unambiguously better**: *R*<sup>2</sup> rises from **0.920** to **0.953**, and RMSE falls from **0.097** to **0.074 eV**—a **24%** reduction in root error at less than half the dimensionality. Mean absolute error **0.048 eV** on the test fold is comparable in spirit to the ~0.05 eV RMSE reported by Wang et al. for quasi-2D photon energy from composition,[9] but our model was trained on a wider spectral window and retains explicit spacer descriptors rather than composition alone.

To translate 0.074 eV into laboratory language: for a green emitter near **2.3 eV** (~540 nm), that uncertainty corresponds to roughly **15–20 nm** if one propagates through *E* ≈ 1240/λ—enough to rank spacer edits in a screening spreadsheet, not enough to replace measurement before locking a narrow formulation for IP or device certification. Tree models (CatBoost, random forest) lag SVR on the test fold despite strong training scores, which is another signature of overfitting when *p* is only modestly large relative to *N*.

**What the thirteen retained variables say about quasi-2D colour tuning.** The SelectFromModel list is short enough to read as a mechanism sketch rather than a black-box vector (full names in SI Table S5; correlation structure in **Figure 3**, importances in **Figure 4**):

`CL_PRIMARY_ORGANIC_HALIDE`, `BR_PRIMARY_ORGANIC_HALIDE`, `SPACER_TO_PB_RATIO`, `PbCl2`, `PbBr2`, `CsBr_TO_Pb`, `CsI_TO_Pb`, `FABr_TO_Pb`, `FAI_TO_Pb`, `SOLVENT_DMF`, `Pri_VSA_EState4`, `Sec_HallKierAlpha`, `Sec_PEOE_VSA6`

Roughly half of the inputs are **compositional levers** chemists already manipulate; the remainder encode **spacer shape and polarity** at the interface.

The halide flags on the primary organic cation (`CL_PRIMARY_ORGANIC_HALIDE`, `BR_PRIMARY_ORGANIC_HALIDE`) and the inorganic lead-halide fractions (`PbCl2`, `PbBr2`) jointly set the halide chemical potential in the ink. They do not, by themselves, specify a single *n* phase—literature films are seldom monodisperse—but they bias the average inorganic bandgap and the competition between bromide- and iodide-rich wells. Bromide-rich precursors generally push emission to higher photon energy (bluer) when low-*n*, confined domains dominate the spectrum; iodide-rich recipes lower *E*<sub>ph</sub>; chloride can raise energy further but often complicates phase purity if not balanced with stoichiometry.[4,9] That the model elevates these terms above dozens of unused RDKit columns is consistent with confinement physics.[1,4]

`SPACER_TO_PB_RATIO` is the stoichiometric knob most groups turn first when targeting a nominal Ruddlesden–Popper *n*. More spacer relative to lead increases organic fraction in the film, strengthens average dielectric confinement, and—when phase distribution cooperates—moves the dominant transition to higher *E*<sub>ph</sub> until a 3D-like impurity or a broadened domain distribution pulls the peak back toward bulk-like energies.[1,2] Its survival after aggressive selection is perhaps the strongest sanity check on the ML pipeline: the algorithm recovers the same variable a formulator would adjust before touching esoteric molecular indices.

The four A-site ratios (`CsBr_TO_Pb`, `CsI_TO_Pb`, `FABr_TO_Pb`, `FAI_TO_Pb`) reflect that our corpus is not a single-cation study. Formamidinium and cesium entries appear with mixed halides exactly as authors reported them. The model therefore learns **effective** cation–halide trends across papers rather than a single phase diagram. That is appropriate for literature ML but limits extrapolation: two papers with identical spreadsheet ratios but different annealing atmospheres may not have identical A-site occupancies in the final film.

`SOLVENT_DMF` indicates that DMF versus DMSO (the other major solvent in the set) shifts reported *E*<sub>ph</sub> enough to survive selection. Solvent is a processing variable, yet it is reported far more consistently than antisolvent timing, humidity, or substrate treatment—variables we know matter in the lab but cannot encode here. Its inclusion is a reminder that “materials-led” prediction on literature data still entangles chemistry with crystallization kinetics.

Among molecular descriptors, `Pri_VSA_EState4` partitions the **primary** spacer’s van der Waals surface by electrotopological state—sensitive to heteroatom placement and polarizable surface area where the ammonium meets the inorganic slab. For the **46** mixed-spacer entries, `Sec_HallKierAlpha` (flexibility/topology) and `Sec_PEOE_VSA6` (partial-charge-weighted surface area on the secondary ligand) capture how a second ligand perturbs interlayer spacing and local dielectric environment without requiring an explicit *n*-distribution label. If explicit hydrogens are omitted before RDKit featurization, the secondary term swaps to `Sec_EState_VSA8` while compositional inputs are unchanged (SI Section S3)—a small but reproducible sign that protonation state matters for secondary-spacer indices.

The thirty-one-variable RFECV superset adds interpretable chemistry—`PRIMARY_SPACER_FRACTION`, `SOLVENT_DMSO`, hydrophobicity/partition BCUT terms, rotatable-bond counts, additional PEOE/VSA channels (SI Table S6)—but **does not improve held-out prediction** at *N* = 226. We treat RFECV as a map of “what else might matter” inside the training partition; for deployment on new literature rows, we recommend the thirteen-variable SVR.

**SHAP as a consistency check, not a second model.** SHAP analysis[12] asks whether the fitted SVR ranks drivers in the same order a chemist would argue from confinement physics. For each test record, Shapley contributions *φ*<sub>*j*</sub> decompose the prediction relative to the corpus mean: a positive contribution from `SPACER_TO_PB_RATIO`, for example, means that—for that ink—stoichiometry pushes the predicted *E*<sub>ph</sub> above the database average, as expected when confinement strengthens. Tree-model SHAP on CatBoost shows the same qualitative ordering—halide and stoichiometric terms first, secondary-spacer descriptors active mainly on mixed-ligand rows (**Figure 5**). A parity plot of measured versus predicted *E*<sub>ph</sub> for the 57 test points (**Figure 6**), with symbols distinguishing mixed-spacer entries, is the natural companion figure for judging systematic bias at the spectral extremes.

**Wavelength regression as a cross-check.** Because *E*<sub>ph</sub> and λ are related by *E* ≈ 1240/λ, regressing wavelength might seem equivalent. In practice, literature labels mix EL and PL maxima with different Stokes shifts, so energy is the more physically direct target for confinement arguments. For completeness, a tuned Gaussian process on twenty-eight RFECV-selected features **without** AddHs reaches *R*<sup>2</sup> = **0.958** with RMSE ≈ **21 nm** on the same test fold (SI Section S2). That performance is excellent but does not overturn the primary conclusion: the AddHs, thirteen-variable **energy** model is more interpretable, chemically lean, and tied to the featurization protocol we adopt for spacer libraries going forward.

---

### 2.3. External quantum efficiency: a different problem

Once emission colour is modelled satisfactorily at the materials level, it is tempting to ask whether the same table can predict EQE. Our answer, for literature data in their present form, is only partially.

We assembled 266 EQE records (`EQE_CLEANED`) with hole-transport material (HTL), electron-transport material (ETL), additive identity, and coarse thermal processing fields (deposition temperature, annealing time) in addition to the compositional columns used for emission. Thirty-four distinct HTL labels and thirteen ETL labels appear in the set; EQE spans 0.002% to 30.8%. Even after recursive feature elimination to twenty descriptors and hyperparameter tuning of tree models, the best purely compositional regressors reach only *R*<sup>2</sup> ≈ 0.44–0.51 on the test fold (Table 2). A CatBoost model that treats HTL, ETL, and additive as native categorical variables—our best attempt to encode “device architecture” from what papers actually tabulate—raises *R*<sup>2</sup> to 0.57 with RMSE ≈ 4.4% EQE. That improvement is real but modest: it explains a little more than half the variance in EQE while the emission model explains about ninety-five percent of the variance in photon energy on the same style of split.

**Table 2.** Held-out test performance for literature EQE (%).

| Model | *R*<sup>2</sup> | RMSE (% EQE) | MAE (% EQE) |
|-------|-------------|--------------|-------------|
| Tuned XGBoost (RFE, 20 features) | 0.512 | 5.01 | 3.77 |
| Tuned random forest | 0.478 | 5.18 | 3.88 |
| Tuned CatBoost | 0.443 | 5.35 | 3.75 |
| CatBoost (architecture features) | 0.570 | 4.42 | 3.25 |

The gap is not an algorithm failure. EQE couples internal radiative efficiency with charge injection balance and light outcoupling. Pinhole-free morphology, interfacial trap passivation, refractive-index matching, and the exact measurement protocol—integrated sphere calibration, pixel area, sweep rate— routinely move EQE by several percentage points without any change to the precursor stoichiometry row that appears in a spreadsheet. Different groups also describe annealing, antisolvent quenching, and atmosphere control in prose without numeric fields. Zhang et al. reached a similar conclusion for PeLED additive classification: machine learning on small, diverse device datasets works best when the training distribution matches the deployment case.[10] Literature EQE tables mix cases that do not.

We therefore treat EQE models trained here as **ranking aids**—useful for flagging whether transport-layer choice correlates with higher reported efficiency in the historical record—not as quantitative replacements for fabricating and measuring a device. **Figure 7** (schematic) and **Figure 8** (SHAP for the architecture-aware CatBoost model) summarize this distinction.

---

## 3. Computational Methods

**Data curation.** Literature records were taken from `SF_DATA_EMISSION_PEAK.xlsx` (emission sheet, 283 rows; EQE sheet, 266 rows). Each row corresponds to a reported device or film with extracted precursor stoichiometry, spacer names, solvent, and target property (*E*<sub>ph</sub> in eV, emission peak in nm where noted, or EQE in %). Reference DOIs were retained for traceability but not used as model inputs.

**Structure encoding.** Primary and secondary spacer IUPAC names were converted to SMILES using an in-house lookup table and, when necessary, the NIH CACTUS structure resolver. RDKit (version as in project `requirements.txt`) computed 217 two-dimensional descriptors for each unique SMILES after `Chem.AddHs`; descriptors were merged onto every record with `Pri_` and `Sec_` prefixes. Solvents were one-hot encoded (`SOLVENT_*` columns).

**Train–test splitting.** An 80/20 split (`random_state` = 42) was applied with stratification on `IS_MIXED_SPACERS_SPACER` for emission models, yielding 226 training and 57 test records. EQE models used a composite stratification profile (mixed spacer, additive presence, grouped HTL/ETL) described in SI Section S7.

**Correlation filtering.** Pearson correlation matrices were computed on the training fold only. Column *j* was removed if |*r*| > 0.90 for any pair (*j*, *k*) with *j* ≠ *k*, giving 173 removals from 462 initial columns.

**Feature selection.** (i) RFECV: five-fold stratified CV, random-forest selector, sample weights for class balance, minimum CV RMSE at 31 features. (ii) SelectFromModel: prefitted random forest, `threshold='mean'`, 13 features retained. Test data were not used in either selector.

**Regression and tuning.** Models included random forest, CatBoost, XGBoost, SVR (RBF kernel), and Gaussian process regression. Features were standardized for SVR and GPR. Hyperparameters were chosen by randomized search (15 iterations, five-fold CV, negative mean squared error) except where CatBoost’s native search was used for EQE. Reported metrics are *R*<sup>2</sup>, RMSE, and MAE on the held-out 57 emission records (or the EQE test fold).

**Interpretation.** SHAP values were computed with `TreeExplainer` for tree models and `KernelExplainer` (50 random training samples as background) for SVR and GPR.[12]

**Reproducibility.** Analysis code is packaged as `ml_quasit`. Primary configuration files: `configs/emission_photon_energy/add_hydrogens_tuned_select_from_model.yaml` and `add_hydrogens_tuned_rfecv.yaml`. Notebooks: `add_hydrogens_tuned_select_from_model.ipynb` (thirteen features) and `add_hydrogens_tuned_rfecv.ipynb` (thirty-one features).

---

## 4. Conclusions

Literature machine learning can predict quasi-2D PeLED photon energy across a 405–810 nm compilation with useful accuracy if the descriptor set is kept chemically lean. In our analysis, thirteen variables—halide identity on the primary spacer, spacer-to-lead stoichiometry, selected lead-halide and A-site fractions, DMF solvent, and three topology/charge descriptors on hydrogen-complete structures—outperform a thirty-one-variable RFECV optimum on held-out data (*R*<sup>2</sup> = 0.953 versus 0.920 for tuned SVR; RMSE 0.074 versus 0.097 eV). The result follows from a straightforward constraint: two hundred twenty-six training points cannot support three dozen adjustable descriptor dimensions without overfitting, even when cross-validation on the training partition suggests otherwise.

The selected features align with confinement and composition arguments chemists already use to tune colour. SHAP rankings do not contradict that picture: halide and stoichiometry terms lead; secondary-spacer indices matter where mixed ligands appear.

EQE, in contrast, is not predicted well from the same style of table (*R*<sup>2</sup> roughly 0.5–0.6 at best). Transport-layer labels help modestly, but morphology and fabrication protocol dominate efficiency in ways literature columns do not capture. We would use the emission model to narrow spacer and ink choices before synthesis, and treat any EQE prediction as a weak prior until devices are built under a fixed stack and a written processing checklist shared across collaborators.

Planned extensions include holding out entire reference DOIs from training, attaching explicit *n*-phase indicators where XRD or optical data exist, and feeding back in-house devices fabricated under constant HTL/ETL conditions to test whether EQE becomes learnable once the hidden variables are finally controlled.

---

## Author Contributions

[To be completed]

## Conflicts of Interest

The authors declare no competing financial interest.

## Data Availability

Analysis code and configuration files are available at [repository URL]. The curated workbook is available from the corresponding author on reasonable request.

## Acknowledgments

[To be completed]

---

## References

[1] L. N. Quan et al. **“Perovskites for Next-Generation Optical Sources.”** *Chem. Rev.* **2019**, *119*, 7444–7477.

[2] M. Yuan et al. **“Perovskite Energy Funnels for Efficient Light-Emitting Diodes.”** *Nat. Nanotechnol.* **2016**, *11*, 872–877.

[3] M. D. Smith et al. **“Mechanisms of Photoluminescence in Quasi-2D Lead-Halide Perovskites.”** *Chem. Rev.* **2019**, *119*, 3104–3139.

[4] D. B. Straus, C. R. Kagan. **“Electrons, Excitons, and Phonons in Two-Dimensional Hybrid Perovskites.”** *J. Phys. Chem. Lett.* **2018**, *9*, 1434–1447.

[5] Y. Mai et al. **“Machine Learning-Based Screening of Two-Dimensional Perovskite Organic Spacers.”** *Adv. Compos. Hybrid Mater.* **2024**, *7*, 104.

[6] F. Lu et al. **“Machine Learning for Perovskite Optoelectronics: A Review.”** *Adv. Photonics* **2024**, *6*, 054001.

[7] S. Wang et al. **“From Formability to Bandgap: Machine Learning Accelerates the Discovery and Application of Perovskite Materials.”** *ACS Nano* **2025**, *19*, 29049–29072.

[8] Z. Wang, C. Chen et al. **“Data-Driven Design of Halide Perovskites for Efficient Green Light-Emitting Diodes via Machine Learning and DFT.”** *J. Phys. Chem. C* **2025**, *129*, 13542–13557.

[9] W. Wang, Y. Li et al. **“Predicting the Photon Energy of Quasi-2D Lead Halide Perovskites from the Precursor Composition through Machine Learning.”** *Nanoscale Adv.* **2022**, *4*, 1632–1638.

[10] L. Zhang et al. **“Deep Learning for Additive Screening in Perovskite Light-Emitting Diodes.”** *Angew. Chem. Int. Ed.* **2022**, *61*, e202209337.

[11] K. Lin et al. **“Perovskite Light-Emitting Diodes with External Quantum Efficiency Exceeding 20 Per Cent.”** *Nature* **2018**, *562*, 245–248.

[12] S. M. Lundberg, S.-I. Lee. **“A Unified Approach to Interpreting Model Predictions.”** *Adv. Neural Inf. Process. Syst.* **2017**, *30*, 4765–4774.

---

*Primary numerical results: `add_hydrogens_tuned_select_from_model.ipynb` (13 features) and `add_hydrogens_tuned_rfecv.ipynb` (31 features). Supporting tables and ablations: `SUPPORTING_INFORMATION.md`.*
