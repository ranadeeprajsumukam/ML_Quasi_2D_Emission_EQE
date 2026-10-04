# Cover Letter — JACS Communications Submission

---

**Date:** October 2026

**To:**
The Editorial Board,
Journal of the American Chemical Society (JACS),
American Chemical Society

---

**Re: Submission of Manuscript — "Cheminformatics and Machine Learning for Quasi-2D Perovskites: Emission Energy Prediction and the Limits of EQE Modelling"**

---

Dear Editor,

We are pleased to submit the above manuscript for consideration as a **JACS Communication**.

Quasi-two-dimensional metal halide perovskites (quasi-2D MHPs) have emerged as highly promising materials for next-generation light-emitting diodes, owing to their widely tunable emission across the visible spectrum, high colour purity, and improved environmental stability relative to their three-dimensional counterparts. However, the vast combinatorial parameter space — spanning organic spacer chemistry, halide identity, A-site composition, and processing conditions — makes systematic experimental optimisation extremely time-consuming and resource-intensive. Machine learning offers a compelling route to navigate this space intelligently, yet prior modelling efforts have been confined to narrow spectral windows using only precursor ion ratios, without explicitly encoding the molecular structure of the organic spacer ligand.

Our work addresses this gap in three ways that we believe are novel and significant:

**1. A broadly applicable, cheminformatics-driven emission prediction model.**
We curated a dataset of 283 records spanning the full visible spectrum (405–810 nm), covering diverse halide systems (Br/I/Cl and mixed), single and mixed organic spacers, and multiple A-site cation combinations. By encoding organic spacer topology explicitly through RDKit two-dimensional molecular descriptors, our champion 13-feature Support Vector Regression (SVR–RBF) model achieves R² = 0.953 and RMSE = 0.074 eV on the held-out test set. Ten-fold cross-validation (R² = 0.931 ± 0.052) and bootstrap 95% confidence intervals (RMSE: 0.053–0.091 eV) confirm the robustness of this result. A learning curve analysis further demonstrates that the model is well-converged and that the dataset size is sufficient for this feature space.

**2. A quantitative and honest analysis of the limits of EQE prediction.**
We demonstrate that external quantum efficiency, unlike emission energy, is fundamentally unlearnable from heterogeneous published literature data. EQE depends critically on film morphology, interfacial trap density, and laboratory-specific fabrication protocols that are not captured in standard reported variables. Our CatBoost model, which incorporates transport-layer identities as native categorical inputs, achieves only R² ≈ 0.57 — a ceiling that is not a failure of the algorithm but a reflection of the data generation process itself. We believe this is an important and underappreciated message for the materials informatics community.

**3. A publicly accessible, zero-installation web application.**
The champion SVR model is deployed as a real-time, interactive web application on Streamlit Cloud (https://mlquasi2demissioneqe-8yvtmbep6kul3izkcuafzs.streamlit.app/). Researchers can input organic spacer structures by IUPAC name or SMILES, specify precursor ratios, and receive an instant emission energy and wavelength prediction — no programming knowledge required. To our knowledge, this is the first such deployed tool for quasi-2D perovskite emission prediction in the published literature.

We believe this manuscript is well-suited for JACS Communications given the broad interest in perovskite optoelectronics, the increasing importance of machine learning in materials discovery, and the practical value of the deployed prediction tool to the community. The manuscript has not been published elsewhere, is not under consideration at any other journal, and all authors have approved the final version.

The complete dataset, model bundle, Jupyter notebooks, and application source code are publicly available at: https://github.com/ranadeeprajsumukam/ML_Quasi_2D_Emission_EQE.

There are no conflicts of interest to declare.

We would be happy to provide any additional information or data the editorial team may require.

Sincerely,

**Ranadeep Raj Sumukam**
School of Chemistry, University of Hyderabad
Hyderabad, Telangana, India

*(On behalf of all co-authors: Ramu Naidu Savu, Santosh Rompivalasa, and Prof. Murali Banavoth)*

---
> **Corresponding Author:**
> Prof. Murali Banavoth
> School of Chemistry, University of Hyderabad
> Hyderabad, Telangana 500046, India
> [Add email address here]
