# Figure 1 — Import and finish in 10 minutes

File to open: **`Figure1_workflow.drawio`**

---

## Step 1: Open the diagram

1. Go to https://app.diagrams.net (or install draw.io Desktop).
2. **File → Open from → Device**
3. Select:
   ```
   ML_Quasi_2D_Emission_EQE/manuscript/Figure1_workflow.drawio
   ```
4. You should see **6 coloured boxes** in a row with arrows.

---

## Step 2: Add the Snowflake icon

1. Download the Snowflake logo (PNG, transparent background) from your Snowflake brand assets or a small official icon.
2. In draw.io: **Arrange → Insert → Image** (or drag PNG onto the canvas).
3. Place the image **on top of** the small dashed box in **Box 2** (top-left, where it says "ICON").
4. Resize to about **40×40 px**.
5. **Delete** the dashed placeholder box and the grey italic text *"[Insert Snowflake icon here...]"* inside Box 2:
   - Double-click Box 2 → edit text → remove those two lines.
6. Keep: "Output: SF_DATA Excel"

---

## Step 3: Optional tweaks

| If you want… | Do this |
|--------------|---------|
| Remove title from figure (caption only in Word) | Select the top title text → Delete |
| Remove legend | Select legend box → Delete |
| Wider for Word full page | Drag right edge of canvas; spread boxes evenly |
| Bigger text for print | Select all boxes → Font size 12 |

---

## Step 4: Export for Word

1. **File → Export as → PNG**
   - Zoom: **300%**
   - Border width: **10**
   - Transparent background: **unchecked**
2. Save as: `Figure1_workflow.png`
3. Also export **PDF** (vector) if the journal accepts PDF figures.

---

## Step 5: Insert in Word

1. Open `MAIN_Manuscript_1 (AutoRecovered).docx`
2. Place cursor where Figure 1 goes
3. **Insert → Pictures →** choose `Figure1_workflow.png`
4. Paste the caption from `MAIN_Manuscript.md` (Figure 1 paragraph) or use:

> **Figure 1.** Workflow for machine-learning prediction of quasi-2D PeLED photon energy from literature precursor tables. Literature device records (*N* = 283) were curated in Snowflake to generate structured precursor fields, including halide identity flags and stoichiometric ratios, before export to the analysis workbook. Spacer IUPAC names were converted to SMILES and featurized with hydrogen-complete RDKit two-dimensional descriptors (`Chem.AddHs`), yielding a 283 × 462 input matrix. After an 80/20 stratified split (226 training, 57 test), 173 highly correlated columns were removed using training data only (|*r*| > 0.90). Two feature-selection strategies were applied in parallel: RFECV (31 features) and SelectFromModel (13 features). Hyperparameter-tuned regressors were evaluated on the held-out test set; the best model was SVR–RBF on 13 features (*R*² = 0.953, RMSE = 0.074 eV), with Kernel SHAP used for mechanistic interpretation.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| File won't open | Rename extension to `.drawio` if needed; try draw.io Desktop |
| Boxes overlap | View → Zoom 50%; drag boxes apart |
| Arrows disconnected | Drag arrow endpoints onto box edges (green dots) |
| Snowflake logo blurry | Use SVG or PNG at least 200×200 px, then scale down in draw.io |

---

## Files in this folder

| File | Purpose |
|------|---------|
| `Figure1_workflow.drawio` | Editable diagram (import this) |
| `Figure1_IMPORT_INSTRUCTIONS.md` | This guide |
| `MAIN_Manuscript.md` | Full caption and Methods text |
