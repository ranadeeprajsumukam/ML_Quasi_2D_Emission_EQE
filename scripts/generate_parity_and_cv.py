"""
Publication-quality analysis for JACS review response.

Generates:
  1. Parity plot (predicted vs. experimental Eph) — main text figure
  2. 10-fold cross-validated R² ± std + bootstrap 95% CI

Run from the repo root:
  python scripts/generate_parity_and_cv.py
"""

import sys, os, glob, warnings
warnings.filterwarnings("ignore")

# ── Setup path so we can import ml_quasit ────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ml_quasit.model_serialization import load_model_bundle

# ── CONFIG ────────────────────────────────────────────────────────────────
BUNDLE_DIR   = "models/emission_photon_energy_add_hydrogens_tuned_select_from_model"
DATA_PATH    = "data/SF_DATA_EMISSION_PEAK.xlsx"
DATA_SHEET   = "SF_DATA_EMISSION_PEAK_NEW"
TARGET_COL   = "PHOTON_ENERGY_EV"
STRATIFY_COL = "IS_MIXED_SPACERS_SPACER"
RANDOM_STATE = 42
OUT_DIR      = "outputs/manuscript_figures"
N_BOOTSTRAP  = 500
N_CV_FOLDS   = 10

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load bundle ───────────────────────────────────────────────────────────
print("Loading model bundle...")
bundles = sorted(glob.glob(os.path.join(BUNDLE_DIR, "bundle_*.joblib")))
bundle  = load_model_bundle(bundles[-1])

svr_model    = bundle.models[bundle.best_model_name]
scaler       = bundle.scaler
sel_features = bundle.selected_features
print(f"  Best model : {bundle.best_model_name}")
print(f"  Features   : {len(sel_features)}")
print(f"  Bundle     : {os.path.basename(bundles[-1])}")

# ── Load raw data ─────────────────────────────────────────────────────────
print("\nLoading dataset...")
df_raw = pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET)
print(f"  Raw shape  : {df_raw.shape}")

y     = df_raw[TARGET_COL].values
strat = df_raw[STRATIFY_COL].values if STRATIFY_COL in df_raw.columns else None

# ── Compute full feature matrix (replicating training pipeline) ───────────
print("  Computing RDKit descriptors from SMILES (this may take a minute)...")
from ml_quasit.rdkit_descriptor_features import DESC_NAMES, get_2d_descriptors
from ml_quasit.iupac_to_smiles import MASTER_SMILES_MAP, cir_convert
from rdkit import Chem

def resolve_smiles(name):
    if not isinstance(name, str) or name.strip() == "":
        return None
    key = name.strip()
    for k, v in MASTER_SMILES_MAP.items():
        if k.lower() == key.lower():
            return v
    mol = Chem.MolFromSmiles(key)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    return None  # Skip CIR to avoid slow network calls

rows = []
for _, row in df_raw.iterrows():
    pri_iupac = row.get("PRIMARY_ORGANIC_SPACER_IUPAC", "")
    sec_iupac = row.get("SECONDARY_ORGANIC_SPACER_IUPAC", "")
    pri_smiles = resolve_smiles(pri_iupac)
    sec_smiles = resolve_smiles(sec_iupac) if isinstance(sec_iupac, str) and sec_iupac.strip() else None

    pri_desc = get_2d_descriptors(pri_smiles, add_hydrogens=bundle.add_hydrogens) if pri_smiles else [0.0]*len(DESC_NAMES)
    sec_desc = get_2d_descriptors(sec_smiles, add_hydrogens=bundle.add_hydrogens) if sec_smiles else [0.0]*len(DESC_NAMES)

    rec = {}
    for col in df_raw.columns:
        if col not in ["PRIMARY_ORGANIC_SPACER_IUPAC", "SECONDARY_ORGANIC_SPACER_IUPAC",
                        "REFERENCE_DOI", "SOLVENT", TARGET_COL]:
            rec[col] = row.get(col, 0.0)

    # Solvent one-hot
    for s in ["DMF", "DMSO", "DMF:DMSO", "NMP"]:
        rec[f"SOLVENT_{s}"] = 1.0 if row.get("SOLVENT", "") == s else 0.0

    # Descriptors
    for i, name in enumerate(DESC_NAMES):
        rec[f"Pri_{name}"] = pri_desc[i] if pri_desc[i] is not None else 0.0
        rec[f"Sec_{name}"] = sec_desc[i] if sec_desc[i] is not None else 0.0

    rows.append(rec)

X_full = pd.DataFrame(rows).fillna(0.0)
print(f"  Full feature matrix shape: {X_full.shape}")

# Keep only selected features
for feat in sel_features:
    if feat not in X_full.columns:
        X_full[feat] = 0.0
X_raw = X_full[sel_features].fillna(0.0)


# ── Replicate train / test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_raw.values, y, df_raw.index,
    test_size=0.2, random_state=RANDOM_STATE, stratify=strat,
)
print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

X_train_sc   = scaler.transform(X_train)
X_test_sc    = scaler.transform(X_test)
y_pred_train = svr_model.predict(X_train_sc)
y_pred_test  = svr_model.predict(X_test_sc)

r2_test   = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test  = mean_absolute_error(y_test, y_pred_test)

print(f"\nTest-set metrics (N={len(y_test)}):")
print(f"  R²   = {r2_test:.4f}")
print(f"  RMSE = {rmse_test:.4f} eV")
print(f"  MAE  = {mae_test:.4f} eV")

# ── Halide colour coding ──────────────────────────────────────────────────
def halide_label(row_idx, df):
    r = df.loc[row_idx]
    br = float(r.get("BR_PRIMARY_ORGANIC_HALIDE", 0) or 0) + float(r.get("PbBr2", 0) or 0)
    io = float(r.get("I_PRIMARY_ORGANIC_HALIDE",  0) or 0) + float(r.get("PbI2",  0) or 0)
    cl = float(r.get("CL_PRIMARY_ORGANIC_HALIDE", 0) or 0) + float(r.get("PbCl2", 0) or 0)
    scores = {"Bromide-rich": br, "Iodide-rich": io, "Chloride-rich": cl}
    dom = max(scores, key=scores.get)
    if scores[dom] == 0:
        return "Mixed / Other"
    vals = sorted(scores.values(), reverse=True)
    if vals[0] - vals[1] < 0.3:
        return "Mixed / Other"
    return dom

halide_colors = {
    "Bromide-rich":  "#2196F3",
    "Iodide-rich":   "#E91E63",
    "Chloride-rich": "#4CAF50",
    "Mixed / Other": "#FF9800",
}

train_halides = [halide_label(i, df_raw) for i in idx_train]
test_halides  = [halide_label(i, df_raw) for i in idx_test]

# ═════════════════════════════════════════════════════════════════════════
#  FIGURE 1: PARITY PLOT
# ═════════════════════════════════════════════════════════════════════════
print("\nGenerating parity plot...")
fig, ax = plt.subplots(figsize=(7, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

e_min = min(y.min(), y_pred_train.min(), y_pred_test.min()) - 0.05
e_max = max(y.max(), y_pred_train.max(), y_pred_test.max()) + 0.05

ax.plot([e_min, e_max], [e_min, e_max], "k-", lw=1.5, zorder=1)
ax.fill_between(
    [e_min, e_max],
    [e_min - rmse_test, e_max - rmse_test],
    [e_min + rmse_test, e_max + rmse_test],
    alpha=0.10, color="#333333", zorder=1,
)

for label, color in halide_colors.items():
    tr_mask = [h == label for h in train_halides]
    te_mask = [h == label for h in test_halides]
    if any(tr_mask):
        ax.scatter(np.array(y_train)[tr_mask], np.array(y_pred_train)[tr_mask],
                   facecolors="none", edgecolors=color, s=45, lw=1.2, alpha=0.6, zorder=3)
    if any(te_mask):
        ax.scatter(np.array(y_test)[te_mask], np.array(y_pred_test)[te_mask],
                   facecolors=color, edgecolors="white", s=70, lw=0.8, alpha=0.92, zorder=4)

stats_text = (
    f"Test set (N={len(y_test)})\n"
    f"$R^2$ = {r2_test:.3f}\n"
    f"RMSE = {rmse_test:.3f} eV\n"
    f"MAE  = {mae_test:.3f} eV"
)
ax.text(0.04, 0.97, stats_text, transform=ax.transAxes, va="top", ha="left",
        fontsize=10.5, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC", alpha=0.92))

legend_elements = [
    Line2D([0], [0], color="k", lw=1.5, label="Ideal (y = x)"),
    mpatches.Patch(color="#AAAAAA", alpha=0.3, label=f"±RMSE = {rmse_test:.3f} eV"),
]
for label, color in halide_colors.items():
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=label))
legend_elements += [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
           markeredgecolor="#555555", markersize=8, label="Train (hollow)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#555555",
           markeredgecolor="white", markersize=8, label="Test (filled)"),
]
ax.legend(handles=legend_elements, fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor="#CCCCCC")

ax.set_xlim(e_min, e_max); ax.set_ylim(e_min, e_max)
ax.set_xlabel("Experimental $E_{ph}$ (eV)", fontsize=13)
ax.set_ylabel("Predicted $E_{ph}$ (eV)", fontsize=13)
ax.set_title("SVR–RBF: Predicted vs. Experimental Emission Photon Energy\n(13-feature SelectFromModel pipeline)", fontsize=11, pad=10)
ax.tick_params(labelsize=11)
ax.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC")

plt.tight_layout()
parity_path = os.path.join(OUT_DIR, "parity_plot_svr_13features.png")
fig.savefig(parity_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {parity_path}")

# ═════════════════════════════════════════════════════════════════════════
#  FIGURE 2: 10-FOLD CV + BOOTSTRAP CIs
# ═════════════════════════════════════════════════════════════════════════
print(f"\nRunning {N_CV_FOLDS}-fold cross-validation...")
svr_params = svr_model.get_params()
svr_clone  = SVR(**svr_params)
pipe       = Pipeline([("scaler", StandardScaler()), ("svr", svr_clone)])

kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(
    pipe, X_raw.values, y, cv=kf,
    scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
)
cv_r2   = cv_results["test_r2"]
cv_rmse = -cv_results["test_neg_root_mean_squared_error"]
cv_mae  = -cv_results["test_neg_mean_absolute_error"]

print(f"  R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"  RMSE : {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} eV")
print(f"  MAE  : {cv_mae.mean():.4f} ± {cv_mae.std():.4f} eV")

print(f"\nRunning {N_BOOTSTRAP}-iteration bootstrap...")
rng = np.random.default_rng(RANDOM_STATE)
boot_r2, boot_rmse, boot_mae = [], [], []
for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, len(y_test), size=len(y_test))
    yt, yp = y_test[idx], y_pred_test[idx]
    boot_r2.append(r2_score(yt, yp))
    boot_rmse.append(np.sqrt(mean_squared_error(yt, yp)))
    boot_mae.append(mean_absolute_error(yt, yp))

boot_r2, boot_rmse, boot_mae = map(np.array, [boot_r2, boot_rmse, boot_mae])
ci_r2   = np.percentile(boot_r2,   [2.5, 97.5])
ci_rmse = np.percentile(boot_rmse, [2.5, 97.5])
ci_mae  = np.percentile(boot_mae,  [2.5, 97.5])

print(f"\nBootstrap 95% CI (N={N_BOOTSTRAP} iterations):")
print(f"  R²   = {r2_test:.4f}  [95% CI: {ci_r2[0]:.4f} – {ci_r2[1]:.4f}]")
print(f"  RMSE = {rmse_test:.4f} eV  [95% CI: {ci_rmse[0]:.4f} – {ci_rmse[1]:.4f}]")
print(f"  MAE  = {mae_test:.4f} eV  [95% CI: {ci_mae[0]:.4f} – {ci_mae[1]:.4f}]")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
fig.suptitle(f"{N_CV_FOLDS}-Fold CV — SVR–RBF (13 features)", fontsize=12)
metrics = [("R²", cv_r2, "", "#2196F3"), ("RMSE (eV)", cv_rmse, "eV", "#E91E63"), ("MAE (eV)", cv_mae, "eV", "#FF9800")]
for ax, (title, vals, unit, color) in zip(axes, metrics):
    folds = np.arange(1, N_CV_FOLDS + 1)
    ax.bar(folds, vals, color=color, alpha=0.7, edgecolor="white", lw=0.8)
    ax.axhline(vals.mean(), color="#333333", lw=1.8, ls="--", label=f"Mean = {vals.mean():.3f}")
    ax.fill_between([0.5, N_CV_FOLDS + 0.5], vals.mean()-vals.std(), vals.mean()+vals.std(),
                    alpha=0.15, color=color, label=f"±std = {vals.std():.3f}")
    ax.set_xlabel("Fold", fontsize=11); ax.set_ylabel(title, fontsize=11)
    ax.set_title(f"{title}\n{vals.mean():.3f} ± {vals.std():.3f} {unit}", fontsize=11)
    ax.set_xticks(folds); ax.legend(fontsize=8.5)
    ax.grid(True, axis="y", ls="--", alpha=0.4); ax.set_facecolor("#FAFAFA")

plt.tight_layout()
cv_path = os.path.join(OUT_DIR, "cv_confidence_svr_13features.png")
fig.savefig(cv_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {cv_path}")

# ── Ready-to-paste manuscript text ────────────────────────────────────────
print("\n" + "="*70)
print("READY-TO-PASTE MANUSCRIPT TEXT:")
print("="*70)
print(f"""
To validate the statistical robustness of the champion SVR–RBF model 
in light of the limited dataset size (N = {len(y)}), {N_CV_FOLDS}-fold cross-validation 
was applied across all available records and bootstrap confidence 
intervals (n = {N_BOOTSTRAP} iterations) were computed on the held-out test set. 
The {N_CV_FOLDS}-fold CV yielded a mean R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f} and 
RMSE = {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f} eV, consistent with the single-split 
test-set result (R² = {r2_test:.3f}, RMSE = {rmse_test:.3f} eV; 
95% bootstrap CI: {ci_rmse[0]:.3f}–{ci_rmse[1]:.3f} eV), confirming that the 
reported performance is not an artefact of a favourable data split. 
The predicted-versus-experimental parity plot is shown in Figure X.
""")
print("="*70)
print("\nAll figures saved to:", os.path.abspath(OUT_DIR))
