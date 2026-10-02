"""Streamlit web app for quasi-2D perovskite emission energy prediction.

Launch with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import streamlit as st
import pandas as pd
import numpy as np

from ml_quasit.inference_pipeline import (
    EmissionPredictor,
    PredictionInput,
    KNOWN_SOLVENTS,
    PRECURSOR_RATIO_COLUMNS,
)
from ml_quasit.model_serialization import find_latest_bundle

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Perovskite Emission Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — explicit overrides to ensure all labels are always visible ─────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ──────────────────────── */
:root {
    --bg-deep:    #0d0f1a;
    --bg-card:    #161929;
    --bg-input:   #1e2235;
    --border:     rgba(255,255,255,0.10);
    --accent:     #7c8ef7;
    --accent2:    #56cfb8;
    --text-main:  #e8eaf6;
    --text-muted: #8b93b8;
    --green:      #56cfb8;
    --yellow:     #f5c842;
    --radius:     12px;
}

/* ── Global ──────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: var(--bg-deep);
    color: var(--text-main);
}

/* ── Sidebar ──────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #111423 !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
    word-break: break-all;
    overflow-wrap: break-word;
    white-space: normal !important;
}

/* ── ALL widget labels — force white ─────── */
label,
.stSelectbox label,
.stRadio label,
.stCheckbox label,
.stNumberInput label,
.stTextInput label,
.stFileUploader label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stRadio > label,
.stCheckbox > label {
    color: var(--text-main) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ── Radio buttons ───────────────────────── */
.stRadio [data-testid="stWidgetLabel"] {
    color: var(--text-main) !important;
    margin-bottom: 6px;
}
.stRadio div[role="radiogroup"] label {
    color: var(--text-main) !important;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 14px !important;
    margin: 3px 4px 3px 0 !important;
    font-size: 0.85rem !important;
    cursor: pointer;
    transition: all 0.15s;
}
.stRadio div[role="radiogroup"] label:hover {
    border-color: var(--accent);
    background: rgba(124,142,247,0.12);
}
.stRadio div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}

/* ── Checkbox ────────────────────────────── */
.stCheckbox label {
    color: var(--text-main) !important;
    font-size: 0.9rem !important;
}
.stCheckbox label p {
    color: var(--text-main) !important;
}

/* ── Selectbox ───────────────────────────── */
.stSelectbox > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-main) !important;
}
.stSelectbox svg { color: var(--text-muted) !important; }

/* ── Number input ────────────────────────── */
.stNumberInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-main) !important;
}
.stNumberInput button {
    background: var(--bg-input) !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}

/* ── Text input ──────────────────────────── */
.stTextInput input {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-main) !important;
}

/* ── Tabs ────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 4px;
    gap: 4px;
    border-bottom: none;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: var(--text-muted) !important;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 8px 20px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
}

/* ── Primary button ──────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c8ef7, #56cfb8) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    color: #0d0f1a !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 14px 0 !important;
    letter-spacing: 0.3px;
    transition: opacity 0.2s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.88; }

/* ── Secondary button ────────────────────── */
.stButton > button:not([kind="primary"]) {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-main) !important;
}

/* ── Expander ────────────────────────────── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-main) !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── Dataframe ───────────────────────────── */
.stDataFrame, iframe { border-radius: var(--radius) !important; }

/* ── Divider ─────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Caption / small text ────────────────── */
.stCaptionContainer, small, caption {
    color: var(--text-muted) !important;
}

/* ── File uploader ───────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--bg-input) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-main) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper components ─────────────────────────────────────────────────────────

def section_header(icon: str, title: str):
    st.markdown(
        f"""<div style="display:flex;align-items:center;gap:8px;
        margin:18px 0 10px 0;padding-bottom:8px;
        border-bottom:1px solid rgba(255,255,255,0.10);">
        <span style="font-size:1.1rem">{icon}</span>
        <span style="font-size:0.95rem;font-weight:600;
        color:#e8eaf6;letter-spacing:0.4px">{title}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def field_label(text: str, required: bool = False, hint: str = ""):
    tag = ' <span style="color:#f5c842;font-size:0.75rem">(required)</span>' if required else ""
    hint_span = f' <span style="color:#8b93b8;font-size:0.78rem">— {hint}</span>' if hint else ""
    st.markdown(
        f'<p style="margin:10px 0 4px 0;font-size:0.85rem;'
        f'font-weight:500;color:#c5c9e0">{text}{tag}{hint_span}</p>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, sub: str = "", accent: str = "#7c8ef7"):
    st.markdown(
        f"""<div style="background:#161929;border:1px solid rgba(255,255,255,0.10);
        border-radius:14px;padding:20px 24px;text-align:center;margin:6px 0">
        <div style="font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;
        color:#8b93b8;margin-bottom:6px">{label}</div>
        <div style="font-size:2.4rem;font-weight:700;
        background:linear-gradient(90deg,{accent},{accent}aa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        line-height:1.1">{value}</div>
        {f'<div style="font-size:0.8rem;color:#8b93b8;margin-top:4px">{sub}</div>' if sub else ''}
        </div>""",
        unsafe_allow_html=True,
    )


def color_swatch(nm: float):
    hex_color = _nm_to_hex(nm)
    region = _color_region(nm)
    st.markdown(
        f"""<div style="background:#161929;border:1px solid rgba(255,255,255,0.10);
        border-radius:14px;padding:20px 24px;margin:6px 0">
        <div style="font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;
        color:#8b93b8;text-align:center;margin-bottom:10px">Predicted Emission Color</div>
        <div style="height:56px;border-radius:10px;
        background:linear-gradient(90deg,{hex_color}55,{hex_color},{hex_color}55);
        border:1px solid rgba(255,255,255,0.15)"></div>
        <div style="text-align:center;margin-top:10px;font-size:0.9rem;
        color:#e8eaf6;font-weight:500">{region}</div>
        <div style="text-align:center;font-size:0.75rem;color:#8b93b8">{hex_color} &nbsp;·&nbsp; {nm:.1f} nm</div>
        </div>""",
        unsafe_allow_html=True,
    )


def info_box(text: str, kind: str = "ok"):
    color = "#56cfb8" if kind == "ok" else "#f5c842"
    bg = "rgba(86,207,184,0.08)" if kind == "ok" else "rgba(245,200,66,0.08)"
    icon = "✔" if kind == "ok" else "⚠"
    st.markdown(
        f"""<div style="background:{bg};border:1px solid {color}40;
        border-radius:10px;padding:10px 16px;margin:10px 0;
        color:{color};font-size:0.85rem">
        <b>{icon}</b>&nbsp; {text}
        </div>""",
        unsafe_allow_html=True,
    )


def _nm_to_hex(nm: float) -> str:
    if nm < 380: return "#8B00FF"
    elif nm < 440:
        r, g, b = -(nm - 440) / 60, 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / 50, 1.0
    elif nm < 510:
        r, g, b = 0.0, 1.0, -(nm - 510) / 20
    elif nm < 580:
        r, g, b = (nm - 510) / 70, 1.0, 0.0
    elif nm < 645:
        r, g, b = 1.0, -(nm - 645) / 65, 0.0
    elif nm < 780:
        r, g, b = 1.0, 0.0, 0.0
    else:
        return "#8B0000"
    factor = (0.3 + 0.7 * (nm - 380) / 40) if nm < 420 else \
             (0.3 + 0.7 * (780 - nm) / 80) if nm > 700 else 1.0
    return "#{:02x}{:02x}{:02x}".format(
        int(min(255, max(0, r * factor * 255))),
        int(min(255, max(0, g * factor * 255))),
        int(min(255, max(0, b * factor * 255))),
    )


def _color_region(nm: float) -> str:
    if nm < 400:   return "UV / Deep Violet"
    elif nm < 450: return "Violet → Blue"
    elif nm < 495: return "Blue"
    elif nm < 520: return "Cyan / Blue-Green"
    elif nm < 565: return "Green"
    elif nm < 590: return "Yellow"
    elif nm < 625: return "Orange"
    elif nm < 700: return "Red"
    else:          return "Deep Red / NIR"


def _render_molecule_b64(smiles: str) -> str | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from io import BytesIO
        import base64
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=(280, 180))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        import traceback
        return f"ERROR:{type(e).__name__}:{str(e)}\n{traceback.format_exc()}"



# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    latest = find_latest_bundle()
    if latest is None:
        return None
    return EmissionPredictor(latest)


# ── University branding ───────────────────────────────────────────────────────
def _render_sidebar_branding():
    """Render SPRL / School of Chemistry / University of Hyderabad at the top of the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="
                text-align:center;
                padding:20px 12px 16px 12px;
                border-bottom:1px solid rgba(255,255,255,0.10);
                margin-bottom:16px;
            ">
                <div style="
                    font-size:1.55rem;
                    font-weight:800;
                    letter-spacing:3px;
                    background:linear-gradient(90deg,#7c8ef7,#56cfb8);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;
                    margin-bottom:6px;
                ">SPRL</div>
                <div style="
                    font-size:0.80rem;
                    font-weight:600;
                    color:#c5c9e0;
                    letter-spacing:0.8px;
                    text-transform:uppercase;
                    margin-bottom:3px;
                ">School of Chemistry</div>
                <div style="
                    font-size:0.72rem;
                    color:#8b93b8;
                    font-weight:400;
                ">University of Hyderabad</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar(predictor):
    with st.sidebar:
        st.markdown(
            '<div style="font-size:1.05rem;font-weight:700;color:#7c8ef7;'
            'padding:8px 0 12px 0;border-bottom:1px solid rgba(255,255,255,0.1);'
            'margin-bottom:14px">📦 Model Information</div>',
            unsafe_allow_html=True,
        )
        meta = predictor.bundle.metadata
        rows = [
            ("Config", meta.get("config_name", "N/A")),
            ("Best model", predictor.best_model_name),
            ("Selected features", str(len(predictor.bundle.selected_features))),
            ("Add Hydrogens", "Yes" if predictor.bundle.add_hydrogens else "No"),
            ("Feature selection", meta.get("feature_selection", "N/A")),
            ("Hyperparameter tuning", "Yes" if meta.get("hyperparameter_tuning") else "No"),
        ]
        for k, v in rows:
            st.markdown(
                f'<div style="margin:6px 0">'
                f'<span style="color:#8b93b8;font-size:0.78rem">{k}</span><br>'
                f'<span style="color:#e8eaf6;font-size:0.84rem;font-weight:500;'
                f'word-break:break-word">{v}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="margin:14px 0"/>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.9rem;font-weight:600;color:#7c8ef7;'
            'margin-bottom:8px">📊 Benchmark (Test Set)</div>',
            unsafe_allow_html=True,
        )
        if predictor.bundle.benchmark is not None:
            bench = predictor.bundle.benchmark.copy()
            st.dataframe(bench.round(4), width='stretch', hide_index=True)

        st.markdown('<hr style="margin:14px 0"/>', unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#8b93b8;font-size:0.78rem;line-height:1.6">'
            'Predicts <b style="color:#e8eaf6">emission photon energy (eV)</b> '
            'and <b style="color:#e8eaf6">wavelength (nm)</b> of quasi-2D '
            'perovskite blue LEDs from organic spacer chemistry and precursor composition.'
            '</div>',
            unsafe_allow_html=True,
        )


# ── SINGLE PREDICTION TAB ─────────────────────────────────────────────────────
def render_single_tab(predictor):
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT: inputs ──────────────────────────────────────────────────────────
    with col_left:
        # ── Organic Spacer ────────────────────────────────────────────────────
        section_header("🔬", "Organic Spacer")

        field_label("Input method")
        spacer_mode = st.radio(
            "input_mode",
            ["Select from known spacers", "Enter IUPAC name", "Enter SMILES"],
            horizontal=True,
            label_visibility="collapsed",
            key="spacer_mode_radio",
        )

        if spacer_mode == "Select from known spacers":
            field_label("Primary organic spacer", required=True)
            default_idx = (
                predictor.known_primary_spacers.index("2-phenylethan-1-amine")
                if "2-phenylethan-1-amine" in predictor.known_primary_spacers
                else 0
            )
            primary_spacer = st.selectbox(
                "primary_spacer_select",
                predictor.known_primary_spacers,
                index=default_idx,
                label_visibility="collapsed",
            )
            st.caption(f"22 spacers in training data  ·  in-domain prediction")

        elif spacer_mode == "Enter IUPAC name":
            field_label("Primary spacer — IUPAC name", required=True,
                        hint="e.g. 2-phenylethan-1-amine")
            primary_spacer = st.text_input(
                "primary_spacer_iupac",
                value="2-phenylethan-1-amine",
                label_visibility="collapsed",
            )

        else:
            field_label("Primary spacer — SMILES string", required=True,
                        hint="e.g. NCCc1ccccc1")
            primary_spacer = st.text_input(
                "primary_spacer_smiles",
                value="NCCc1ccccc1",
                label_visibility="collapsed",
            )

        col_h, col_f = st.columns([1, 1])
        with col_h:
            field_label("Counter-ion halide")
            primary_halide = st.selectbox(
                "primary_halide_sel",
                ["Br", "Cl", "I"],
                index=0,
                label_visibility="collapsed",
            )
        with col_f:
            field_label("Spacer molar fraction", hint="1.0 = single spacer")
            primary_fraction = st.number_input(
                "primary_frac",
                min_value=0.0, max_value=1.0,
                value=1.0, step=0.05, format="%.2f",
                label_visibility="collapsed",
            )

        # ── Mixed spacer ──────────────────────────────────────────────────────
        field_label("Mixed spacer system")
        is_mixed = st.checkbox(
            "Use two organic spacer cations (mixed system)",
            value=False,
            key="mixed_checkbox",
        )

        secondary_spacer = None
        secondary_halide = "Br"
        secondary_fraction = 0.0

        if is_mixed:
            st.markdown(
                '<div style="background:#161929;border:1px solid rgba(255,255,255,0.08);'
                'border-radius:12px;padding:14px 16px;margin:8px 0">',
                unsafe_allow_html=True,
            )
            field_label("Secondary organic spacer")
            sec_options = ["(None)"] + predictor.known_secondary_spacers
            sec_sel = st.selectbox(
                "sec_spacer_sel",
                sec_options,
                label_visibility="collapsed",
                key="sec_spacer",
            )
            secondary_spacer = None if sec_sel == "(None)" else sec_sel

            col_sh, col_sf = st.columns([1, 1])
            with col_sh:
                field_label("Secondary halide")
                secondary_halide = st.selectbox(
                    "sec_halide_sel",
                    ["Br", "Cl", "I"],
                    label_visibility="collapsed",
                    key="sec_halide",
                )
            with col_sf:
                field_label(
                    "How much of the secondary spacer to use?",
                    hint="e.g. 0.20 means 20% secondary, 80% primary"
                )
                secondary_fraction = st.number_input(
                    "sec_frac",
                    min_value=0.0, max_value=1.0,
                    value=0.2, step=0.05, format="%.2f",
                    label_visibility="collapsed",
                    key="sec_frac",
                )
                primary_fraction = 1.0 - secondary_fraction
            # Show live split summary
            st.markdown(
                f'<p style="font-size:0.8rem;color:#56cfb8;margin:6px 0 0 0">'
                f'Mix ratio: <b>{primary_fraction:.0%}</b> primary'
                f' + <b>{secondary_fraction:.0%}</b> secondary</p>',
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Processing ────────────────────────────────────────────────────────
        section_header("⚙️", "Processing Conditions")

        col_sol, col_ratio = st.columns([1, 1])
        with col_sol:
            field_label("Solvent", required=True)
            default_solvent = KNOWN_SOLVENTS.index("DMSO") if "DMSO" in KNOWN_SOLVENTS else 0
            solvent = st.selectbox(
                "solvent_sel",
                KNOWN_SOLVENTS,
                index=default_solvent,
                label_visibility="collapsed",
            )
        with col_ratio:
            field_label("Spacer : Pb molar ratio", hint="typically 1.6 – 3.0")
            spacer_to_pb = st.number_input(
                "spacer_pb",
                min_value=0.1, max_value=20.0,
                value=2.0, step=0.1, format="%.1f",
                label_visibility="collapsed",
            )

        # ── Precursor composition ─────────────────────────────────────────────
        section_header("⚗️", "Precursor Composition  (molar ratios to Pb)")

        st.markdown(
            '<div style="background:#161929;border:1px solid rgba(255,255,255,0.08);'
            'border-radius:12px;padding:14px 16px">',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-size:0.78rem;color:#8b93b8;margin:0 0 10px 0">'
            'Lead halides</p>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            field_label("PbCl₂")
            PbCl2 = st.number_input("pbcl2", value=0.0, step=0.1,
                                    format="%.2f", label_visibility="collapsed")
        with c2:
            field_label("PbBr₂")
            PbBr2 = st.number_input("pbbr2", value=1.0, step=0.1,
                                    format="%.2f", label_visibility="collapsed")
        with c3:
            field_label("PbI₂")
            PbI2 = st.number_input("pbi2", value=0.0, step=0.1,
                                   format="%.2f", label_visibility="collapsed")

        st.markdown(
            '<p style="font-size:0.78rem;color:#8b93b8;margin:12px 0 8px 0">'
            'Cs⁺ salts</p>',
            unsafe_allow_html=True,
        )
        c4, c5, c6 = st.columns(3)
        with c4:
            field_label("CsCl / Pb")
            CsCl = st.number_input("cscl", value=0.0, step=0.1,
                                   format="%.2f", label_visibility="collapsed")
        with c5:
            field_label("CsBr / Pb")
            CsBr = st.number_input("csbr", value=1.0, step=0.1,
                                   format="%.2f", label_visibility="collapsed")
        with c6:
            field_label("CsI / Pb")
            CsI = st.number_input("csi", value=0.0, step=0.1,
                                  format="%.2f", label_visibility="collapsed")

        # Initialize optional cation defaults before expander
        RbBr = RbI = FABr = FAI = MABr = MAI = 0.0

        with st.expander("➕  Add Rb⁺ / FA⁺ / MA⁺ cations  (optional)"):
            r1, r2 = st.columns(2)
            with r1:
                field_label("RbBr / Pb")
                RbBr = st.number_input("rbbr", value=0.0, step=0.1,
                                       format="%.2f", label_visibility="collapsed")
            with r2:
                field_label("RbI / Pb")
                RbI = st.number_input("rbi", value=0.0, step=0.1,
                                      format="%.2f", label_visibility="collapsed")
            f1, f2 = st.columns(2)
            with f1:
                field_label("FABr / Pb")
                FABr = st.number_input("fabr", value=0.0, step=0.1,
                                       format="%.2f", label_visibility="collapsed")
            with f2:
                field_label("FAI / Pb")
                FAI = st.number_input("fai", value=0.0, step=0.1,
                                      format="%.2f", label_visibility="collapsed")
            m1, m2 = st.columns(2)
            with m1:
                field_label("MABr / Pb")
                MABr = st.number_input("mabr", value=0.0, step=0.1,
                                       format="%.2f", label_visibility="collapsed")
            with m2:
                field_label("MAI / Pb")
                MAI = st.number_input("mai", value=0.0, step=0.1,
                                      format="%.2f", label_visibility="collapsed")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button(
            "🔮  Predict Emission Energy",
            type="primary",
            width='stretch',
        )

    # ── RIGHT: results ────────────────────────────────────────────────────────
    with col_right:
        if predict_clicked:
            try:
                inp = PredictionInput(
                    primary_spacer=primary_spacer,
                    solvent=solvent,
                    primary_halide=primary_halide,
                    primary_spacer_fraction=primary_fraction,
                    secondary_spacer=secondary_spacer if is_mixed else None,
                    secondary_halide=secondary_halide,
                    secondary_spacer_fraction=secondary_fraction if is_mixed else 0.0,
                    spacer_to_pb_ratio=spacer_to_pb,
                    PbCl2=PbCl2, PbBr2=PbBr2, PbI2=PbI2,
                    CsCl_TO_Pb=CsCl, CsBr_TO_Pb=CsBr, CsI_TO_Pb=CsI,
                    RbBr_TO_Pb=RbBr, RbI_TO_Pb=RbI,
                    FABr_TO_Pb=FABr, FAI_TO_Pb=FAI,
                    MABr_TO_Pb=MABr, MAI_TO_Pb=MAI,
                )

                with st.spinner("Computing RDKit descriptors and predicting..."):
                    result = predictor.predict(inp, return_features=False)

                # ── Main metrics ──────────────────────────────────────────────
                section_header("📊", "Prediction Results")

                m1, m2 = st.columns(2)
                with m1:
                    metric_card("Photon Energy",
                                f"{result.predicted_ev:.4f}",
                                sub="eV", accent="#7c8ef7")
                with m2:
                    metric_card("Wavelength",
                                f"{result.predicted_nm:.1f}",
                                sub="nm", accent="#56cfb8")

                # Color swatch
                color_swatch(result.predicted_nm)

                # Confidence badge
                if "[OK]" in result.confidence_note:
                    info_box(result.confidence_note.replace("[OK] ", ""), kind="ok")
                else:
                    info_box(result.confidence_note.replace("[WARNING] ", ""), kind="warn")

                st.markdown(
                    f'<p style="font-size:0.82rem;color:#8b93b8;margin:6px 0 0 0">'
                    f'Prediction by: <b style="color:#e8eaf6">{result.model_name}</b></p>',
                    unsafe_allow_html=True,
                )

                # ── All model predictions ─────────────────────────────────────
                section_header("📈", "All Model Predictions")
                preds_df = pd.DataFrame([
                    {
                        "Model": name,
                        "Energy (eV)": f"{ev:.4f}",
                        "Wavelength (nm)": f"{1240.0 / ev:.1f}" if ev > 0 else "—",
                    }
                    for name, ev in sorted(
                        result.all_model_predictions.items(),
                        key=lambda x: x[1],
                    )
                ])
                st.dataframe(preds_df, width='stretch', hide_index=True)

                # ── Molecule structure(s) ─────────────────────────────────────
                section_header("⬡", "Spacer Molecule Structure")

                from ml_quasit.inference_pipeline import _resolve_smiles
                pri_smiles = _resolve_smiles(primary_spacer)
                sec_smiles = _resolve_smiles(secondary_spacer) if (
                    is_mixed and secondary_spacer
                ) else None

                if pri_smiles or sec_smiles:
                    n_cols = 2 if sec_smiles else 1
                    mol_cols = st.columns(n_cols)

                    # Primary molecule
                    with mol_cols[0]:
                        if pri_smiles:
                            b64 = _render_molecule_b64(pri_smiles)
                            if b64 and b64.startswith("ERROR:"):
                                st.error(f"Image Error: {b64}")
                            elif b64:
                                st.markdown(
                                    f'<div style="background:#fff;border-radius:12px;'
                                    f'padding:10px;text-align:center;'
                                    f'border:1px solid rgba(255,255,255,0.1)">'
                                    f'<img src="data:image/png;base64,{b64}" '
                                    f'style="max-width:100%;border-radius:6px"/></div>',
                                    unsafe_allow_html=True,
                                )
                            st.markdown(
                                f'<p style="text-align:center;font-size:0.78rem;'
                                f'color:#8b93b8;margin:4px 0">'
                                f'<b style="color:#7c8ef7">Primary spacer</b><br>'
                                f'{primary_spacer}<br>'
                                f'<span style="font-size:0.72rem">SMILES: {pri_smiles}</span></p>',
                                unsafe_allow_html=True,
                            )

                    # Secondary molecule (mixed system only)
                    if sec_smiles and len(mol_cols) > 1:
                        with mol_cols[1]:
                            b64_sec = _render_molecule_b64(sec_smiles)
                            if b64_sec and b64_sec.startswith("ERROR:"):
                                st.error(f"Image Error: {b64_sec}")
                            elif b64_sec:
                                st.markdown(
                                    f'<div style="background:#fff;border-radius:12px;'
                                    f'padding:10px;text-align:center;'
                                    f'border:1px solid rgba(255,255,255,0.1)">'
                                    f'<img src="data:image/png;base64,{b64_sec}" '
                                    f'style="max-width:100%;border-radius:6px"/></div>',
                                    unsafe_allow_html=True,
                                )
                            st.markdown(
                                f'<p style="text-align:center;font-size:0.78rem;'
                                f'color:#8b93b8;margin:4px 0">'
                                f'<b style="color:#56cfb8">Secondary spacer</b><br>'
                                f'{secondary_spacer}<br>'
                                f'<span style="font-size:0.72rem">SMILES: {sec_smiles}</span></p>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.warning("Could not resolve spacer structure for visualization.")

            except ValueError as e:
                st.error(f"Input error: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                import traceback
                with st.expander("Debug traceback"):
                    st.code(traceback.format_exc())

        else:
            # Placeholder
            st.markdown(
                """<div style="background:#161929;border:1px solid rgba(255,255,255,0.08);
                border-radius:16px;padding:80px 32px;text-align:center;margin-top:40px">
                <div style="font-size:3rem;margin-bottom:16px">🔮</div>
                <div style="color:#c5c9e0;font-size:1.05rem;font-weight:500;margin-bottom:8px">
                Configure your perovskite system</div>
                <div style="color:#8b93b8;font-size:0.88rem">
                Fill in the spacer, solvent and precursor details on the left,<br>
                then click <b style="color:#7c8ef7">Predict Emission Energy</b>
                </div></div>""",
                unsafe_allow_html=True,
            )


# ── BATCH TAB ─────────────────────────────────────────────────────────────────
def render_batch_tab(predictor):
    section_header("📋", "Batch Prediction from CSV")

    st.markdown(
        '<p style="color:#8b93b8;font-size:0.88rem">Upload a CSV with one '
        'composition per row. Required columns: '
        '<code style="background:#1e2235;padding:2px 6px;border-radius:4px">'
        'primary_spacer</code> and '
        '<code style="background:#1e2235;padding:2px 6px;border-radius:4px">'
        'solvent</code>.</p>',
        unsafe_allow_html=True,
    )

    # Template
    template_df = pd.DataFrame({
        "primary_spacer":  ["2-phenylethan-1-amine", "butan-1-amine", "NCCc1ccccc1"],
        "solvent":         ["DMSO",                   "DMF",           "DMSO"],
        "primary_halide":  ["Br",                     "Br",            "Br"],
        "PbBr2":           [1.0,                      1.0,             1.0],
        "CsBr_TO_Pb":      [1.0,                      1.0,             0.8],
        "spacer_to_pb_ratio": [2.0,                   2.0,             2.5],
    })

    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        st.download_button(
            "⬇  Download CSV Template",
            template_df.to_csv(index=False),
            file_name="prediction_template.csv",
            mime="text/csv",
            width='stretch',
        )
    with col_info:
        st.markdown(
            '<p style="color:#8b93b8;font-size:0.8rem;padding-top:8px">'
            'All columns from the input form are supported as CSV headers.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    field_label("Upload your CSV file")
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        label_visibility="collapsed",
        key="batch_upload",
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.markdown(
            f'<p style="color:#56cfb8;font-size:0.88rem;margin:8px 0">'
            f'Loaded <b>{len(df)}</b> rows from <code>{uploaded.name}</code></p>',
            unsafe_allow_html=True,
        )
        st.dataframe(df.head(5), width='stretch', hide_index=True)

        if st.button("🚀  Run Batch Prediction", type="primary", width='content'):
            inputs = []
            for _, row in df.iterrows():
                kwargs = {
                    "primary_spacer": str(row["primary_spacer"]),
                    "solvent": str(row["solvent"]),
                }
                for col in ["primary_halide", "secondary_spacer", "secondary_halide"]:
                    if col in row and pd.notna(row[col]):
                        kwargs[col] = str(row[col])
                for col in ["primary_spacer_fraction", "secondary_spacer_fraction",
                            "spacer_to_pb_ratio"] + PRECURSOR_RATIO_COLUMNS:
                    if col in row and pd.notna(row[col]):
                        kwargs[col] = float(row[col])
                inputs.append(PredictionInput(**kwargs))

            with st.spinner(f"Predicting {len(inputs)} compositions..."):
                results_df = predictor.predict_batch(inputs)

            st.success(f"Predicted {len(results_df)} compositions successfully.")

            show_cols = ["primary_spacer", "solvent",
                         "predicted_ev", "predicted_nm", "model", "confidence"]
            show_cols = [c for c in show_cols if c in results_df.columns]
            st.dataframe(results_df[show_cols], width='stretch', hide_index=True)

            st.download_button(
                "⬇  Download Results CSV",
                results_df.to_csv(index=False),
                file_name="predictions_output.csv",
                mime="text/csv",
            )


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # ── University branding in sidebar ───────────────────────────────────────
    _render_sidebar_branding()

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        """<div style="text-align:center;padding:24px 0 16px 0">
        <div style="font-size:1.85rem;font-weight:700;
        background:linear-gradient(90deg,#7c8ef7,#56cfb8);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        margin-bottom:6px">
        🔬 Quasi-2D Perovskite Emission Predictor
        </div>
        <div style="color:#8b93b8;font-size:0.9rem">
        ML-powered prediction of emission photon energy &amp; wavelength of Quasi-2D Metal Halide Perovskites
        </div></div>""",
        unsafe_allow_html=True,
    )

    predictor = load_predictor()

    if predictor is None:
        st.error(
            "No trained model found. Train first:\n\n"
            "```bash\n"
            "python scripts/run_experiment.py --config "
            "configs/emission_photon_energy/add_hydrogens_tuned_select_from_model.yaml\n"
            "```"
        )
        return

    render_sidebar(predictor)

    tab1, tab2 = st.tabs(["🧪  Single Prediction", "📋  Batch Prediction"])
    with tab1:
        render_single_tab(predictor)
    with tab2:
        render_batch_tab(predictor)


if __name__ == "__main__":
    main()
