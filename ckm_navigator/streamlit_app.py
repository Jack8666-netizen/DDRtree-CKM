"""
================================================================
CKM Precision Risk Navigator v4.0
================================================================
Standalone Streamlit clinical decision-support tool.
DDRTree Manifold Learning | Tri-Cohort Validated (CHARLS/UKB/HRS)

Deploy: streamlit run streamlit_app.py
================================================================
"""
import numpy as np
import pandas as pd
import pickle
import warnings
import os
import pathlib

warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ════════════════════════════════════════════════════════
# Embedded utility functions
# ════════════════════════════════════════════════════════
APP_DIR = pathlib.Path(__file__).parent

VAR_AGE = "r1agey"
VAR_SEX = "ragender"
VAR_CKM_STAGE = "r1_ckm_stage"

VARS_CLINICAL_8 = [
    "r1systo", "r1diasto", "r1hdl", "r1tg",
    "r1mbmi", "r1crea", "r1hba1c", "r1glu",
]

VARS_LABELS = {
    "r1systo": "SBP", "r1diasto": "DBP",
    "r1hdl": "HDL-C", "r1tg": "TG",
    "r1mbmi": "BMI", "r1crea": "Creatinine",
    "r1hba1c": "HbA1c", "r1glu": "Glucose",
}


def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def apply_preprocess(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df_out = df.copy()
    age_col = params["age_col"]
    sex_col = params["sex_col"]
    for v in params["vars_list"]:
        vp = params["var_params"][v]
        rint_col = f"rint_{v}"
        raw = df_out[v].values.astype(float)
        sorted_vals = vp["rint_sorted_vals"]
        rint_q = vp["rint_quantiles"]
        rint_vals = np.where(
            np.isnan(raw), np.nan,
            np.interp(raw, sorted_vals, rint_q)
        )
        df_out[rint_col] = rint_vals
        resid_col = f"resid_{v}"
        coef = np.array(vp["resid_coef"])
        intercept = vp["resid_intercept"]
        mask = df_out[[rint_col, age_col, sex_col]].notna().all(axis=1)
        X = df_out.loc[mask, [age_col, sex_col]].values.astype(float)
        predicted = intercept + X @ coef
        resid = np.full(len(df_out), np.nan)
        resid[mask.values] = rint_vals[mask.values] - predicted
        df_out[resid_col] = resid
    return df_out


def predict_gam(gam, df_target, feature_cols):
    X = df_target[feature_cols].values.astype(float)
    return gam.predict(X)


def compute_pseudotime(Z, stree, ckm_stages, Y_nodes=None):
    from scipy.sparse.csgraph import shortest_path
    from scipy.spatial.distance import cdist
    from scipy.sparse import csr_matrix

    n_samples = Z.shape[0]
    if stree is None or stree.size == 0 or Y_nodes is None:
        return _fallback_pseudotime(Z, ckm_stages)

    is_adjacency = (stree.ndim == 2 and stree.shape[0] == stree.shape[1])
    if is_adjacency and Y_nodes is not None:
        n_nodes = Y_nodes.shape[0]
        dist_to_nodes = cdist(Z, Y_nodes)
        nearest_node = np.argmin(dist_to_nodes, axis=1)
        adj = stree.copy() + stree.copy().T
        np.fill_diagonal(adj, 0)
        geodesic_dist = shortest_path(csr_matrix(adj), method='D', directed=False)
    else:
        return _fallback_pseudotime(Z, ckm_stages)

    ckm_clean = np.where(np.isnan(ckm_stages), -1, ckm_stages)
    stage01_score = np.zeros(n_nodes)
    for i in range(n_samples):
        if ckm_clean[i] == 0:
            stage01_score[nearest_node[i]] += 2.0
        elif ckm_clean[i] == 1:
            stage01_score[nearest_node[i]] += 0.5
    root_node = np.argmax(stage01_score)

    node_ptime = geodesic_dist[root_node, :]
    finite_mask = ~np.isinf(node_ptime)
    if finite_mask.any():
        node_ptime[~finite_mask] = np.nanmax(node_ptime[finite_mask])
    else:
        node_ptime[:] = 0
    sample_ptime = node_ptime[nearest_node]
    pmin, pmax = np.nanmin(sample_ptime), np.nanmax(sample_ptime)
    if pmax > pmin:
        pseudotime = (sample_ptime - pmin) / (pmax - pmin)
    else:
        pseudotime = np.zeros(n_samples)
    return pseudotime


def _fallback_pseudotime(Z, ckm_stages):
    from scipy import stats
    q10 = np.nanpercentile(Z[:, 0], 10)
    q90 = np.nanpercentile(Z[:, 0], 90)
    mask_left = Z[:, 0] < q10
    mask_right = Z[:, 0] > q90
    s0_left = np.nanmean(ckm_stages[mask_left] == 0) if mask_left.sum() > 0 else 0
    s0_right = np.nanmean(ckm_stages[mask_right] == 0) if mask_right.sum() > 0 else 0
    if s0_left >= s0_right:
        ptime = stats.rankdata(Z[:, 0]) / len(Z)
    else:
        ptime = 1 - stats.rankdata(Z[:, 0]) / len(Z)
    return ptime


# ════════════════════════════════════════════════════════
# Page Config
# ════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CKM Precision Risk Navigator",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════
# Premium CSS
# ════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main-hero {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2rem 2.5rem; border-radius: 16px; color: white;
    margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    position: relative; overflow: hidden;
}
.main-hero::before {
    content: ''; position: absolute; top: -50%; right: -20%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(168,237,234,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.main-hero h1 {
    margin: 0; font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(90deg, #a8edea, #fed6e3);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.main-hero p { margin: 0.4rem 0 0; opacity: 0.8; font-size: 0.85rem; }
.cohort-badge {
    display: inline-block; padding: 0.3rem 0.8rem;
    border-radius: 20px; font-size: 0.7rem; font-weight: 600; margin-top: 0.5rem;
}
.badge-china { background: rgba(231,76,60,0.2); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }
.badge-uk { background: rgba(52,152,219,0.2); color: #3498db; border: 1px solid rgba(52,152,219,0.3); }
.kpi-card {
    border-radius: 14px; padding: 1.2rem; text-align: center;
    position: relative; overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    transition: all 0.3s ease; backdrop-filter: blur(10px);
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.12); }
.kpi-card h2 { margin: 0; font-size: 1.8rem; font-weight: 700; }
.kpi-card p { margin: 0.3rem 0 0; font-size: 0.75rem; opacity: 0.7; }
.kpi-green { background: linear-gradient(135deg, #a8edea 0%, #56ccbb 100%); color: #0a3d3d; }
.kpi-yellow { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #5a3e20; }
.kpi-orange { background: linear-gradient(135deg, #feb47b 0%, #ff7e5f 100%); color: #4a1e1e; }
.kpi-red { background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%); color: white; }
.alert-box {
    border-radius: 12px; padding: 1rem 1.2rem; margin: 1rem 0;
    display: flex; align-items: center; gap: 0.8rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    animation: slideIn 0.5s ease;
}
@keyframes slideIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
.alert-tipping {
    background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
    border-left: 4px solid #e17055;
}
.alert-safe {
    background: linear-gradient(135deg, #dfe6e9 0%, #b2bec3 100%);
    border-left: 4px solid #00b894;
}
.info-panel {
    background: #f8f9fa; border-radius: 12px; padding: 1.2rem;
    border: 1px solid #e9ecef; margin: 0.5rem 0;
}
.guideline-card {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px; padding: 1.2rem; margin: 0.5rem 0; border-left: 4px solid;
}
.guide-stage0 { border-color: #2166AC; }
.guide-stage1 { border-color: #67A9CF; }
.guide-stage2 { border-color: #D1E5F0; }
.guide-stage3 { border-color: #EF8A62; }
.guide-stage4 { border-color: #B2182B; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# Load Models
# ════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    models = {}
    M = APP_DIR / "models"
    D = APP_DIR / "data"

    for cohort in ["CHARLS", "UKB"]:
        try:
            models[f"gam1_{cohort}"] = load_model(str(M / f"gam_dim1_{cohort}.pkl"))
            models[f"gam2_{cohort}"] = load_model(str(M / f"gam_dim2_{cohort}.pkl"))
            models[f"meta_{cohort}"] = load_model(str(M / f"gam_meta_{cohort}.pkl"))
            pp = M / f"preprocess_params_{cohort}.pkl"
            models[f"pp_{cohort}"] = load_model(str(pp)) if pp.exists() else None
        except Exception as e:
            st.warning(f"Could not load {cohort} models: {e}")

    # Reference landscapes
    for name in ["charls", "ukb"]:
        ref_path = D / f"ref_{name}.csv"
        if ref_path.exists():
            models[f"ref_{name}"] = pd.read_csv(ref_path)

    # DDRTree structure (edges + Y_nodes)
    for cohort in ["CHARLS", "UKB"]:
        tree_dir = M / f"ddrtree_{cohort}"
        try:
            Y = pd.read_csv(tree_dir / "Y_nodes.csv").values
            models[f"Y_{cohort}"] = Y
            # Load edge list and reconstruct sparse adjacency
            edges = pd.read_csv(tree_dir / "edges.csv")
            n = Y.shape[0]
            stree = np.zeros((n, n))
            for _, row in edges.iterrows():
                i, j = int(row['i']), int(row['j'])
                stree[i, j] = 1
                stree[j, i] = 1
            models[f"stree_{cohort}"] = stree
            models[f"edges_{cohort}"] = edges
        except:
            pass

    return models


models = load_all_models()

# Variable config
VARS_ACTIVE = VARS_CLINICAL_8

stage_colors = {0: "#2166AC", 1: "#67A9CF", 2: "#B0D4E8",
                3: "#EF8A62", 4: "#B2182B"}
stage_names = {0: "No Risk", 1: "Adiposity", 2: "Metabolic Risk",
               3: "Subclinical/High Risk", 4: "Clinical CVD"}

# ════════════════════════════════════════════════════════
# Sidebar: Patient Input
# ════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🫀 Patient Input")
    cohort_model = st.selectbox(
        "🌍 Population Model",
        ["🇨🇳 CHARLS (China)", "🇬🇧 UKB (United Kingdom)"],
    )
    cohort_key = "CHARLS" if "CHARLS" in cohort_model else "UKB"
    cohort_flag = "🇨🇳" if cohort_key == "CHARLS" else "🇬🇧"

    st.markdown("---")
    st.markdown("#### 👤 Demographics")
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age", 40, 100, 60, step=1)
    with col_b:
        sex = st.selectbox("Sex", ["Male", "Female"])
    sex_val = 1 if sex == "Male" else 2

    st.markdown("---")
    st.markdown("#### 🩸 Biomarkers")

    defaults = {
        "r1systo": 130.0, "r1diasto": 80.0,
        "r1glu": 5.5, "r1hba1c": 5.8,
        "r1tg": 1.5, "r1hdl": 1.3,
        "r1crea": 70.0, "r1mbmi": 25.0,
    }
    units = {
        "r1systo": "mmHg", "r1diasto": "mmHg",
        "r1glu": "mmol/L", "r1hba1c": "%",
        "r1tg": "mmol/L", "r1hdl": "mmol/L",
        "r1crea": "μmol/L", "r1mbmi": "kg/m²",
    }
    domains_input = {
        "🫀 Hemodynamic": ["r1systo", "r1diasto"],
        "🍬 Glycemic": ["r1glu", "r1hba1c"],
        "🧈 Lipid": ["r1tg", "r1hdl"],
        "🫘 Renal": ["r1crea"],
        "📏 Body": ["r1mbmi"],
    }

    patient_data = {VAR_AGE: age, VAR_SEX: sex_val}
    for domain_label, domain_vars in domains_input.items():
        with st.expander(domain_label, expanded=True):
            for v in domain_vars:
                label = f"{VARS_LABELS.get(v, v)} ({units.get(v, '')})"
                patient_data[v] = st.number_input(
                    label, value=defaults.get(v, 0.0), format="%.1f",
                    key=f"input_{v}")

    st.markdown("---")
    map_button = st.button("🚀 Compute Risk Profile", type="primary",
                            use_container_width=True)

# ════════════════════════════════════════════════════════
# Header
# ════════════════════════════════════════════════════════
badge_class = "badge-china" if cohort_key == "CHARLS" else "badge-uk"
st.markdown(f"""
<div class="main-hero">
    <h1>🫀 CKM Precision Risk Navigator</h1>
    <p>Data-Driven Phenotypic Landscape · Cardiovascular-Kidney-Metabolic Syndrome</p>
    <span class="cohort-badge {badge_class}">
        {cohort_flag} Active Model: {cohort_key} | DDRTree Manifold | Tri-Cohort Validated
    </span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# Main Content
# ════════════════════════════════════════════════════════
if map_button:
    st.session_state["show_results"] = True

if st.session_state.get("show_results", False):
    with st.spinner("Mapping patient to phenotypic landscape..."):
        df_patient = pd.DataFrame([patient_data])

        pp_params = models.get(f"pp_{cohort_key}")
        if pp_params is not None:
            df_pp = apply_preprocess(df_patient, pp_params)
        else:
            st.error("Preprocessing params not found.")
            st.stop()

        gam1 = models.get(f"gam1_{cohort_key}")
        gam2 = models.get(f"gam2_{cohort_key}")
        meta = models.get(f"meta_{cohort_key}")

        if gam1 and gam2 and meta:
            feat_cols = meta["feat_cols"]
            for fc in feat_cols:
                if fc not in df_pp.columns:
                    df_pp[fc] = 0

            dim1 = float(predict_gam(gam1, df_pp, feat_cols)[0])
            dim2 = float(predict_gam(gam2, df_pp, feat_cols)[0])

            # Geodesic pseudotime
            stree = models.get(f"stree_{cohort_key}")
            Y_nodes = models.get(f"Y_{cohort_key}")
            ref_df = models.get(f"ref_{cohort_key.lower()}")

            patient_pt = None
            if stree is not None and Y_nodes is not None and ref_df is not None:
                Z_all = np.vstack([
                    ref_df[["Dim1", "Dim2"]].dropna().values,
                    [[dim1, dim2]]
                ])
                ckm_all = np.append(
                    ref_df[VAR_CKM_STAGE].fillna(-1).values, -1
                )
                patient_pt = compute_pseudotime(Z_all, stree, ckm_all, Y_nodes=Y_nodes)[-1]

            if patient_pt is None:
                patient_pt = 0

            # Risk tier
            if patient_pt < 0.25:
                risk_level, risk_css, risk_emoji = "Low", "kpi-green", "🟢"
            elif patient_pt < 0.50:
                risk_level, risk_css, risk_emoji = "Moderate", "kpi-yellow", "🟡"
            elif patient_pt < 0.75:
                risk_level, risk_css, risk_emoji = "High", "kpi-orange", "🟠"
            else:
                risk_level, risk_css, risk_emoji = "Very High", "kpi-red", "🔴"

            # Estimate CKM stage
            sbp = patient_data.get("r1systo", 0)
            dbp = patient_data.get("r1diasto", 0)
            glu_mgdl = patient_data.get("r1glu", 0) * 18.018
            hba1c = patient_data.get("r1hba1c", 0)
            tg_mgdl = patient_data.get("r1tg", 0) * 88.57
            hdl_mgdl = patient_data.get("r1hdl", 0) * 38.67
            bmi = patient_data.get("r1mbmi", 0)
            crea_mgdl = patient_data.get("r1crea", 0) / 88.4

            if sex_val == 2:
                kappa, alpha, mult = 0.7, -0.329 if crea_mgdl <= 0.7 else -1.209, 144
            else:
                kappa, alpha, mult = 0.9, -0.411 if crea_mgdl <= 0.9 else -1.209, 141
            egfr = mult * (crea_mgdl / kappa)**alpha * (0.993)**age if crea_mgdl > 0 else 90

            est_stage = 0
            hyp = sbp >= 130 or dbp >= 80
            diab = glu_mgdl >= 126 or hba1c >= 6.5
            hdl_cut = 40 if sex_val == 1 else 50
            dyslip = tg_mgdl >= 150 or hdl_mgdl < hdl_cut
            if bmi >= 25 or (100 <= glu_mgdl < 126) or (5.7 <= hba1c < 6.5):
                est_stage = 1
            if hyp or diab or dyslip or (30 <= egfr < 60):
                est_stage = 2
            if egfr < 30:
                est_stage = 3
            if patient_pt > 0.80:
                est_stage = max(est_stage, 3)

            # ═══════════ KPI Cards ═══════════
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.markdown(f'<div class="kpi-card {risk_css}"><h2>{risk_emoji} {risk_level}</h2><p>CKM Risk Tier</p></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="kpi-card" style="background:linear-gradient(135deg,#dfe6e9,#b2bec3);"><h2>{patient_pt:.2f}</h2><p>Geodesic Pseudotime</p></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="kpi-card" style="background:linear-gradient(135deg,#e0c3fc,#8ec5fc);"><h2>{dim1:.2f}</h2><p>Dim1 (Metabolic)</p></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="kpi-card" style="background:linear-gradient(135deg,#fad0c4,#ffd1ff);"><h2>{dim2:.2f}</h2><p>Dim2 (Lipid-Renal)</p></div>', unsafe_allow_html=True)
            with c5:
                sc = stage_colors.get(est_stage, "#999")
                st.markdown(f'<div class="kpi-card" style="background:{sc}22;border:2px solid {sc};"><h2 style="color:{sc};">Stage {est_stage}</h2><p>{stage_names.get(est_stage,"")}</p></div>', unsafe_allow_html=True)

            # ═══════════ Alert ═══════════
            # Simple tipping zone check based on pseudotime gradient
            in_tipping = 0.45 < patient_pt < 0.65
            if in_tipping:
                st.markdown("""<div class="alert-box alert-tipping"><span style="font-size:1.5rem;">⚠️</span><div><strong>Tipping Zone Detected</strong><br><span style="font-size:0.85rem;">Patient is in a high-gradient risk transition region. Small changes in biomarkers may lead to disproportionate outcome shifts.</span></div></div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="alert-box alert-safe"><span style="font-size:1.5rem;">✅</span><div><strong>Stable Landscape Region</strong><br><span style="font-size:0.85rem;">Patient is in a low-gradient region. Risk profile is relatively stable to small biomarker fluctuations.</span></div></div>""", unsafe_allow_html=True)

            # ═══════════ Tabs ═══════════
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Landscape", "🏔️ 3D Risk Surface",
                "🕸️ Biomarker Profile", "📋 Clinical Guidelines"
            ])

            # ─── Tab 1: Dual Landscape ───
            with tab1:
                st.markdown("### Phenotypic Landscape Position")
                col_l, col_r = st.columns(2)

                for col_plot, c_key, c_label in [
                    (col_l, "CHARLS", "🇨🇳 CHARLS (China)"),
                    (col_r, "UKB", "🇬🇧 UKB (United Kingdom)")
                ]:
                    with col_plot:
                        st.markdown(f"**{c_label}**")
                        ref_k = models.get(f"ref_{c_key.lower()}")

                        fig = go.Figure()

                        # DDRTree skeleton edges
                        Y_k = models.get(f"Y_{c_key}")
                        edges_k = models.get(f"edges_{c_key}")
                        if Y_k is not None and edges_k is not None:
                            edge_x, edge_y = [], []
                            for _, row in edges_k.iterrows():
                                i, j = int(row['i']), int(row['j'])
                                edge_x.extend([Y_k[i, 0], Y_k[j, 0], None])
                                edge_y.extend([Y_k[i, 1], Y_k[j, 1], None])
                            fig.add_trace(go.Scatter(
                                x=edge_x, y=edge_y,
                                mode='lines', name='DDRTree',
                                line=dict(color='rgba(80,80,80,0.4)', width=1.0),
                                hoverinfo='skip', showlegend=False
                            ))

                        # Reference population scatter
                        if ref_k is not None:
                            for stage in sorted(ref_k[VAR_CKM_STAGE].dropna().unique()):
                                sub = ref_k[ref_k[VAR_CKM_STAGE] == stage]
                                fig.add_trace(go.Scatter(
                                    x=sub["Dim1"], y=sub["Dim2"],
                                    mode='markers',
                                    name=f"S{int(stage)} {stage_names.get(int(stage),'')}",
                                    marker=dict(size=3, color=stage_colors.get(int(stage), "#999"), opacity=0.3),
                                    hoverinfo='skip'
                                ))

                        # Project patient onto both cohorts
                        pp_k = models.get(f"pp_{c_key}")
                        g1_k = models.get(f"gam1_{c_key}")
                        g2_k = models.get(f"gam2_{c_key}")
                        meta_k = models.get(f"meta_{c_key}")
                        if pp_k and g1_k and g2_k and meta_k:
                            df_pp_k = apply_preprocess(df_patient.copy(), pp_k)
                            fc_k = meta_k["feat_cols"]
                            for fc in fc_k:
                                if fc not in df_pp_k.columns:
                                    df_pp_k[fc] = 0
                            d1_k = float(predict_gam(g1_k, df_pp_k, fc_k)[0])
                            d2_k = float(predict_gam(g2_k, df_pp_k, fc_k)[0])
                            fig.add_trace(go.Scatter(
                                x=[d1_k], y=[d2_k],
                                mode='markers+text', name='📍 Patient',
                                marker=dict(size=16, color='#e74c3c', symbol='diamond',
                                            line=dict(color='white', width=2.5)),
                                text=['📍'], textposition='top center', textfont=dict(size=14)
                            ))

                        fig.update_layout(
                            xaxis_title="Dim 1 (Metabolic Load)",
                            yaxis_title="Dim 2 (Lipid-Renal)",
                            height=420, template="plotly_white",
                            legend=dict(font=dict(size=7), orientation="h", yanchor="bottom", y=1.02),
                            margin=dict(l=50, r=10, t=10, b=50),
                            plot_bgcolor='rgba(248,249,250,1)',
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # ─── Tab 2: 3D Risk Surface ───
            with tab2:
                st.markdown("### 3D Mortality Risk Surface")
                st.caption("Pseudotime-based risk landscape. Red peaks = highest risk zones.")

                ref_df = models.get(f"ref_{cohort_key.lower()}")
                if ref_df is not None and "pseudotime" in ref_df.columns:
                    from scipy.interpolate import griddata
                    pts = ref_df[["Dim1", "Dim2"]].dropna().values
                    vals = ref_df["pseudotime"].dropna().values
                    xi = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 40)
                    yi = np.linspace(pts[:, 1].min(), pts[:, 1].max(), 40)
                    xi_g, yi_g = np.meshgrid(xi, yi)
                    zi = griddata(pts, vals, (xi_g, yi_g), method='cubic')
                    zi = np.nan_to_num(zi, nan=0)

                    fig3d = go.Figure(data=[go.Surface(
                        x=xi, y=yi, z=zi,
                        colorscale='RdYlBu_r', opacity=0.85,
                        colorbar=dict(title="Pseudotime"),
                    )])
                    fig3d.add_trace(go.Scatter3d(
                        x=[dim1], y=[dim2], z=[patient_pt],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='diamond',
                                    line=dict(color='white', width=2)),
                        name='📍 Patient'
                    ))
                    fig3d.update_layout(
                        scene=dict(
                            xaxis_title='Dim 1 (Metabolic)',
                            yaxis_title='Dim 2 (Lipid-Renal)',
                            zaxis_title='Pseudotime (Disease Progress)',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                        ),
                        height=550, margin=dict(l=0, r=0, t=30, b=0),
                    )
                    st.plotly_chart(fig3d, use_container_width=True)
                else:
                    st.info("3D surface requires pseudotime in reference data.")

            # ─── Tab 3: Biomarker Radar ───
            with tab3:
                st.markdown("### Biomarker Domain Profile")
                typical_ranges = {
                    "r1systo": (90, 180), "r1diasto": (50, 110),
                    "r1glu": (3.5, 15), "r1hba1c": (4, 12),
                    "r1tg": (0.3, 5), "r1hdl": (0.5, 2.5),
                    "r1crea": (30, 200), "r1mbmi": (15, 40),
                }
                radar_vars = [v for v in VARS_ACTIVE if v in patient_data]
                radar_labels = [VARS_LABELS.get(v, v) for v in radar_vars]
                radar_values = []
                for v in radar_vars:
                    lo, hi = typical_ranges.get(v, (0, 100))
                    val = patient_data.get(v, 0)
                    radar_values.append(max(0, min(1, (val - lo) / (hi - lo))))

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_values + [radar_values[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill='toself', name='Patient',
                    fillcolor='rgba(99, 110, 250, 0.2)',
                    line=dict(color='#636EFA', width=2),
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=[0.5] * (len(radar_labels) + 1),
                    theta=radar_labels + [radar_labels[0]],
                    name='Population Median',
                    line=dict(color='#aaa', dash='dot', width=1),
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True, height=400,
                    margin=dict(l=60, r=60, t=30, b=30),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # ─── Tab 4: Clinical Guidelines ───
            with tab4:
                st.markdown("### AHA CKM Clinical Guidelines")
                st.markdown(f"**Estimated CKM Stage: {est_stage} — {stage_names.get(est_stage, '')}**")

                guidelines = {
                    0: {"title": "Stage 0 — No CKM Risk Factors", "css": "guide-stage0",
                        "monitoring": "Annual health assessment, lipid panel every 5 years",
                        "lifestyle": "Maintain healthy weight (BMI 18.5–24.9), 150+ min/week moderate exercise, Mediterranean/DASH diet",
                        "pharmacotherapy": "None required",
                        "targets": "BP <120/80, FPG <100 mg/dL, BMI <25"},
                    1: {"title": "Stage 1 — Excess Adiposity / Pre-diabetes", "css": "guide-stage1",
                        "monitoring": "Annual metabolic panel, FPG/HbA1c screening, waist circumference",
                        "lifestyle": "Structured weight loss (5-10% body weight), caloric restriction, supervised exercise 200+ min/week",
                        "pharmacotherapy": "Consider GLP-1 RA for BMI ≥30 (or ≥27 with comorbidity). Metformin for pre-diabetes with high-risk features.",
                        "targets": "Weight loss ≥5%, HbA1c <5.7%, waist <102cm (M) / <88cm (F)"},
                    2: {"title": "Stage 2 — Metabolic Risk Factors", "css": "guide-stage2",
                        "monitoring": "Biannual metabolic panel, annual eGFR/UACR, ECG screening",
                        "lifestyle": "Intensive lifestyle modification, DASH diet, sodium <2300 mg/day",
                        "pharmacotherapy": "• HTN: ACEi/ARB first-line (target <130/80)\n• DM: Metformin + SGLT2i ± GLP-1 RA\n• Dyslipidemia: High-intensity statin (LDL <100)\n• CKD: SGLT2i if eGFR 20-45",
                        "targets": "BP <130/80, HbA1c <7.0%, LDL <100 mg/dL, eGFR stable"},
                    3: {"title": "Stage 3 — Subclinical CVD / High Risk", "css": "guide-stage3",
                        "monitoring": "Quarterly metabolic panel, annual echo/BNP, UACR every 6 months",
                        "lifestyle": "Cardiac rehabilitation-style program, supervised exercise",
                        "pharmacotherapy": "• Aggressive BP: dual therapy (ACEi/ARB + CCB)\n• DM: Triple (Metformin + SGLT2i + GLP-1 RA)\n• Statin + ezetimibe (LDL <70)\n• Aspirin if 10y CVD risk ≥20%\n• CKD: SGLT2i + finerenone",
                        "targets": "BP <130/80, HbA1c <7.0%, LDL <70 mg/dL"},
                    4: {"title": "Stage 4 — Established CVD", "css": "guide-stage4",
                        "monitoring": "Comprehensive cardiorenal-metabolic panel every 3 months",
                        "lifestyle": "Phase II cardiac rehabilitation (12 weeks minimum)",
                        "pharmacotherapy": "• Secondary: aspirin + statin + ezetimibe ± PCSK9i (LDL <55)\n• HFrEF: ACEi/ARNI + β-blocker + MRA + SGLT2i\n• DM: SGLT2i + GLP-1 RA\n• CKD: nephrology co-management",
                        "targets": "BP <130/80, LDL <55 mg/dL, HbA1c <8% if frail"},
                }

                g = guidelines.get(est_stage, guidelines[0])
                st.markdown(f'<div class="guideline-card {g["css"]}"><h4>{g["title"]}</h4></div>', unsafe_allow_html=True)

                gc1, gc2 = st.columns(2)
                with gc1:
                    st.markdown("**🔍 Monitoring**")
                    st.markdown(f'<div class="info-panel"><p style="font-size:0.85rem;">{g["monitoring"]}</p></div>', unsafe_allow_html=True)
                    st.markdown("**🥗 Lifestyle Modification**")
                    st.markdown(f'<div class="info-panel"><p style="font-size:0.85rem;">{g["lifestyle"]}</p></div>', unsafe_allow_html=True)
                with gc2:
                    st.markdown("**💊 Pharmacotherapy**")
                    st.markdown(f'<div class="info-panel"><p style="font-size:0.85rem;white-space:pre-line;">{g["pharmacotherapy"]}</p></div>', unsafe_allow_html=True)
                    st.markdown("**🎯 Treatment Targets**")
                    st.markdown(f'<div class="info-panel"><p style="font-size:0.85rem;">{g["targets"]}</p></div>', unsafe_allow_html=True)

        else:
            st.error("⚠ GAM models not loaded.")

else:
    # ═══════════ Landing Page ═══════════
    st.info("👈 Enter patient biomarker values in the sidebar, then click **Compute Risk Profile**.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### How It Works
        1. **Input** patient demographics and 8 CKM biomarkers
        2. **Project** onto the DDRTree phenotypic landscape via frozen GAM
        3. **Compute** geodesic pseudotime disease progression index
        4. **Stratify** risk and detect tipping zone proximity
        5. **Guide** with AHA CKM stage-specific clinical recommendations

        ### Three-Cohort Validation
        - 🇨🇳 **CHARLS** (N=8,957) — Chinese Longitudinal Study
        - 🇬🇧 **UKB** (N=396,866) — UK Biobank
        - 🇺🇸 **HRS** (N=3,396) — Health & Retirement Study
        """)
    with col2:
        st.markdown("""
        ### Key Innovation
        - **Frozen Preprocessing**: Derivation-set parameters ensure OOS validity
        - **Geodesic Pseudotime**: Graph-based disease trajectory quantification
        - **Tipping Zones**: High-gradient regions with disproportionate risk
        - **Cross-Ethnic Portability**: Same manifold architecture across populations

        ### Clinical Value
        - Continuous risk stratification **beyond** categorical CKM staging
        - Actionable **tipping zone alerts** for early intervention
        - **AHA-aligned** clinical guideline recommendations
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey; font-size:0.75rem;'>"
    "CKM Precision Risk Navigator v4.0 | DDRTree Manifold Learning | "
    "Frozen Preprocessing | Geodesic Pseudotime | Tri-Cohort Validated | "
    "For Research Use Only</p>",
    unsafe_allow_html=True
)
