# 🫀 CKM Precision Risk Navigator

**Data-Driven Phenotypic Landscape for Cardiovascular-Kidney-Metabolic Syndrome Risk Stratification**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## Overview

A clinical decision-support tool that maps individual patients onto the CKM phenotypic landscape using DDRTree manifold learning. Validated across three international cohorts:

| Cohort | N | Population | Role |
|--------|------|------------|------|
| 🇨🇳 CHARLS | 8,957 | Chinese | Derivation |
| 🇬🇧 UK Biobank | 396,866 | European | External Validation |
| 🇺🇸 HRS | 3,396 | American | External Validation |

## Features

- **Dual-Cohort Model Switching**: China (CHARLS) / UK (UKB) population models
- **Interactive DDRTree Landscape**: Tree skeleton with CKM stage-colored scatter
- **3D Risk Surface**: Pseudotime-based mortality risk visualization with patient positioning
- **Geodesic Pseudotime**: Graph-based disease progression quantification (0→1)
- **Tipping Zone Detection**: High-gradient transition region alerts
- **AHA Clinical Guidelines**: Stage-specific (0-4) monitoring, lifestyle, and pharmacotherapy recommendations
- **Biomarker Radar Chart**: Multi-domain profile visualization

## 8 Core CKM Biomarkers

| Domain | Variable | Unit |
|--------|----------|------|
| Hemodynamic | SBP, DBP | mmHg |
| Glycemic | Glucose, HbA1c | mmol/L, % |
| Lipid | TG, HDL-C | mmol/L |
| Renal | Creatinine | μmol/L |
| Adiposity | BMI | kg/m² |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `streamlit_app.py`
5. Deploy!

## File Structure

```
ckm_navigator/
├── streamlit_app.py          # Main application (standalone)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .streamlit/
│   └── config.toml           # Theme configuration
├── models/                   # Frozen model files
│   ├── gam_dim1_CHARLS.pkl   # GAM projection models
│   ├── gam_dim2_CHARLS.pkl
│   ├── gam_meta_CHARLS.pkl
│   ├── gam_dim1_UKB.pkl
│   ├── gam_dim2_UKB.pkl
│   ├── gam_meta_UKB.pkl
│   ├── preprocess_params_CHARLS.pkl  # Frozen RINT + residualization
│   ├── preprocess_params_UKB.pkl
│   ├── ddrtree_CHARLS/
│   │   ├── Y_nodes.csv       # Tree node coordinates
│   │   └── edges.csv         # Tree edge list
│   └── ddrtree_UKB/
│       ├── Y_nodes.csv
│       └── edges.csv
└── data/                     # Reference population samples
    ├── ref_charls.csv        # 3000 representative samples
    └── ref_ukb.csv
```

## Methodology

1. **Frozen Preprocessing**: RINT normalization + age/sex residualization using derivation-set parameters
2. **GAM Projection**: Maps 8 biomarkers → 2D manifold coordinates (Dim1, Dim2)
3. **Geodesic Pseudotime**: Shortest-path distance on DDRTree spanning tree from CKM Stage 0 root
4. **Risk Stratification**: Pseudotime quartiles (Low/Moderate/High/Very High)
5. **CKM Staging**: AHA-based algorithm for clinical guideline matching

## Citation

> DDRTree Manifold-Based CKM Risk Stratification: A Tri-Cohort Validation Study

## Disclaimer

⚠️ **For Research Use Only.** This tool is not a substitute for clinical judgment. Always consult qualified healthcare professionals for medical decisions.
