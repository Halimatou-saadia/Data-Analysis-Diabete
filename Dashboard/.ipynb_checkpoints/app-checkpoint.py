import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import pickle
import os

# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="TontiTrack — Détection Diabète",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# CHEMINS DES FICHIERS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "projet1.csv")

# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================
@st.cache_resource
def charger_modele():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def charger_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

model = charger_modele()
scaler = charger_scaler()

# ============================================================
# TITRE ET INTRODUCTION
# ============================================================
st.title("🩺 Système de Détection Précoce du Diabète de Type 2")

st.markdown("""
> **Projet 01 — Analyse de Données Avancée avec Python**  
> Département Intelligence Artificielle · ADU 2025–2026  
> ⚠️ *Outil d'aide à la décision — ne remplace pas un diagnostic médical*
""")

st.divider()

# ============================================================
# SIDEBAR — SAISIE DES DONNÉES PATIENT
# ============================================================
st.sidebar.header("📋 Données du Patient")
st.sidebar.markdown("Renseignez les valeurs cliniques :")

pregnancies = st.sidebar.slider(
    "Nombre de grossesses",
    min_value=0,
    max_value=17,
    value=3
)

glucose = st.sidebar.slider(
    "Glucose plasmatique (mg/dL)",
    min_value=50,
    max_value=200,
    value=120,
    help="Valeur normale : 70-100 mg/dL à jeun"
)

blood_pressure = st.sidebar.slider(
    "Pression artérielle diastolique (mm Hg)",
    min_value=30,
    max_value=122,
    value=70
)

skin_thickness = st.sidebar.slider(
    "Épaisseur pli cutané triceps (mm)",
    min_value=5,
    max_value=99,
    value=23
)

insulin = st.sidebar.slider(
    "Insuline sérique 2h (mu U/ml)",
    min_value=15,
    max_value=846,
    value=80
)

bmi = st.sidebar.slider(
    "IMC — Indice de Masse Corporelle (kg/m²)",
    min_value=15.0,
    max_value=67.0,
    value=32.0,
    step=0.1
)

dpf = st.sidebar.slider(
    "DiabetesPedigreeFunction (hérédité)",
    min_value=0.07,
    max_value=2.42,
    value=0.47,
    step=0.01
)

age = st.sidebar.slider(
    "Âge (années)",
    min_value=21,
    max_value=81,
    value=33
)

# ============================================================
# PRÉDICTION
# ============================================================
patient_data = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]])

patient_scaled = scaler.transform(patient_data)

proba = model.predict_proba(patient_scaled)[0][1]
prediction = model.predict(patient_scaled)[0]

# ============================================================
# SECTION 1 — RÉSULTAT PRINCIPAL
# ============================================================
st.header("📊 Résultat de l'Analyse")

col1, col2, col3 = st.columns(3)

with col1:

    if prediction == 1:
        st.error("### ⚠️ Risque Élevé\n**Diabète probable**")
    else:
        st.success("### ✅ Risque Faible\n**Pas de diabète détecté**")

with col2:

    st.metric(
        label="Probabilité de diabète",
        value=f"{proba*100:.1f}%",
        delta=f"{(proba - 0.349)*100:+.1f}% vs moyenne dataset"
    )

with col3:

    niveau = (
        "Faible"
        if proba < 0.3
        else "Modéré"
        if proba < 0.6
        else "Élevé"
    )

    couleur = (
        "green"
        if proba < 0.3
        else "orange"
        if proba < 0.6
        else "red"
    )

    st.markdown(f"**Niveau de risque :** :{couleur}[**{niveau}**]")
    st.progress(float(proba))

st.divider()

# ============================================================
# SECTION 2 — JAUGE VISUELLE
# ============================================================
st.header("🎯 Jauge de Risque")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=proba * 100,

    delta={
        'reference': 34.9,
        'suffix': '%'
    },

    title={
        'text': "Probabilité de Diabète (%)"
    },

    gauge={
        'axis': {'range': [0, 100]},

        'bar': {'color': "darkred"},

        'steps': [
            {'range': [0, 30], 'color': '#90EE90'},
            {'range': [30, 60], 'color': '#FFD700'},
            {'range': [60, 100], 'color': '#FF6B6B'}
        ],

        'threshold': {
            'line': {
                'color': "black",
                'width': 4
            },
            'thickness': 0.75,
            'value': 50
        }
    }
))

fig_gauge.update_layout(height=350)

st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# ============================================================
# SECTION 3 — SHAP
# ============================================================
st.header("🔍 Facteurs Explicatifs (SHAP)")
st.markdown("*Pourquoi le modèle a-t-il fait cette prédiction ?*")

try:

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(patient_scaled)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    feature_names = [
        'Pregnancies',
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI',
        'DiabetesPedigreeFunction',
        'Age'
    ]

    feature_values = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]

    contrib_df = pd.DataFrame({
        'Variable': feature_names,
        'Valeur': feature_values,
        'SHAP': shap_vals
    })

    contrib_df = contrib_df.sort_values(
        'SHAP',
        key=abs,
        ascending=True
    )

    colors = [
        '#ff6b6b' if v > 0 else '#66b3ff'
        for v in contrib_df['SHAP']
    ]

    fig_shap = go.Figure(go.Bar(
        x=contrib_df['SHAP'],
        y=contrib_df.apply(
            lambda r: f"{r['Variable']} = {r['Valeur']}",
            axis=1
        ),
        orientation='h',
        marker_color=colors,
        text=contrib_df['SHAP'].round(3),
        textposition='outside'
    ))

    fig_shap.add_vline(
        x=0,
        line_dash='dash',
        line_color='black'
    )

    fig_shap.update_layout(
        title="Contribution des variables",
        xaxis_title="Impact sur le risque",
        height=450
    )

    st.plotly_chart(fig_shap, use_container_width=True)

except Exception as e:

    st.warning(f"Erreur SHAP : {e}")

st.divider()

# ============================================================
# SECTION 4 — COMPARAISON DATASET
# ============================================================
st.header("📈 Comparaison avec le Dataset")

data_ref = pd.read_csv(DATA_PATH)

col1, col2 = st.columns(2)

# ------------------------------------------------------------
# GLUCOSE
# ------------------------------------------------------------
with col1:

    fig_gluc = go.Figure()

    for outcome, color, label in [
        (0, '#66b3ff', 'Non-diabétique'),
        (1, '#ff6b6b', 'Diabétique')
    ]:

        fig_gluc.add_trace(go.Histogram(
            x=data_ref[data_ref['Outcome'] == outcome]['Glucose'],
            name=label,
            marker_color=color,
            opacity=0.6,
            nbinsx=20
        ))

    fig_gluc.add_vline(
        x=glucose,
        line_color='black',
        line_width=3,
        annotation_text=f"Patient ({glucose})",
        annotation_position="top"
    )

    fig_gluc.update_layout(
        title="Distribution du Glucose",
        barmode='overlay',
        height=350
    )

    st.plotly_chart(fig_gluc, use_container_width=True)

# ------------------------------------------------------------
# BMI
# ------------------------------------------------------------
with col2:

    fig_bmi = go.Figure()

    for outcome, color, label in [
        (0, '#66b3ff', 'Non-diabétique'),
        (1, '#ff6b6b', 'Diabétique')
    ]:

        fig_bmi.add_trace(go.Histogram(
            x=data_ref[data_ref['Outcome'] == outcome]['BMI'],
            name=label,
            marker_color=color,
            opacity=0.6,
            nbinsx=20
        ))

    fig_bmi.add_vline(
        x=bmi,
        line_color='black',
        line_width=3,
        annotation_text=f"Patient ({bmi})",
        annotation_position="top"
    )

    fig_bmi.update_layout(
        title="Distribution de l'IMC (BMI)",
        barmode='overlay',
        height=350
    )

    st.plotly_chart(fig_bmi, use_container_width=True)

st.divider()

# ============================================================
# SECTION 5 — AVERTISSEMENT
# ============================================================
st.warning("""
⚠️ **Avertissement éthique important**

Ce système est un outil d'aide à la décision entraîné sur des données
de femmes d'origine amérindienne âgées de 21 ans et plus.

Les prédictions peuvent ne pas être généralisables à d'autres populations.

Toute décision médicale doit être confirmée par un professionnel de santé qualifié.
""")