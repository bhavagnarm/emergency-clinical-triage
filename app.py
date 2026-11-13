import os
import streamlit as st
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Emergency Clinical Triage CDSS", page_icon="🏥", layout="wide")

COLUMN_MAPPING = {
    "age": "Age",
    "heart_rate": "HR",
    "systolic_bp": "SBP",
    "diastolic_bp": "DBP",
    "spo2": "Saturation",
    "temperature_c": "BT",
    "pain_score": "Pain",
    "target_acuity": "KTAS_expert"
}

@st.cache_resource
def train_triage_model():
    raw_df = pd.read_csv("data.csv", sep=";", encoding="latin-1")
    df = pd.DataFrame()
    for std_col, act_col in COLUMN_MAPPING.items():
        df[std_col] = pd.to_numeric(raw_df[act_col], errors="coerce") if act_col in raw_df.columns else np.nan

    df = df.dropna(subset=["target_acuity"])
    df["target_acuity"] = df["target_acuity"].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))

    # Clinical Domain Feature Engineering
    df["shock_index"] = df["heart_rate"] / df["systolic_bp"].replace(0, np.nan)
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["map"] = df["diastolic_bp"] + (df["pulse_pressure"] / 3.0)
    df["temp_deviation"] = (df["temperature_c"] - 37.0).abs()

    feature_cols = [c for c in df.columns if c != "target_acuity"]
    X = df[feature_cols]
    y = df["target_acuity"].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.03,
        class_weight="balanced", subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, explainer

model, feature_cols, explainer = train_triage_model()

st.title("🏥 AI-Assisted Emergency Clinical Triage & Safe Dispensary System")
st.markdown("**Hybrid Architecture:** Balanced LightGBM + Clinical Physiological Biomarkers + Deterministic Safety Interlocks")

st.sidebar.header("📋 Patient Intake Vitals")
patient_name = st.sidebar.text_input("Patient Name", value="John Doe")
age = st.sidebar.number_input("Age (Years)", min_value=1, max_value=110, value=68)
pain = st.sidebar.slider("Pain Score (0 - 10)", min_value=0, max_value=10, value=8)

hr = st.sidebar.number_input("Heart Rate (BPM)", min_value=20.0, max_value=220.0, value=124.0)
sbp = st.sidebar.number_input("Systolic BP (mmHg)", min_value=40.0, max_value=260.0, value=86.0)
dbp = st.sidebar.number_input("Diastolic BP (mmHg)", min_value=20.0, max_value=160.0, value=54.0)
spo2 = st.sidebar.number_input("SpO2 Saturation (%)", min_value=50.0, max_value=100.0, value=91.0)
temp = st.sidebar.number_input("Body Temperature (°C)", min_value=32.0, max_value=42.5, value=38.6, step=0.1)

if st.sidebar.button("Run Clinical Triage Assessment", type="primary"):
    pp = sbp - dbp
    si = hr / sbp if sbp > 0 else np.nan
    map_val = dbp + (pp / 3.0)
    temp_dev = abs(temp - 37.0)

    input_df = pd.DataFrame([{
        "age": age, "heart_rate": hr, "systolic_bp": sbp, "diastolic_bp": dbp,
        "spo2": spo2, "temperature_c": temp, "pain_score": pain,
        "shock_index": si, "pulse_pressure": pp, "map": map_val, "temp_deviation": temp_dev
    }])[feature_cols]

    probs = model.predict_proba(input_df)[0]
    pred_tier = int(np.argmax(probs))

    is_extreme = (
        (spo2 < 90.0) or (sbp > 200.0 or sbp < 80.0) or
        (hr > 140.0 or hr < 40.0) or (si >= 1.0)
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Shock Index (HR/SBP)", f"{si:.2f}", delta="High Risk" if si >= 0.9 else "Normal", delta_color="inverse")
    col2.metric("Mean Arterial Pressure", f"{map_val:.1f} mmHg")
    col3.metric("Pulse Pressure", f"{pp:.0f} mmHg")
    col4.metric("Temp Deviation", f"{temp_dev:.1f} °C")

    st.divider()

    if (pred_tier == 0 and probs[0] >= 0.80) or is_extreme:
        st.error(f"🚨 **TRIAGE RESULT FOR {patient_name.upper()}: CRITICAL (Level 0 — KTAS 1–2)**")
        st.warning("🔒 **SAFETY INTERLOCK ACTIVATED:** Automated medicine dispensary LOCKED. Rapid Response Emergency Unit dispatched immediately.")
    else:
        tier_label = "URGENT (Level 1 — KTAS 3)" if pred_tier == 1 else "NON-URGENT (Level 2 — KTAS 4–5)"
        st.success(f"✅ **TRIAGE RESULT FOR {patient_name.upper()}: {tier_label}**")
        if pain >= 5.0 or temp >= 38.0:
            rx = "Ibuprofen 400mg"
        elif pain >= 1.0:
            rx = "Acetaminophen 500mg"
        else:
            rx = "Cetirizine 10mg"
        st.info(f"💊 **DISPENSARY APPROVED:** Dispensing **{rx}** from local automated dispensary unit.")

    left, right = st.columns(2)
    with left:
        st.subheader("Acuity Class Probabilities")
        prob_df = pd.DataFrame({
            "Acuity Tier": ["Critical (0)", "Urgent (1)", "Non-Urgent (2)"],
            "Probability": probs
        })
        st.bar_chart(prob_df.set_index("Acuity Tier"))

    with right:
        st.subheader("Patient-Level TreeSHAP Feature Attribution")
        sv = explainer.shap_values(input_df)
        target_idx = 0 if ((pred_tier == 0 and probs[0] >= 0.80) or is_extreme) else pred_tier
        vals = sv[target_idx][0] if isinstance(sv, list) else (sv[0, :, target_idx] if sv.ndim == 3 else sv[0])
        shap_series = pd.Series(vals, index=feature_cols).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        shap_series.plot(kind="barh", color=["#dc2626" if v > 0 else "#2563eb" for v in shap_series], ax=ax)
        ax.set_xlabel("SHAP Impact on Predicted Class")
        plt.tight_layout()
        st.pyplot(fig)
