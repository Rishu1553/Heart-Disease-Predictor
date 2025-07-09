import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load('end-to-end-heart-disease-classification.pkl')

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Title
st.title("💓 Heart Disease Predictor")
st.markdown(
    "This simple ML app predicts whether a patient may have heart disease based on medical data. "
    "Fill in the details below and click **Predict**."
)

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.slider("Chest Pain Type (0-3)", 0, 3, 0)
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", [0, 1])

with col2:
    restecg = st.slider("Resting ECG (0-2)", 0, 2, 0)
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina (1 = yes, 0 = no)", [0, 1])
    oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0)
    slope = st.slider("Slope of ST Segment (0-2)", 0, 2, 1)
    ca = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)
    thal = st.slider("Thal: 1=Normal, 2=Fixed Defect, 3=Reversible Defect", 1, 3, 2)

st.markdown("---")

# Predict button
if st.button("💡 Predict"):
    sex_num = 1 if sex == "Male" else 0

    input_features = np.array([[age, sex_num, cp, trestbps, chol, fbs,
                                restecg, thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(input_features)[0]

    try:
        prob = model.predict_proba(input_features)[0][1]
    except:
        prob = None

    if prediction == 0:
        st.success("✅ No Heart Disease Detected.")
    else:
        st.error("⚠️ Risk of Heart Disease Detected!")

    if prob is not None:
        st.write(f"**Model Confidence:** {prob*100:.2f}%")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")

