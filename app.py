import streamlit as st
import numpy as np
import joblib

# ====== Page Config ======
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
)

# ====== Load Model ======
@st.cache_resource
def load_model():
    return joblib.load('end-to-end-heart-disease-classification.pkl')

model = load_model()

# ====== Title & Intro ======
st.title("💓 Heart Disease Predictor")
st.markdown(
    """
    This interactive **machine learning web app** predicts whether a patient might have **heart disease**
    based on key medical parameters.
    
    Fill out the form below and click **Predict** to see the result.
    """
)

# ====== Input Form ======
with st.form("input_form"):
    st.header("📝 Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 30)
        sex = st.radio("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.radio("Fasting Blood Sugar > 120 mg/dl?", [0, 1])

    with col2:
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
        exang = st.radio("Exercise Induced Angina?", [0, 1])
        oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0)
        slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [1, 2, 3])

    submitted = st.form_submit_button("💡 Predict")

# ====== Prediction ======
if submitted:
    sex_num = 1 if sex == "Male" else 0

    features = np.array([[age, sex_num, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]])

    prediction = model.predict(features)[0]

    try:
        prob = model.predict_proba(features)[0][1]
    except AttributeError:
        prob = None

    st.subheader("🔍 Result")
    if prediction == 0:
        st.success("✅ **No Heart Disease Detected.**")
    else:
        st.error("⚠️ **Risk of Heart Disease Detected!**")

    if prob is not None:
        st.info(f"**Model Confidence:** {prob * 100:.2f}%")

# ====== Footer ======
st.markdown("---")
st.caption("Made by Rishabh Joshi")
