"""
CHURNGUARD AI — Customer Risk Predictor
Optional Advanced Challenge: simple Streamlit web interface for the
churn model trained in the CHURNGUARD AI notebook.

How to run:
    1. Make sure "churn_guard_final_model.pkl" (saved with joblib.dump
       in the notebook) is in the same folder as this file.
    2. pip install streamlit pandas scikit-learn joblib
    3. streamlit run app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="CHURNGUARD AI", page_icon="📉", layout="wide")

st.title("📉 CHURNGUARD AI")
st.subheader("Customer Risk Predictor")
st.write("Enter a customer's details to estimate their churn risk.")


# ----------------------------------------------------------------------
# Load the trained pipeline (preprocessing + model bundled together)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_guard_final_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file 'churn_guard_final_model.pkl' not found. "
        "Place it in the same folder as this app (it's created by the "
        "notebook's joblib.dump step)."
    )
    st.stop()


# Same thresholds used in the notebook's get_risk_level()
def get_risk_level(probability):
    if probability >= 0.70:
        return "HIGH"
    elif probability >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
with st.form("customer_form"):
    st.markdown("### Customer Profile")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col3:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col4:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges (₹)", min_value=0.0, value=60.0, step=1.0)
        total_charges = st.number_input(
            "Total Charges (₹)", min_value=0.0, value=round(tenure * monthly_charges, 2), step=1.0
        )

    submitted = st.form_submit_button("Predict Churn Risk")


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    customer_df = pd.DataFrame([customer_data])

    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]
    risk = get_risk_level(probability)

    st.markdown("---")
    st.markdown("### CHURNGUARD AI — Customer Risk Assessment")

    if prediction == 1:
        st.error(f"**Prediction:** HIGH CHURN RISK")
    else:
        st.success(f"**Prediction:** Customer likely to stay")

    st.metric("Churn Probability", f"{probability * 100:.1f}%")

    risk_colors = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
    st.write(f"**Risk Level:** {risk_colors[risk]} {risk}")

    if risk == "HIGH":
        st.info("Suggested Action: Review customer for possible retention intervention.")
    elif risk == "MEDIUM":
        st.info("Suggested Action: Monitor customer; consider a proactive check-in.")
    else:
        st.info("Suggested Action: No immediate action needed.")