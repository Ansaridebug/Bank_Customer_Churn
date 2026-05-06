import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Bank Churn Prediction", layout="wide", page_icon="🏦")

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #1a73e8;
        color: white;
        font-weight: bold;
        padding: 0.6rem;
        border-radius: 10px;
        font-size: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .churn {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef9a9a;
    }
    .no-churn {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #a5d6a7;
    }
    </style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained RandomForest model saved with joblib."""
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Churn_Prediction.pkl")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}\n\nMake sure 'Churn_Prediction.pkl' is in the same folder as app.py.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ── Page header ────────────────────────────────────────────────────────────────
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Predict whether a customer will **leave the bank** using a trained Random Forest model.")
st.markdown("---")

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("📋 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    year = st.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)

with col2:
    balance = st.number_input("Account Balance (€)", min_value=0.0, max_value=300000.0, value=75000.0, step=100.0)
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card?", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])
    is_active = st.selectbox("Is Active Member?", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])

with col3:
    salary = st.number_input("Estimated Salary (€)", min_value=0.0, max_value=250000.0, value=50000.0, step=500.0)
    geography = st.selectbox("Geography", options=["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", options=["Female", "Male"])

# ── Encode categorical inputs ──────────────────────────────────────────────────
geography_germany = 1 if geography == "Germany" else 0
geography_spain   = 1 if geography == "Spain"   else 0
gender_male       = 1 if gender == "Male"        else 0
has_cr_card_val   = has_cr_card[1]
is_active_val     = is_active[1]

# Feature order must match the training data exactly:
# Year, CreditScore, Age, Tenure, Balance, NumOfProducts,
# HasCrCard, IsActiveMember, EstimatedSalary,
# Geography_Germany, Geography_Spain, Gender_Male
user_input = np.array([[
    year, credit_score, age, tenure, balance, num_products,
    has_cr_card_val, is_active_val, salary,
    geography_germany, geography_spain, gender_male
]])

# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Predict Churn"):
    prediction      = model.predict(user_input)[0]
    prediction_prob = model.predict_proba(user_input)[0]

    churn_prob    = round(prediction_prob[1] * 100, 2)
    no_churn_prob = round(prediction_prob[0] * 100, 2)

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if prediction == 1:
            st.markdown(
                f'<div class="result-box churn">⚠️ Customer is likely to CHURN<br>'
                f'<small>Churn probability: {churn_prob}%</small></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-box no-churn">✅ Customer is likely to STAY<br>'
                f'<small>Retention probability: {no_churn_prob}%</small></div>',
                unsafe_allow_html=True
            )

    st.markdown("#### 📊 Prediction Confidence")
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.metric("🟢 Will Stay", f"{no_churn_prob}%")
    with prob_col2:
        st.metric("🔴 Will Churn", f"{churn_prob}%")

    st.markdown("---")
    st.markdown("#### 🧾 Input Summary")
    summary = {
        "Year": year, "Credit Score": credit_score, "Age": age, "Tenure": tenure,
        "Balance (€)": balance, "Num of Products": num_products,
        "Has Credit Card": "Yes" if has_cr_card_val else "No",
        "Is Active Member": "Yes" if is_active_val else "No",
        "Estimated Salary (€)": salary,
        "Geography": geography, "Gender": gender
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(summary, index=["Value"]).T, use_container_width=True)

st.markdown("---")
st.caption("Model: Random Forest Classifier | Dataset: European Bank Customer Data")