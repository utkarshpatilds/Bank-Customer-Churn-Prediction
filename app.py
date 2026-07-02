import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")

# Title
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to leave the bank based on their profile.")
st.markdown("---")

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(BASE_DIR, 'rf_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    return model, scaler

model, scaler = load_assets()

# Sidebar for user input
st.sidebar.header("👤 Customer Profile")

# Categorical mappings (Alphabetical order as per LabelEncoder)
geography_map = {"France": 0, "Germany": 1, "Spain": 2}
geography = st.sidebar.selectbox("Geography", list(geography_map.keys()))

gender_map = {"Female": 0, "Male": 1}
gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))

card_type_map = {"DIAMOND": 0, "GOLD": 1, "PLATINUM": 2, "SILVER": 3}
card_type = st.sidebar.selectbox("Card Type", list(card_type_map.keys()))

# Numerical inputs
st.sidebar.subheader("Financial & Demographic Details")
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance ($)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
satisfaction_score = st.sidebar.slider("Satisfaction Score", 1, 5, 3)
point_earned = st.sidebar.number_input("Points Earned", min_value=0, value=500, step=10)

# Binary inputs
has_cr_card = 1 if st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"]) == "Yes" else 0
is_active_member = 1 if st.sidebar.selectbox("Is Active Member?", ["Yes", "No"]) == "Yes" else 0

# Map categorical variables
geo_encoded = geography_map[geography]
gender_encoded = gender_map[gender]
card_encoded = card_type_map[card_type]

# Create input dataframe
# The order of columns MUST match the training data exactly
input_data = {
    'CreditScore': credit_score,
    'Geography': geo_encoded,
    'Gender': gender_encoded,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary,
    'Satisfaction Score': satisfaction_score,
    'Card Type': card_encoded,
    'Point Earned': point_earned
}

input_df = pd.DataFrame([input_data])

# Predict button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Churn", use_container_width=True)

if predict_button:
    # FIX: Convert to numpy array (.values) to bypass strict scikit-learn feature name checking
    scaled_input = scaler.transform(input_df.values)
    
    # Predict
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    
    # Display result
    st.markdown("---")
    st.subheader("📊 Prediction Result")
    
    if prediction[0] == 1:
        st.error("🚨 **High Risk:** The customer is likely to LEAVE the bank.")
    else:
        st.success("✅ **Low Risk:** The customer is likely to STAY with the bank.")
        
    st.markdown("### Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Outcome': ['Staying', 'Leaving'],
        'Probability': [f"{prediction_proba[0][0]*100:.2f}%", f"{prediction_proba[0][1]*100:.2f}%"]
    })
    st.dataframe(prob_df, hide_index=True, use_container_width=True)
    
    with st.expander("View Input Data"):
        st.dataframe(input_df, hide_index=True, use_container_width=True)
