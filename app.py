import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to leave the bank based on their profile.")
st.markdown("---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model = joblib.load(os.path.join(BASE_DIR, 'rf_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    return model, scaler

model, scaler = load_assets()

# Sidebar inputs
st.sidebar.header("👤 Customer Profile")
geography_map = {"France": 0, "Germany": 1, "Spain": 2}
geography = st.sidebar.selectbox("Geography", list(geography_map.keys()))
gender_map = {"Female": 0, "Male": 1}
gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
card_type_map = {"DIAMOND": 0, "GOLD": 1, "PLATINUM": 2, "SILVER": 3}
card_type = st.sidebar.selectbox("Card Type", list(card_type_map.keys()))

st.sidebar.subheader("Financial & Demographic Details")
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance ($)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
estimated_salary = st.sidebar.number_input("Estimated Salary ($)", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
satisfaction_score = st.sidebar.slider("Satisfaction Score", 1, 5, 3)
point_earned = st.sidebar.number_input("Points Earned", min_value=0, value=500, step=10)
has_cr_card = 1 if st.sidebar.selectbox("Has Credit Card?", ["Yes", "No"]) == "Yes" else 0
is_active_member = 1 if st.sidebar.selectbox("Is Active Member?", ["Yes", "No"]) == "Yes" else 0

geo_encoded = geography_map[geography]
gender_encoded = gender_map[gender]
card_encoded = card_type_map[card_type]

# --- BULLETPROOF FEATURE ALIGNMENT ---
# Dynamically check how many features the scaler expects
expected_features = scaler.n_features_in_

# Build the feature list in the exact order of the training data
features_list = [
    credit_score, geo_encoded, gender_encoded, age, tenure, 
    balance, num_of_products, has_cr_card, is_active_member, estimated_salary
]

# If the scaler expects 14 features, it means the 'Complain' column was included during training.
# We inject a dummy '0' for 'Complain' at the exact correct index to match the scaler.
if expected_features == 14:
    features_list.append(0) 

features_list.extend([satisfaction_score, card_encoded, point_earned])

# Convert to a 2D numpy array (bypasses all pandas feature-name checking)
input_array = np.array(features_list).reshape(1, -1)

# Predict button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Churn", use_container_width=True)

if predict_button:
    # Scale using the raw numpy array
    scaled_input = scaler.transform(input_array)
    
    # Predict
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    
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
