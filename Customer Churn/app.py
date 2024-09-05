import joblib
import streamlit as st
import pandas as pd

model = joblib.load("Churn_model.pkl")
scaler = joblib.load("scaler.pkl")
enc = joblib.load("label_enc.pkl")

st.title("Churn Model")

gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.number_input('Tenure', min_value=0)
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input('Monthly Charges', min_value=0.0)

input_data = pd.DataFrame({
    'gender': [enc["gender"].transform([gender])[0]],
    'SeniorCitizen': [1 if senior_citizen == 'Yes' else 0],
    'Partner': [1 if partner == 'Yes' else 0],
    'Dependents': [1 if dependents == 'Yes' else 0],
    'tenure': [tenure],
    'PhoneService': [1 if phone_service == 'Yes' else 0],
    'MultipleLines': [enc["MultipleLines"].transform([multiple_lines])[0]],
    'InternetService': [enc["InternetService"].transform([internet_service])[0]],
    'OnlineSecurity': [enc["OnlineSecurity"].transform([online_security])[0]],
    'OnlineBackup': [enc["OnlineBackup"].transform([online_backup])[0]],
    'DeviceProtection': [enc["DeviceProtection"].transform([device_protection])[0]],
    'TechSupport': [enc["TechSupport"].transform([tech_support])[0]],
    'StreamingTV': [enc["StreamingTV"].transform([streaming_tv])[0]],
    'StreamingMovies': [enc["StreamingMovies"].transform([streaming_movies])[0]],
    'Contract': [enc["Contract"].transform([contract])[0]],
    'PaperlessBilling': [1 if paperless_billing == 'Yes' else 0],
    'PaymentMethod': [enc["PaymentMethod"].transform([payment_method])[0]],
    'MonthlyCharges': [monthly_charges],
})
input_data = scaler.transform(input_data)

pred = model.predict(input_data)
pred_prob = model.predict_proba(input_data)

st.write('Churn Prediction:', 'Yes' if pred[0] else 'No')
st.write('Prediction Probability:', pred_prob)