import joblib
import streamlit as st
import pandas as pd

model = joblib.load("Churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Churn Model")


gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.number_input('Tenure')
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
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
monthly_charges = st.number_input('Monthly Charges')
total_charges = st.number_input('Total Charges')
"""
gender_male = 1 if gender == 'Male' else 0
senior_citizen = 1 if senior_citizen == 'Yes' else 0
partner = 1 if partner == 'Yes' else 0
dependents_yes = 1 if dependents == 'Yes' else 0
phone_service_yes = 1 if phone_service == 'Yes' else 0
multiple_lines_yes = 1 if multiple_lines == 'Yes' else 0
internet_service_dsl = 1 if internet_service == 'DSL' else 0
internet_service_fiber = 1 if internet_service == 'Fiber optic' else 0
internet_service_no = 1 if internet_service == 'No' else 0
online_security_yes = 1 if online_security == 'Yes' else 0
online_backup_yes = 1 if online_backup == 'Yes' else 0
device_protection_yes = 1 if device_protection == 'Yes' else 0
tech_support_yes = 1 if tech_support == 'Yes' else 0
streaming_tv_yes = 1 if streaming_tv == 'Yes' else 0
streaming_movies_yes = 1 if streaming_movies == 'Yes' else 0
contract_one_year = 1 if contract == 'One year' else 0
contract_two_year = 1 if contract == 'Two year' else 0
paperless_billing_yes = 1 if paperless_billing == 'Yes' else 0
payment_method_electronic = 1 if payment_method == 'Electronic check' else 0
payment_method_mailed = 1 if payment_method == 'Mailed check' else 0
payment_method_bank = 1 if payment_method == 'Bank transfer (automatic)' else 0
payment_method_credit = 1 if payment_method == 'Credit card (automatic)' else 0
"""
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'gender_Male': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService_DSL': [internet_service],
    'InternetService_Fiber optic': [internet_service],
    'InternetService_No': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract_One year': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod_Electronic check': [payment_method],
    'PaymentMethod_Mailed check': [payment_method],
    'PaymentMethod_Bank transfer (automatic)': [payment_method],
    'PaymentMethod_Credit card (automatic)': [payment_method]
})

input_data = pd.get_dummies(input_data)

pred = model.predict(input_data)
pred_prob = model.predict_proba(input_data)

st.write('Churn Prediction:', 'Yes' if pred[0] else 'No')
st.write('Prediction Probability:', pred_prob)