import pickle
import streamlit as st
import numpy as np

# Load the trained model
model = pickle.load(open("loan_model.sav", "rb"))
columns = pickle.load(open("model_columns.pkl", "rb"))

# App title
st.title("🏦 Loan Approval Prediction System")

# Input fields for user
Gender = st.selectbox("Gender", ("Male", "Female"))
Married = st.selectbox("Married", ("Yes", "No"))
Dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))
Education = st.selectbox("Education", ("Graduate", "Not Graduate"))
Self_Employed = st.selectbox("Self Employed", ("Yes", "No"))
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term (in days)", min_value=0)
Credit_History = st.selectbox("Credit History", (1.0, 0.0))
Property_Area = st.selectbox("Property Area", ("Urban", "Rural", "Semiurban"))

# Convert categorical inputs to numeric like in training
gender_val = 1 if Gender == "Male" else 0
married_val = 1 if Married == "Yes" else 0
education_val = 1 if Education == "Graduate" else 0
self_employed_val = 1 if Self_Employed == "Yes" else 0

if Property_Area == "Urban":
    property_val = 2
elif Property_Area == "Semiurban":
    property_val = 1
else:
    property_val = 0

# Collect input into array
input_data = np.array([
    gender_val, married_val, int(Dependents.replace("3+", "3")),
    education_val, self_employed_val, ApplicantIncome,
    CoapplicantIncome, LoanAmount, Loan_Amount_Term,
    Credit_History, property_val
]).reshape(1, -1)

# Prediction button
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
