# train_and_save_model.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------- CONFIG --------------
# Put CSV in same folder OR give absolute path here:
CSV_PATH = "Loan_Prediction_DataSet.csv"   # <-- put your actual filename here (or full path)
MODEL_FILENAME = "loan_model.sav"
COLUMNS_FILENAME = "model_columns.pkl"
RANDOM_STATE = 42
# ------------------------------------

# 1) check file exists
if not os.path.exists(CSV_PATH):
    print("ERROR: CSV file not found at:", CSV_PATH)
    print("Files in current directory:", os.listdir("."))
    raise SystemExit("Put the CSV in the same folder or update CSV_PATH.")

# 2) load data
df = pd.read_csv(CSV_PATH)
print("Raw data shape:", df.shape)
print(df.head())

# 3) basic cleaning & consistent encoding (must match what your app expects)
# - Convert '3+' to 3 in Dependents, convert types
df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = pd.to_numeric(df['Dependents'], errors='coerce')

# Fill missing values:
# numeric medians
num_cols = ['Dependents', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for c in num_cols:
    if c in df.columns:
        df[c].fillna(df[c].median(), inplace=True)

# categorical modes
cat_cols = ['Gender', 'Married', 'Self_Employed', 'Education', 'Property_Area']
for c in cat_cols:
    if c in df.columns:
        df[c].fillna(df[c].mode()[0], inplace=True)

# 4) encode categorical values exactly as the app expects
df.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
df.replace({"Gender": {'Male': 1, 'Female': 0}}, inplace=True)
df.replace({"Married": {'Yes': 1, 'No': 0}}, inplace=True)
df.replace({"Education": {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
df.replace({"Self_Employed": {'Yes': 1, 'No': 0}}, inplace=True)
df.replace({"Property_Area": {'Urban': 2, 'Semiurban': 1, 'Rural': 0}}, inplace=True)

# make sure LoanAmount is numeric (sometimes dataset in thousands etc.)
if 'LoanAmount' in df.columns:
    df['LoanAmount'] = pd.to_numeric(df['LoanAmount'], errors='coerce').fillna(df['LoanAmount'].median())

# 5) choose features in the same ORDER your app uses
feature_cols = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area'
]

# verify all feature columns exist
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing expected columns from CSV: {missing}")

X = df[feature_cols]
y = df['Loan_Status']

print("Features shape:", X.shape, "Target shape:", y.shape)

# 6) split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 7) train model (RandomForest is stable without scaling)
model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# 8) evaluate quickly
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Train accuracy:", accuracy_score(y_train, train_pred))
print("Test accuracy:", accuracy_score(y_test, test_pred))
print("\nClassification report (test):\n", classification_report(y_test, test_pred))
print("\nConfusion matrix (test):\n", confusion_matrix(y_test, test_pred))

# 9) save model and the feature order (so app and model agree)
pickle.dump(model, open(MODEL_FILENAME, "wb"))
pickle.dump(feature_cols, open(COLUMNS_FILENAME, "wb"))

print(f"\nSaved model to {MODEL_FILENAME}")
print(f"Saved feature list to {COLUMNS_FILENAME}")

# 10) quick smoke-test: predict using the model and a sample row
sample = np.array([1, 0, 0, 1, 0, 5849, 0, 146, 360, 1.0, 2]).reshape(1, -1)
print("Sample input shape:", sample.shape)
print("Sample prediction (0=Not Approved, 1=Approved):", model.predict(sample))
