
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Get user input for each loan applicant detail
credit_score = int(input("Enter credit score: "))
annual_income = float(input("Enter annual income: "))
loan_amount = float(input("Enter loan amount: "))
loan_term_years = int(input("Enter loan term in years: "))

# Create the dictionary with the user-provided values
new_applicant = {
    'credit_score': credit_score,
    'annual_income': annual_income,
    'loan_amount': loan_amount,
    'loan_term_years': loan_term_years
}

# You can now print the dictionary to verify the values
print("\nNew Applicant Data:")
print(new_applicant)

try:
    loaded_model = joblib.load('credit_risk_model.joblib')
    print("Model loaded successfully for real-time predictions.")
except FileNotFoundError:
    print("Error: Model file not found. Please train and save the model first.")

def assess_credit_risk(applicant_data):
    # The applicant_data would be a dictionary from an API request, e.g.:
    # {'credit_score': 720, 'annual_income': 85000, 'loan_amount': 20000, 'loan_term_years': 4}
    
    # Convert the new data to a format the model expects (a DataFrame or 2D array)
    df_new = pd.DataFrame([applicant_data])
    
    # Make the prediction
    prediction = loaded_model.predict(df_new)
    
    # The prediction is 0 for 'No Default' and 1 for 'Default'
    if prediction[0] == 0:
        return "Low Risk: Approve Loan"
    else:
        return "High Risk: Reject Loan"

# To test this function, you could call it with a new data point:
# new_applicant = {'credit_score': 700, 'annual_income': 75000, 'loan_amount': 30000, 'loan_term_years': 3}
print("\nAssessing a new applicant...")
print(f"Applicant Data: {new_applicant}")
# For this example, we'll just show the concept, not run it.
prediction = assess_credit_risk(new_applicant)
print(f"Prediction: {prediction}")
