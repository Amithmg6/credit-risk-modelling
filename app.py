import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize the Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the model once when the application starts
try:
    loaded_model = joblib.load('credit_risk_model.joblib')
    print("Model loaded successfully for real-time predictions.")
except FileNotFoundError:
    loaded_model = None
    print("Error: Model file not found. Please train and save the model first.")

def assess_credit_risk_internal(applicant_data):
    """
    Assesses credit risk from a dictionary of applicant data.
    Returns "Low Risk: Approve Loan" or "High Risk: Reject Loan".
    """
    if loaded_model is None:
        return "Error: Model not available"

    # Convert the new data to a DataFrame, as your model expects
    df_new = pd.DataFrame([applicant_data])
    
    # Make the prediction
    prediction = loaded_model.predict(df_new)
    
    # The prediction is 0 for 'No Default' and 1 for 'Default'
    if prediction[0] == 0:
        return "Low Risk: Approve Loan"
    else:
        return "High Risk: Reject Loan"

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint to receive new applicant data and return a prediction.
    """
    # Get the JSON data sent from the frontend
    data = request.get_json(force=True)
    
    # Extract the applicant data from the JSON
    applicant_data = {
        'credit_score': data.get('credit_score'),
        'annual_income': data.get('annual_income'),
        'loan_amount': data.get('loan_amount'),
        'loan_term_years': data.get('loan_term_years')
    }
    
    # Assess the risk using your model logic
    prediction = assess_credit_risk_internal(applicant_data)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)