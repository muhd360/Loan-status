import joblib
import numpy as np
import pandas as pd
import os

def load_object(file_path):
    return joblib.load(file_path)

def predict(features):
    try:
        model_path = os.path.join(r"model.pkl")
        preprocessor_path = os.path.join("proprocessor.pkl")

        print("Before Loading")
        model = load_object(file_path=model_path)
        preprocessor = load_object(file_path=preprocessor_path)
        print("After Loading")

        # Convert features to DataFrame
        data = [features]
        columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                   'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
                   'Credit_History', 'Property_Area']
        df = pd.DataFrame(data, columns=columns)

        # Process data
        data_scaled = preprocessor.transform(df)
        
        # Make predictions
        preds = model.predict(data_scaled)
        return preds
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    # Example feature set
    features = ['LP001059', 'Male', 'Yes', 2, 'Graduate', '', 13633, 0, 280, 240, 1, 'Urban']
    predictions = predict(features)
    print(predictions)
