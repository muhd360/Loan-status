import pandas as pd
from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging
data = ["1111","Male", "Yes", "2", "Graduate", "Yes", 12173, 0, 166.0, 360.0, 0.0, "Semiurban","Y"]

# Column names (assuming you have these)
columns = ["ID","Gender", "Married", "Dependents", "Education", "Self_Employed", "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area","Loan_Status"]

# Create DataFrame
df = pd.DataFrame([data], columns=columns)

def func():
    try:
        # Initialize prediction pipeline
        predict_pipeline = PredictPipeline()

        # Get prediction
        result = predict_pipeline.predict(df)

        # Print result
        print(result)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    func()