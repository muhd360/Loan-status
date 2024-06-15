from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            
        
            # Retrieve form data
            data = CustomData(
                id=request.form.get('id'),
                gender=request.form.get('Gender'),
                married=request.form.get('Married'),
                dependents=request.form.get('Dependents'),
                education=request.form.get('Education'),
                self_employed=request.form.get('Self_Employed'),
                applicant_income=float(request.form.get('ApplicantIncome')),  # Assuming income is numerical
                coapplicant_income=float(request.form.get('CoapplicantIncome')),  # Assuming income is numerical
                loan_amount=float(request.form.get('LoanAmount')),  # Assuming loan amount is numerical
                loan_amount_term=float(request.form.get('Loan_Amount_Term')),
                credit_history=float(request.form.get('Credit_History')),
                property_area=request.form.get('Property_Area')
            )

            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            

            print("before")

            predict_pipeline=PredictPipeline()
            print("mid")
            results=predict_pipeline.predict(pred_df)
            print("after")

            # Debug information
            return render_template('home.html',renuka=results[0])

        except Exception as e:
            print(f"Error occurred: {e}")


    






if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
