import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

import os
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                gender: str,
                married: bool,
                dependents: str,
                education: str,
                self_employed: bool,
                applicant_income: float,
                coapplicant_income: float,
                loan_amount: float,
                loan_amount_term: str,
                credit_history: int,  # Assuming credit history is a numerical score
                property_area: str,
                loan_status: str):

        self.gender = gender
        self.married = married
        self.dependents = dependents
        self.education = education
        self.self_employed = self_employed
        self.applicant_income = applicant_income
        self.coapplicant_income = coapplicant_income
        self.loan_amount = loan_amount
        self.loan_amount_term = loan_amount_term
        self.credit_history = credit_history
        self.property_area = property_area


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
#"Loan_ID": [],  # Assuming Loan_ID is system generated, so an empty list
                "Gender": [self.gender],
                "Married": [self.married],
                "Dependents": [self.dependents],
                "Education": [self.education],
                "Self_Employed": [self.self_employed],
                "ApplicantIncome": [self.applicant_income],
                "CoapplicantIncome": [self.coapplicant_income],
                "LoanAmount": [self.loan_amount],
                "Loan_Amount_Term": [self.loan_amount_term],
                "Credit_History": [self.credit_history],
                "Property_Area": [self.property_area],
                
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

