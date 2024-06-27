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
            model_path=os.path.join("src/components/model.pkl")
            preprocessor_path=os.path.join('src/components/preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            ds=features.drop(columns="ID",axis=1)
            ds=preprocessor.transform(features)
            preds=model.predict(ds)
            #print(features)
            #data_scaled=preprocessor.transform(features)
            
            #preds=model.predict(data_scaled)
            # print(preds)
            # for column in preds.index:
            #     print(f"Column: {column}, Type: {type(preds[column])}")

            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                id:str,
                gender: str,
                married: str,
                dependents: str,
                education: str,
                self_employed: str,
                applicant_income: int,
                coapplicant_income: int,
                loan_amount: float,
                loan_amount_term: float,
                credit_history: float,  # Assuming credit history is a numerical score
                property_area: str,
                ):
        self.id=self.clean_input(id)
        self.gender = self.clean_input(gender)
        self.married = self.clean_input(married)
        self.dependents = self.clean_input(dependents)
        self.education = self.clean_input(education)
        self.self_employed = self.clean_input(self_employed)
        self.applicant_income = applicant_income
        self.coapplicant_income = coapplicant_income
        self.loan_amount = loan_amount
        self.loan_amount_term = loan_amount_term
        self.credit_history = credit_history
        self.property_area = self.clean_input(property_area)

    def clean_input(self, value: str) -> str:
        if isinstance(value, str):
            return value.strip("'\"")  # Remove both single and double quotes if present
        return value


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
#"Loan_ID": [],  # Assuming Loan_ID is system generated, so an empty list
                "ID":[self.id],
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

