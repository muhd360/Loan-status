import pandas as pd
import os
train_data_path ='tr.csv'
data=pd.read_csv('tr.csv')
for column in data.columns:
    print(column)

numerical_columns = data.select_dtypes(include=['int', 'float']).columns
print("Numerical Columns:")
print(numerical_columns)

# Print categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
print("\nCategorical Columns:")
# target=['Loan_Status']
# data_p=data.drop(columns=target,axis=1)
# print(data_p)