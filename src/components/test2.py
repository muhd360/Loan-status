import pickle
import pandas as pd

# Load the preprocessor and the model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the test data
test_df = pd.read_csv('artifacts/test.csv')

# Assuming 'Loan_ID' is the first column and 'Loan_Status' is not included
X_test = test_df.drop(['Loan_ID'], axis=1)

# Apply preprocessing on the test data
X_test_preprocessed = preprocessor.transform(X_test)

# Make predictions using the model
predictions = model.predict(X_test_preprocessed)

# Assuming 'predictions' is for a classification task (1 or 0)
# You can convert it back to original labels if needed
# For binary classification, you might threshold probabilities or keep them as is

# Example of saving predictions back to a DataFrame
output_df = pd.DataFrame({
    'Loan_ID': test_df['Loan_ID'],  # Assuming 'Loan_ID' column is in 'test_df'
    'Loan_Status_Predicted': predictions  # Adjust column name as per your model output
})

# Save predictions to a CSV file
output_df.to_csv('predictions.csv', index=False)

print('Predictions saved to predictions.csv')
