import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv('artifacts/train.csv')  # Replace 'data.csv' with your actual dataset

# Assuming 'Loan_Status' is the target variable
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])
y = df['Loan_Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing steps
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute missing values with a constant
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with the preprocessor and the model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the pipeline on the training data
model_pipeline.fit(X_train, y_train)

# Extract and save the preprocessor and model separately
preprocessor = model_pipeline.named_steps['preprocessor']
regressor = model_pipeline.named_steps['regressor']

with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(regressor, f)

# Apply preprocessing and evaluate the model
X_test_preprocessed = preprocessor.transform(X_test)
predictions = regressor.predict(X_test_preprocessed)

# Convert predictions to binary outcome if necessary (for classification tasks)
# For example, assuming a threshold of 0.5
threshold = 0.5
binary_predictions = (predictions > threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, binary_predictions)

print(f'Accuracy on test data: {accuracy}')
