import pandas as pd

# Read the CSV file into a DataFrame
data = pd.read_csv('test.csv')

# Get the first row of the DataFrame
first_row = data.iloc[0]

# Print the type of each element in the first row
for column in first_row.index:
    print(f"Column: {column}, Type: {type(first_row[column])}")
