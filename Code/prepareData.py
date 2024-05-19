import pandas as pd

# Load the dataset, skipping the initial header row
data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data_path, header=0)

# Define column names
col_names = ['source', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10']
df.columns = col_names

# Check for the correct column headers and data types
print(df.head())
print(df.info())

# Drop the first row if it contains headers or irrelevant data
if df.iloc[0, 0] == 'source':
    df = df.drop(0)

# Convert numerical columns to float
for col in col_names[1:]:  # Exclude 'source' which is categorical
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check for missing data
print(df.isnull().sum())

# Handle missing data if any
df.dropna(inplace=True)

# Convert 'source' to categorical integer labels if needed
df['source'] = pd.Categorical(df['source']).codes

# Inspect cleaned data
print(df.head())
print(df.info())

# Now save the cleaned DataFrame for future use
df.to_csv('./cleaned_robot_data.csv', index=False)
