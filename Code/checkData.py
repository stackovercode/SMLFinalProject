import pandas as pd

# Load the dataset to inspect
data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data_path, header=None)
print(df.head(10))
print(df.info())