import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the default sample size
default_sample_size = 0.2

# Define sample fractions for each class
sample_fracs = {
    0: default_sample_size,
    1: default_sample_size,
    2: default_sample_size,
    3: default_sample_size,
    4: default_sample_size
}

# Function to plot data distribution
def plot_data_distribution(df, title):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='source', data=df)
    plt.title(title)
    plt.savefig(f'./plot/preprocessing/{title.replace(" ", "_")}.png')
    plt.close()

# Load the data
data_path = './data/4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data_path, header=0)
print("Data loaded successfully")

plot_data_distribution(df, "Initial Data Distribution")
print(df['source'].value_counts())  # Debug: Print initial class distribution

# Convert columns to numeric and handle infinite values
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')
print("Numerical columns converted to float")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Infinity values replaced with NaN")

# Encode target class
df['source'] = pd.Categorical(df['source']).codes
print("Target class encoded")

# Handle missing values
df.dropna(inplace=True)
print("Missing values handled")

# Calculate weights based on duplication
df['weight'] = df.groupby(df.columns[1:].tolist())['source'].transform('count')
df['weight'] = 1 / df['weight']
print("Weights assigned based on duplication")

# Remove duplicates across all classes
df_deduped = df.drop_duplicates(subset=df.columns[1:], keep='first')
print("Duplicates across all classes removed")

# Merge weights back to deduped dataframe
df_deduped = df_deduped.merge(df[['weight']], left_index=True, right_index=True, how='left')
print("Weights assigned to deduped dataframe")

# Remove the incorrect weight_x and rename weight_y to weight
df_deduped = df_deduped.drop(columns=['weight_x']).rename(columns={'weight_y': 'weight'})

# Debug: Print columns to verify the presence of the weight column
print("Columns in df_deduped:", df_deduped.columns)

plot_data_distribution(df_deduped, "Data after Removing Inter-class Duplicates")
print(df_deduped['source'].value_counts())  # Debug: Print class distribution after removing inter-class duplicates

# Perform stratified sampling with weights
default_sample_size = 0.2
sample_fracs = {i: default_sample_size for i in range(5)}

def weighted_sample(df, sample_frac, weights_column, random_state):
    # Debug: Print columns to verify the presence of the weight column before sampling
    print("Columns in weighted_sample df:", df.columns)
    return df.sample(frac=sample_frac, weights=weights_column, random_state=random_state)

df_sample = pd.concat([
    weighted_sample(df_deduped[df_deduped['source'] == robot_id], sample_fracs[robot_id], 'weight', random_state=42)
    for robot_id in sample_fracs
])
print("Data shuffled and stratified sampled using weighted sampling")
plot_data_distribution(df_sample, "Data after Weighted Stratified Sampling")
print(df_sample['source'].value_counts())  # Debug: Print class distribution after weighted stratified sampling

# Balance the classes to the size of the smallest class
min_class_size = df_sample['source'].value_counts().min()
df_sample_balanced = pd.concat([
    df_sample[df_sample['source'] == robot_id].sample(n=min_class_size, random_state=42)
    for robot_id in df_sample['source'].unique()
])
print("Classes balanced after weighted stratified sampling and cleaning")
plot_data_distribution(df_sample_balanced, "Data after Balancing Classes")
print(df_sample_balanced['source'].value_counts())  # Debug: Print class distribution after balancing

# Save intermediary data for manual inspection if needed
# df.to_csv('./data/processed_data_after_cleanup.csv', index=False)
# df_sample.to_csv('./data/processed_data_after_sampling.csv', index=False)
# df_sample_balanced.to_csv('./data/processed_data_after_balancing.csv', index=False)

# Assign the balanced sample to df_sample for further use
df_sample = df_sample_balanced