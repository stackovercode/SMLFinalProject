import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
data = '../4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data, header=None)

# Assign column names
col_names = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10', 'class']
df.columns = col_names

# Convert numerical columns to float and handle errors
for col in col_names[:-1]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode target class as categorical integer labels
df['class'] = pd.Categorical(df['class']).codes


import matplotlib.pyplot as plt
import seaborn as sns

# Plot original data (before handling missing values)
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Dataset')
plt.show()

# Handle missing data
df.dropna(inplace=True)

# Assuming df is the cleaned DataFrame
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('After Handling Missing Values')
plt.show()

# Sample the data to reduce size for quicker processing
df_sample = df.sample(frac=0.1, random_state=42)

# Split dataset into features and target variable
X = df_sample.drop('class', axis=1).values
y = df_sample['class'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Plot original data before normalization
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_sample.iloc[:, :-1])
plt.title('Original Data Before Normalization')
plt.xlabel('Features')
plt.ylabel('Value')
plt.show()

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled data back to DataFrame for visualization
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=col_names[:-1])

# Plot standardized data
plt.figure(figsize=(12, 6))
sns.boxplot(data=X_train_scaled_df)
plt.title('Standardized Data After Normalization')
plt.xlabel('Features')
plt.ylabel('Standardized Value')
plt.show()

from sklearn.decomposition import PCA
import numpy as np

# Apply PCA
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Plot explained variance ratio
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.show()

# Plot first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', edgecolor='k', s=40)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA - First Two Principal Components')
plt.colorbar()
plt.show()