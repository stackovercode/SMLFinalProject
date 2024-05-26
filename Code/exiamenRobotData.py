import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_and_preprocess_data(data_path):
    """Loads and cleans the dataset."""
    df = pd.read_csv(data_path, header=0, dtype=str)
    col_names = ['source'] + [f'num{i}' for i in range(1, 11)]
    df.columns = col_names
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Infinity values replaced with NaN")
    df = df.iloc[1:].copy()
    df[df.columns[1:]] = df[df.columns[1:]].astype(float)
    df['source'] = df['source'].astype('category')
    return df

def examine_robot_data(df, robot_id):
    """Examine the data for a specific robot."""
    robot_data = df[df['source'] == robot_id]
    print(f"Data for Robot {robot_id}:")
    print(robot_data.describe())
    
    # Visualize data distribution
    plt.figure(figsize=(12, 8))
    sns.pairplot(robot_data.drop('source', axis=1))
    plt.suptitle(f"Feature Distribution for Robot {robot_id}", y=1.02)
    plt.show()

    # Check for missing values
    missing_values = robot_data.isnull().sum()
    print(f"Missing values for Robot {robot_id}:\n{missing_values}")

    # Check class distribution in features
    for col in robot_data.columns[1:]:
        print(f"\nDistribution of feature {col} for Robot {robot_id}:")
        print(robot_data[col].value_counts().head())

if __name__ == '__main__':
    data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
    
    # Load and preprocess data
    df = load_and_preprocess_data(data_path)

    # Examine data for Robot 3
    examine_robot_data(df, '3')
