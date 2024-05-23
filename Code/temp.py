import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, message=".*use_inf_as_na.*")

class Preprocessing:
    def __init__(self, data_path, processed_data_path, default_sample_size=0.33):
        self.data_path = data_path
        self.processed_data_path = processed_data_path
        self.default_sample_size = default_sample_size
        self.df_sample = None

    @staticmethod
    def show_progress(label, full, prog):
        sys.stdout.write("\r{0}: {1}%  [{2}{3}]".format(label, prog, "█" * full, " " * (30 - full)))
        sys.stdout.flush()

    def load_data(self, sample_fracs, random_state=42):
        df = pd.read_csv(self.data_path, header=0)
        print("Data loaded successfully")
        
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print("Numerical columns converted to float")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("Infinity values replaced with NaN")

        df['source'] = pd.Categorical(df['source']).codes
        print("Target class encoded")

        df = df.sample(frac=1, random_state=random_state)
        df_sample = pd.concat([
            df[df['source'] == robot_id].sample(frac=sample_fracs[robot_id], random_state=random_state)
            for robot_id in sample_fracs
        ])
        print("Data shuffled and stratified sampled")

        df_sample.dropna(inplace=True)
        print("Missing values handled")

        df_sample.drop_duplicates(inplace=True)
        print("Duplicate rows removed")

        self.df_sample = df_sample

    def preprocess_data(self, apply_pca=False, n_components=10):
        robot_data_dict = {}
        for robot_id in range(5):
            robot_data = self.df_sample[self.df_sample['source'] == robot_id]
            
            # Compute statistical features
            robot_data['mean'] = robot_data.mean(axis=1)
            robot_data['std'] = robot_data.std(axis=1)
            
            X = robot_data.drop('source', axis=1).values
            y = robot_data['source'].values
            
            print(f"Data split into features and target variable for robot {robot_id}")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
            print(f"Data split into training and testing sets for robot {robot_id}")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            print(f"Data standardized for robot {robot_id}")

            if apply_pca:
                pca = PCA(n_components=n_components)
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                print(f"PCA applied for robot {robot_id}")
                robot_data_dict[robot_id] = (X_train_pca, X_test_pca, y_train, y_test)
            else:
                robot_data_dict[robot_id] = (X_train_scaled, X_test_scaled, y_train, y_test)
        
        return robot_data_dict

    def save_cleaned_data(self):
        self.df_sample.to_csv(self.processed_data_path, index=False)
