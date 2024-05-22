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

    @staticmethod
    def plot_data_distribution(df, title):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='source', data=df)
        plt.title(title)
        plt.savefig(f'./plot/preprocessing/{title.replace(" ", "_")}.png')
        plt.close()

    @staticmethod
    def plot_missing_values(df, title):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title(title)
        plt.savefig(f'./plot/preprocessing/{title.replace(" ", "_")}.png')
        plt.close()

    @staticmethod
    def plot_feature_distribution(X, title):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=X)
        plt.title(title)
        plt.savefig(f'./plot/preprocessing/{title.replace(" ", "_")}.png')
        plt.close()

    @staticmethod
    def plot_pca_variance(pca, title):
        plt.figure(figsize=(10, 6))
        plt.bar(range(pca.n_components_), pca.explained_variance_ratio_, alpha=0.5, align='center')
        plt.title(title)
        plt.savefig(f'./plot/preprocessing/{title.replace(" ", "_")}.png')
        plt.close()

    @staticmethod
    def plot_correlation_heatmap(df):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.savefig('./plot/preprocessing/correlation_heatmap.png')
        plt.close()

    def load_data(self, sample_fracs, random_state=42):
        df = pd.read_csv(self.data_path, header=0)
        print("Data loaded successfully")
        self.plot_data_distribution(df, "Initial Data Distribution")

        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        print("Numerical columns converted to float")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("Infinity values replaced with NaN")
        self.plot_missing_values(df, "Data with NaNs Replaced")

        df['source'] = pd.Categorical(df['source']).codes
        print("Target class encoded")

        df = df.sample(frac=1, random_state=random_state)
        df_sample = pd.concat([
            df[df['source'] == robot_id].sample(frac=sample_fracs[robot_id], random_state=random_state)
            for robot_id in sample_fracs
        ])
        print("Data shuffled and stratified sampled")
        self.plot_data_distribution(df_sample, "Data after Stratified Sampling")

        df_sample.dropna(inplace=True)
        print("Missing values handled")
        self.plot_missing_values(df_sample, "Data after Handling Missing Values")

        df_sample.drop_duplicates(inplace=True)
        print("Duplicate rows removed")
        self.plot_data_distribution(df_sample, "Data after Removing Duplicates")

        self.df_sample = df_sample

    def preprocess_data(self, apply_pca=False, n_components=10):
        df = self.df_sample
        X = df.drop('source', axis=1).values
        y = df['source'].values
        print("Data split into features and target variable")
        self.plot_feature_distribution(X, "Feature Distribution Before Split")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
        print("Data split into training and testing sets")
        self.plot_feature_distribution(X_train, "Training Set Feature Distribution")
        self.plot_feature_distribution(X_test, "Testing Set Feature Distribution")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        print("Data standardized")
        self.plot_feature_distribution(X_train_scaled, "Standardized Training Set Feature Distribution")
        self.plot_feature_distribution(X_test_scaled, "Standardized Testing Set Feature Distribution")

        if apply_pca:
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            print("PCA applied")
            self.plot_pca_variance(pca, "PCA Explained Variance")
            self.plot_feature_distribution(X_train_pca, "PCA Transformed Training Set Feature Distribution")
            self.plot_feature_distribution(X_test_pca, "PCA Transformed Testing Set Feature Distribution")
            return X_train_pca, X_test_pca, y_train, y_test, pca.explained_variance_ratio_

        return X_train_scaled, X_test_scaled, y_train, y_test

    def examine_robot_data(self, robot_id, progress, total):
        plot_dir = './plot/preprocessing'
        os.makedirs(plot_dir, exist_ok=True)

        Features = os.path.join(plot_dir, 'Feature' + str(robot_id) + '.png')

        robot_data = self.df_sample[self.df_sample['source'] == robot_id]

        plt.figure(figsize=(12, 8))
        sns.pairplot(robot_data.drop('source', axis=1))
        plt.savefig(Features)
        plt.close()

        missing_values = robot_data.isnull().sum()

        prog = int((progress / total) * 30)
        self.show_progress("Processing", prog, int((progress / total) * 100))

    def save_cleaned_data(self):
        self.df_sample.to_csv(self.processed_data_path, index=False)
