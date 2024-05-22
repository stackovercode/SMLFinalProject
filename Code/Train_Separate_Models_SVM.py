
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from time import time
from collections import defaultdict
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import PCA

# Load and preprocess data
def load_and_preprocess_data(data_path, sample_fracs, random_state=42):
    df = pd.read_csv(data_path, header=None, dtype=str)
    col_names = ['source'] + [f'num{i}' for i in range(1, 11)]
    df.columns = col_names
    df = df.iloc[1:].copy()
    df[df.columns[1:]] = df[df.columns[1:]].astype(float)
    df['source'] = df['source'].astype('category')

    df = df.sample(frac=1, random_state=random_state)
    df_sample = pd.concat([
        df[df['source'] == robot_id].sample(frac=sample_fracs[robot_id], random_state=random_state)
        for robot_id in sample_fracs
    ])

    for robot_id in sample_fracs:
        if len(df_sample[df_sample['source'] == robot_id]) < 10:
            raise ValueError(f"Insufficient data for robot {robot_id} after sampling")

    return df_sample.drop('source', axis=1), df_sample['source']

# Train SVM model for each robot
def train_robot_model(X, y, robot_id):
    mask = (y == robot_id)
    X_robot = X[mask]
    y_robot = y[mask]

    scaler = StandardScaler()
    X_robot_scaled = scaler.fit_transform(X_robot)

    svm = SVC(kernel='rbf', C=100, gamma='auto', probability=True)
    svm.fit(X_robot_scaled, y_robot)

    return svm, scaler

if __name__ == '__main__':
    data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
    sample_fracs = {
        '0': 0.10,
        '1': 0.30,
        '2': 0.10,
        '3': 0.30,
        '4': 0.10
    }

    X, y = load_and_preprocess_data(data_path, sample_fracs)

    # Train separate models for each robot
    models = {}
    scalers = {}
    for robot_id in ['0', '1', '2', '3', '4']:
        model, scaler = train_robot_model(X, y, robot_id)
        models[robot_id] = model
        scalers[robot_id] = scaler

    print("Models trained successfully for each robot.")

# Create a meta-classifier combining individual robot models
def create_meta_classifier(models):
    estimators = [(f'robot_{robot_id}', models[robot_id]) for robot_id in models]
    meta_classifier = VotingClassifier(estimators=estimators, voting='soft')
    return meta_classifier

if __name__ == '__main__':
    meta_classifier = create_meta_classifier(models)

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the entire training and validation set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train meta-classifier
    meta_classifier.fit(X_train_scaled, y_train)

    # Validate the model
    y_pred = meta_classifier.predict(X_val_scaled)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    print(classification_report(y_val, y_pred))
