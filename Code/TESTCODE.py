from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(data_path, sample_fracs, random_state=42):
    """Loads, cleans, and samples the dataset."""
    df = pd.read_csv(data_path, header=0, dtype=str)
    col_names = ['source'] + [f'num{i}' for i in range(1, 11)]
    df.columns = col_names
    df = df.iloc[1:].copy()  # Skip the first row if it's not needed
    df[df.columns[1:]] = df[df.columns[1:]].astype(float)
    df['source'] = df['source'].astype('category')

    # Shuffle and sample
    df = df.sample(frac=1, random_state=random_state)
    df_sample = pd.concat([
        df[df['source'] == robot_id].sample(frac=sample_fracs[robot_id], random_state=random_state)
        for robot_id in sample_fracs
    ])

    return df_sample.drop('source', axis=1), df_sample['source']

if __name__ == '__main__':
    data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'

    # Define sample fractions for each robot ID
    sample_fracs = {
        '0': 0.1,
        '1': 0.1,
        '2': 0.1,
        '3': 0.1,
        '4': 0.1
    }

    X, y = load_and_preprocess_data(data_path, sample_fracs)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning for RandomForestClassifier
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }

    rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1, verbose=2)
    rf_grid_search.fit(X_train_scaled, y_train)
    best_rf = rf_grid_search.best_estimator_

    # Train and evaluate the RandomForestClassifier
    best_rf.fit(X_train_scaled, y_train)
    y_pred_rf = best_rf.predict(X_test_scaled)

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_report = classification_report(y_test, y_pred_rf)

    print(f'Random Forest Accuracy: {rf_accuracy}')
    print('Random Forest Classification Report:')
    print(rf_report)
