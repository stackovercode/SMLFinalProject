# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from time import time

# # Load dataset with dtype specification and error handling
# data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
# df = pd.read_csv(data_path, header=0)

# # Assign column names
# col_names = ['source', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10']
# df.columns = col_names

# # Drop the first row if it contains column headers or irrelevant data
# if df.iloc[0, 0] == 'source':
#     df = df.drop(0).reset_index(drop=True)

# # Convert numerical columns to float and handle missing data
# for col in col_names[1:]:  # Exclude 'source' which is categorical
#     df[col] = pd.to_numeric(df[col], errors='coerce')

# # Convert 'source' to categorical integer labels
# df['source'] = pd.Categorical(df['source']).codes

# # Handle missing data if any
# df.dropna(inplace=True)

# # Verify the cleaned data
# print(df.head())
# print(df.info())

# # Stratified sampling to maintain class distribution
# df_sample = df.groupby('source', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=42)).reset_index(drop=True)

# # Verify the stratified sampling
# print(df_sample['source'].value_counts(normalize=True))
# print(df['source'].value_counts(normalize=True))

# # Function to train and evaluate SVM for a specific robot
# def train_evaluate_robot(df, robot_id):
#     # Filter the dataset for the specific robot
#     df_robot = df[df['source'] == robot_id]
    
#     # Split dataset into features and target variable
#     X = df_robot.drop('source', axis=1)
#     y = df_robot['source']
    
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    
#     # Standardize the data for SVM
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Train SVM
#     print(f"Training the model for robot {robot_id}...")
#     start_time = time()
#     svm = SVC(kernel='rbf', C=100, gamma='auto', probability=True)
#     svm.fit(X_train_scaled, y_train)
#     training_time = time() - start_time
#     print(f"Training completed in {training_time:.2f} seconds for robot {robot_id}")
    
#     # Evaluate SVM
#     y_pred_svm = svm.predict(X_test_scaled)
#     print(f'SVM Accuracy for robot {robot_id}:', accuracy_score(y_test, y_pred_svm))
#     print(confusion_matrix(y_test, y_pred_svm))
#     print(classification_report(y_test, y_pred_svm))
    
#     # Hyperparameter tuning (optional)
#     # param_grid = {
#     #     'C': [0.1, 1, 10, 100],
#     #     'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
#     #     'kernel': ['rbf', 'poly']
#     # }
#     # grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
#     # grid_search.fit(X_train_scaled, y_train)
    
#     # print(f"Best parameters for robot {robot_id}: {grid_search.best_params_}")
#     # print(f"Best cross-validation score for robot {robot_id}: {grid_search.best_score_:.4f}")
    
#     # Optionally evaluate the best model
#     # best_svm = grid_search.best_estimator_
#     # y_pred_best = best_svm.predict(X_test_scaled)
#     # print(f'Best SVM Accuracy for robot {robot_id}:', accuracy_score(y_test, y_pred_best))
#     # print(confusion_matrix(y_test, y_pred_best))
#     # print(classification_report(y_test, y_pred_best))

# # Train and evaluate for one specific robot
# robot_id = 0  # Change this value to evaluate a different robot
# train_evaluate_robot(df_sample, robot_id)

# # Commented out the loop to evaluate all robots
# # for robot_id in df['source'].unique():
# #     train_evaluate_robot(df_sample, robot_id)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from time import time

# Load dataset with dtype specification and error handling
data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data_path, header=0)

# Assign column names
col_names = ['source', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10']
df.columns = col_names

# Drop the first row if it contains column headers or irrelevant data
if df.iloc[0, 0] == 'source':
    df = df.drop(0).reset_index(drop=True)

# Convert numerical columns to float and handle missing data
for col in col_names[1:]:  # Exclude 'source' which is categorical
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert 'source' to categorical integer labels
df['source'] = pd.Categorical(df['source']).codes

# Handle missing data if any
df.dropna(inplace=True)

# Verify the cleaned data
print(df.head())
print(df.info())

# Stratified sampling to maintain class distribution
df_sample = df.groupby('source', group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=42)).reset_index(drop=True)

# Verify the stratified sampling
print(df_sample['source'].value_counts(normalize=True))
print(df['source'].value_counts(normalize=True))

# Split dataset into features and target variable
X = df_sample.drop('source', axis=1)
y = df_sample['source']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Standardize the data for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
print("Training the model...")
start_time = time()
svm = SVC(kernel='rbf', C=100, gamma='auto', probability=True)
svm.fit(X_train_scaled, y_train)
training_time = time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# Evaluate SVM
y_pred_svm = svm.predict(X_test_scaled)
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Evaluate the model for each robot individually
def evaluate_robot_performance(robot_id):
    mask = (y_test == robot_id)
    print(f"Performance for robot {robot_id}:")
    print(confusion_matrix(y_test[mask], y_pred_svm[mask]))
    print(classification_report(y_test[mask], y_pred_svm[mask]))

# Evaluate performance for each robot
for robot_id in df['source'].unique():
    evaluate_robot_performance(robot_id)
