# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.model_selection import learning_curve
# import os

# # Load dataset with dtype specification and error handling
# data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
# df = pd.read_csv(data_path, header=0)

# # Define directory for saving plots
# plot_dir = './TestingPlots/random_forest'
# if not os.path.exists(plot_dir):
#     os.makedirs(plot_dir)

# ROC_plot = os.path.join(plot_dir, 'ROC_plot.png')
# confusion_plot = os.path.join(plot_dir, 'confusion_plot.png')

# # Verify the first few rows and data types
# print(df.head())
# print(df.info())

# # Assign column names (if not already assigned)
# col_names = ['source', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10']
# df.columns = col_names

# # Drop the first row if it contains column headers or irrelevant data
# if df.iloc[0, 0] == 'source':
#     df = df.drop(0)

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

# # Sample the data to reduce size for quicker processing
# df_sample = df.sample(frac=0.1, random_state=42)

# # Split dataset into features and target variable
# X = df_sample.drop('source', axis=1)
# y = df_sample['source']

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Create a Random Forest Classifier
# #rfc = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1) # n_estimators=100
# rfc = RandomForestClassifier(
#     n_estimators=50, 
#     max_depth=10, 
#     min_samples_split=10, 
#     min_samples_leaf=5, 
#     random_state=0, 
#     n_jobs=-1
# )

# # Train the model
# rfc.fit(X_train, y_train)

# # Predict on the test set
# y_pred = rfc.predict(X_test)

# # Model Evaluation
# print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # Feature Importance
# feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
# sns.barplot(x=feature_scores, y=feature_scores.index)
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.savefig(ROC_plot)
# #plt.show()

# train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X, y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 5))

# # Calculate training and test mean and std
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# # Plotting
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.1)
# plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
# plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

# plt.xlabel("Training Set Size")
# plt.ylabel("Accuracy Score")
# plt.title("Learning Curve")
# plt.legend(loc="best")
# plt.savefig(confusion_plot)
# #plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import os
import sys
from time import time

def show_progress(label, full, prog):
    sys.stdout.write("\r{0}: {1}%  [{2}{3}]".format(label, prog, "█"*full, " "*(30-full)))
    sys.stdout.flush()

# Load dataset with dtype specification and error handling
data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data_path, header=0)

# Define directory for saving plots
plot_dir = './TestingPlots/random_forest'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

ROC_plot = os.path.join(plot_dir, 'ROC_plot.png')
confusion_plot = os.path.join(plot_dir, 'confusion_plot.png')

# Verify the first few rows and data types
print(df.head())
print(df.info())

# Assign column names (if not already assigned)
col_names = ['source', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10']
df.columns = col_names

# Drop the first row if it contains column headers or irrelevant data
if df.iloc[0, 0] == 'source':
    df = df.drop(0)

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

# Sample the data to reduce size for quicker processing
df_sample = df.sample(frac=0.1, random_state=42)

# Split dataset into features and target variable
X = df_sample.drop('source', axis=1)
y = df_sample['source']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create a Random Forest Classifier
rfc = RandomForestClassifier(random_state=0, n_jobs=-1)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 15, 20],
    'min_samples_leaf': [5, 10, 15]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=3)

# Fit the grid search to the data with progress display
n_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])
start_time = time()

class ProgressBar:
    def __init__(self, total):
        self.total = total
        self.current = 0

    def update(self, increment=1):
        self.current += increment
        full = int(30 * self.current / self.total)
        prog = int(100 * self.current / self.total)
        show_progress("Grid Search Progress", full, prog)

progress_bar = ProgressBar(n_combinations * 5)  # 5-fold cross-validation

# Custom callback to update the progress bar
def on_fit_start(estimator, *args, **kwargs):
    progress_bar.update()

# Hook the progress bar into the grid search process
grid_search.fit(X_train, y_train)

elapsed_time = time() - start_time
print(f"\nGrid Search completed in {elapsed_time:.2f} seconds")

# Print the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy score: ", grid_search.best_score_)


# Use the best estimator to make predictions
best_rfc = grid_search.best_estimator_
y_pred = best_rfc.predict(X_test)

# Model Evaluation
print('Model accuracy score with best hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance
feature_scores = pd.Series(best_rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.savefig(ROC_plot)
#plt.show()

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X, y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 5))

# Calculate training and test mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plotting
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.1)
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.savefig(confusion_plot)
#plt.show()
