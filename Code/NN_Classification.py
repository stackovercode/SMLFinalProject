import pandas as pd
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import graphviz
from graphviz import Source
from IPython.display import SVG

### ML Models ###
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import export_text
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

##################################

### Metrics ###
from sklearn import metrics
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, classification_report

"""# Part 1: Load and clean the data"""


# Load the data.
data = '4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data, header=None)

# Display information about the DataFrame
df.info()

# Shape of the data set.
print("The data set has {} rows and {} columns.".format(df.shape[0], df.shape[1]))

# Check for missing values.
df.isna().any()

# Handle missing values
df.dropna(inplace=True)  # Remove rows with any NaN values

# Check for duplicate rows.
df.duplicated().any()

# Remove duplicates
df.drop_duplicates(inplace=True)

# Checking the values from each column.
for col in df.columns:
    print("Column:", col)
    print(df[col].value_counts(),'\n')

# Plotting the values of each column.
for i in df.columns:
    labels = df[i].unique()
    values = df[i].value_counts()
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_layout(title=go.layout.Title(text='Value distribution for column: "{}"'.format(i), x=0.5))
    fig.show()

# Define column names for the DataFrame
column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class_val']
df.columns = column_names

# Create category types
buying_type = CategoricalDtype(['low','med','high','vhigh'], ordered=True)
maint_type = CategoricalDtype(['low','med','high','vhigh'], ordered=True)
doors_type = CategoricalDtype(['2','3','4','5more'], ordered=True)
persons_type = CategoricalDtype(['2','4','more'], ordered=True)
lug_boot_type = CategoricalDtype(['small','med','big'], ordered=True)
safety_type = CategoricalDtype(['low','med','high'], ordered=True)
class_type = CategoricalDtype(['unacc','acc','good','vgood'], ordered=True)

# Convert all categorical values to category type in the DataFrame
df['buying'] = df['buying'].astype(buying_type)
df['maint'] = df['maint'].astype(maint_type)
df['doors'] = df['doors'].astype(doors_type)
df['persons'] = df['persons'].astype(persons_type)
df['lug_boot'] = df['lug_boot'].astype(lug_boot_type)
df['safety'] = df['safety'].astype(safety_type)
df['class_val'] = df['class_val'].astype(class_type)

"""# Part 2: Preprocessing
In this part we prepare our data for our models. This means that we choose the columns that will be our independed variables and which column the class that we want to predict. Once we are done with that, we split our data into train and test sets and perfom a standardization upon them.
"""

# Convert categories into integers for each column.
df['buying'] = df['buying'].replace({'low':0, 'med':1, 'high':2, 'vhigh':3})
df['maint'] = df['maint'].replace({'low':0, 'med':1, 'high':2, 'vhigh':3})
df['doors'] = df['doors'].replace({'2':0, '3':1, '4':2, '5more':3})
df['persons'] = df['persons'].replace({'2':0, '4':1, 'more':2})
df['lug_boot'] = df['lug_boot'].replace({'small':0, 'med':1, 'big':2})
df['safety'] = df['safety'].replace({'low':0, 'med':1, 'high':2})
df['class_val'] = df['class_val'].replace({'unacc':0, 'acc':1, 'good':2, 'vgood':3})

# The data set after the conversion.
df.head()

plt.figure(figsize=(10,6))
sns.set(font_scale=1.2)
sns.heatmap(df.corr(),annot=True, cmap='rainbow',linewidth=0.5)
plt.title('Correlation matrix');

# # Choose attribute columns and class column.
# X = df[df.columns[:-1]]  # All columns except the last one
# y = df['class_val']      # The last column as the target
# # Assuming 'data' is loaded as a DataFrame similar to the car dataset example
X = data.iloc[:, 1:]  # all rows, all columns except the first column (labels)
y = data.iloc[:, 0]   # all rows, first column only (labels)


# Drop rows with any NaN values
df = df.dropna()

# Choose attribute columns and class column
# X = df[df.columns[:-1]]  # Feature columns
# y = df['class_val']      # Target column
# # Assuming 'data' is loaded as a DataFrame similar to the car dataset example
X = data.iloc[:, 1:]  # all rows, all columns except the first column (labels)
y = data.iloc[:, 0]   # all rows, first column only (labels)


# # Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Verify shapes of the datasets
print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

"""# Part 3: Modeling

"""

# # Initialize a Multi-layer Perceptron classifier.
# mlp = MLPClassifier(hidden_layer_sizes=(5),max_iter=1000, random_state=42, shuffle=True, verbose=False)

# # Train the classifier.
# mlp.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', random_state=42)
mlp.fit(X_train, y_train)

# Make predictions.
mlp_pred = mlp.predict(X_test)

# CV score
mlp_cv = cross_val_score(mlp,X_train,y_train,cv=10)

"""## Metrics for Neural Network (MLP)"""

# # The mean squared error (relative error).
# print("Mean squared error (MSE): %.3f" % mean_squared_error(y_test, mlp_pred))

# # Explained average absolute error (average error).
# print("Mean absolute error (MAE): %.3f" % mean_absolute_error(y_test, mlp_pred))

# # Explained variance score: 1 is perfect prediction.
# print('Accuracy: %.3f' % mlp.score(X_test, y_test))

# Evaluate the model
print("Mean squared error (MSE):", mean_squared_error(y_test, mlp_pred))
print("Mean absolute error (MAE):", mean_absolute_error(y_test, mlp_pred))
print('Accuracy:', mlp.score(X_test, y_test))

# CV Accuracy
print('CV Accuracy: %.3f' % mlp_cv.mean())

# # Plot confusion matrix for MLP.
# mlp_matrix = confusion_matrix(y_test,mlp_pred)
# plt.figure(figsize=(8,8))
# sns.set(font_scale=1.4)
# sns.heatmap(mlp_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")
# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.title('Confusion Matrix for MLP');

# Confusion matrix
mlp_matrix = confusion_matrix(y_test, mlp_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(mlp_matrix, annot=True, cmap='Blues', fmt="d")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for MLP')
plt.show()


"""## **Task: Analyze the importance of different features in predicting car evaluations using the trained neural network model.**

## Grid search for Neural Network
"""

# Hyperparameters to be checked (simplified for quicker execution)
# parameters = {
#     'activation': ['logistic', 'relu'],  # Consider fewer options
#     'solver': ['adam'],  # Focus on the best solver
#     'alpha': 10.0 ** -np.arange(1, 2),  # Reduce the range of 'alpha'
#     'hidden_layer_sizes': [(5), (100), (3, 1)]  # Fewer and simpler configurations
# }

# Setting up a grid search for hyperparameter tuning
parameters = {
    'activation': ['logistic', 'relu'],
    'solver': ['adam'],
    'alpha': 10.0 ** -np.arange(1, 7),
    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 100)]
}


# MLP estimator.
default_mlp = MLPClassifier(random_state=42)

# # GridSearchCV estimator.
# gs_mlp = GridSearchCV(default_mlp, parameters, cv=5, n_jobs=-1, verbose=1)  # Reduced number of folds

# # Train the GridSearchCV estimator and search for the best parameters.
# gs_mlp.fit(X_train, y_train)

gs_mlp = GridSearchCV(MLPClassifier(random_state=42), parameters, cv=5)
gs_mlp.fit(X_train, y_train)

"""## **Task: Train the network for hidden_layer_sizes':[(5),(100),(3),(4),(3,1),(5,3)] and see the impact on the outcome.**"""

# Make predictions with the best parameters.
gs_mlp_pred=gs_mlp.predict(X_test)

"""## Metrics for GridSearchCV MLP"""

# Best parameters.
# print("Best MLP Parameters: {}".format(gs_mlp.best_params_))
# Best parameters and model performance
print("Best MLP Parameters:", gs_mlp.best_params_)


# The mean squared error (relative error).
print("Mean squared error (MSE): %.3f" % mean_squared_error(y_test, gs_mlp_pred))

# Explained average absolute error (average error).
print("Average absolute error (MAE): %.3f" % mean_absolute_error(y_test, gs_mlp_pred))

# Cross validation accuracy for the best parameters.
# print('CV accuracy: %0.3f' % gs_mlp.best_score_)
print('Grid Search CV Accuracy:', gs_mlp.score(X_test, y_test))

# Accuracy: 1 is perfect prediction.
print('Accuracy: %0.3f' % (gs_mlp.score(X_test,y_test)))

# Print confusion matrix for GridSearchCV MLP.
gs_mlp_matrix = confusion_matrix(y_test,gs_mlp_pred)
plt.figure(figsize=(8,8))
sns.set(font_scale=1.4)
sns.heatmap(gs_mlp_matrix,annot=True, cbar=False, cmap='twilight',linewidth=0.5,fmt="d")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for GridSearchCV MLP');

# # Ploting metrics
# errors=['Accuracy','CV-accuracy','MSE', 'MAE']

# fig = go.Figure(data=[
#     go.Bar(name='MLP', x=errors, y=[mlp.score(X_test, y_test),mlp_cv.mean(),mean_squared_error(y_test, mlp_pred), mean_absolute_error(y_test, mlp_pred)]),
#     go.Bar(name='GridSearchCV+MLP', x=errors, y=[gs_mlp.score(X_test, y_test),gs_mlp.best_score_,mean_squared_error(y_test, gs_mlp_pred), mean_absolute_error(y_test, gs_mlp_pred)])
# ])

# fig.update_layout(
#     title='Metrics for each model',
#     xaxis_tickfont_size=14,
#     barmode='group',
#     bargap=0.15, # gap between bars of adjacent location coordinates.
#     bargroupgap=0.1 # gap between bars of the same location coordinate.
# )
# fig.show()

"""## **Task: Compare the outcomes of the Neural Network training with previously trained models in previous lectures.**"""