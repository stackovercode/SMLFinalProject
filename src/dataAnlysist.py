
############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.neural_network import MLPClassifier

# # Load the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)

# # Separate features and target
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# # Normalize the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Define the parameter grid for Grid Search
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10]
# }

# # Create a Random Forest model
# rf = RandomForestClassifier(random_state=42)

# # Perform Grid Search with cross-validation
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# # Get the best parameters and best model
# best_params = grid_search.best_params_
# best_rf = grid_search.best_estimator_

# # Predict on the test set with the best model
# y_pred_rf = best_rf.predict(X_test)

# # Evaluate the best model
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# report_rf = classification_report(y_test, y_pred_rf)

# print("Best Parameters for Random Forest:", best_params)
# print("Random Forest Model Accuracy:", accuracy_rf)
# print("Random Forest Classification Report:\n", report_rf)

# # Optionally, train and evaluate a Neural Network
# mlp = MLPClassifier(random_state=42, max_iter=300)
# mlp.fit(X_train, y_train)

# # Predict on the test set with the neural network
# y_pred_mlp = mlp.predict(X_test)

# # Evaluate the neural network model
# accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
# report_mlp = classification_report(y_test, y_pred_mlp)

# print("Neural Network Model Accuracy:", accuracy_mlp)
# print("Neural Network Classification Report:\n", report_mlp)


############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################


# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Load the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)

# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Train the Random Forest model with class weights
# rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, class_weight='balanced', random_state=42)
# rf.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred = rf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print("Random Forest Model Accuracy after SMOTE:", accuracy)
# print("Random Forest Classification Report after SMOTE:\n", report)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# from imblearn.over_sampling import SMOTE

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Train the Gradient Boosting model
# gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
# gb.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred_gb = gb.predict(X_test)
# accuracy_gb = accuracy_score(y_test, y_pred_gb)
# report_gb = classification_report(y_test, y_pred_gb)

# print("Gradient Boosting Model Accuracy after SMOTE:", accuracy_gb)
# print("Gradient Boosting Classification Report after SMOTE:\n", report_gb)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pandas as pd

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Train the Random Forest model with class weights
# rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, class_weight='balanced', random_state=42)
# rf.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred = rf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)

# print("Random Forest Model Accuracy after SMOTE:", accuracy)
# print("Random Forest Classification Report after SMOTE:\n", report)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# import pandas as pd
# import os

# # Set the number of threads for sklearn to use
# os.environ["OMP_NUM_THREADS"] = "1"  # or any number of threads you want to use

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Train the Gradient Boosting model
# gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
# gb.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred_gb = gb.predict(X_test)
# accuracy_gb = accuracy_score(y_test, y_pred_gb)
# report_gb = classification_report(y_test, y_pred_gb)

# print("Gradient Boosting Model Accuracy with Feature Engineering after SMOTE:", accuracy_gb)
# print("Gradient Boosting Classification Report with Feature Engineering after SMOTE:\n", report_gb)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# import threadpoolctl
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # Set threading limits
# threadpoolctl.threadpool_limits(limits=1, user_api='blas')

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Train the Gradient Boosting model
# gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
# gb.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred_gb = gb.predict(X_test)
# accuracy_gb = accuracy_score(y_test, y_pred_gb)
# report_gb = classification_report(y_test, y_pred_gb)

# print("Gradient Boosting Model Accuracy with Feature Engineering after SMOTE:", accuracy_gb)
# print("Gradient Boosting Classification Report with Feature Engineering after SMOTE:\n", report_gb)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Define the parameter grid for RandomizedSearchCV
# param_dist = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'max_depth': [3, 4, 5, 6],
#     'subsample': [0.8, 0.9, 1.0],
#     'min_samples_split': [2, 5, 10]
# }

# # Create the Gradient Boosting model
# gb = GradientBoostingClassifier(random_state=42)

# # Perform RandomizedSearchCV
# random_search = RandomizedSearchCV(gb, param_distributions=param_dist, n_iter=50, cv=3, random_state=42, n_jobs=-1)
# random_search.fit(X_train_smote, y_train_smote)

# # Get the best model
# best_gb = random_search.best_estimator_

# # Predict and evaluate
# y_pred_best_gb = best_gb.predict(X_test)
# accuracy_best_gb = accuracy_score(y_test, y_pred_best_gb)
# report_best_gb = classification_report(y_test, y_pred_best_gb)

# print("Best Gradient Boosting Model Accuracy with Feature Engineering after SMOTE:", accuracy_best_gb)
# print("Best Gradient Boosting Classification Report with Feature Engineering after SMOTE:\n", report_best_gb)
# print("Best Parameters found by RandomizedSearchCV:", random_search.best_params_)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Define base models
# rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42)
# gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
# mlp = MLPClassifier(random_state=42, max_iter=300)

# # Define the stacking ensemble
# estimators = [
#     ('rf', rf),
#     ('gb', gb),
#     ('mlp', mlp)
# ]
# stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# # Train the stacking model
# stacking_clf.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred_stack = stacking_clf.predict(X_test)
# accuracy_stack = accuracy_score(y_test, y_pred_stack)
# report_stack = classification_report(y_test, y_pred_stack)

# print("Stacking Model Accuracy after SMOTE:", accuracy_stack)
# print("Stacking Classification Report after SMOTE:\n", report_stack)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Train the XGBoost model
# xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
# xgb.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred_xgb = xgb.predict(X_test)
# accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
# report_xgb = classification_report(y_test, y_pred_xgb)

# print("XGBoost Model Accuracy with Feature Engineering after SMOTE:", accuracy_xgb)
# print("XGBoost Classification Report with Feature Engineering after SMOTE:\n", report_xgb)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from sklearn.model_selection import RandomizedSearchCV
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Define the parameter grid for RandomizedSearchCV
# param_dist = {
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'max_depth': [3, 4, 5, 6],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
#     'gamma': [0, 0.1, 0.2, 0.3],
#     'min_child_weight': [1, 3, 5]
# }

# # Create the XGBoost model
# xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# # Perform RandomizedSearchCV
# random_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=50, cv=3, random_state=42, n_jobs=-1)
# random_search.fit(X_train_smote, y_train_smote)

# # Get the best model
# best_xgb = random_search.best_estimator_

# # Predict and evaluate
# y_pred_best_xgb = best_xgb.predict(X_test)
# accuracy_best_xgb = accuracy_score(y_test, y_pred_best_xgb)
# report_best_xgb = classification_report(y_test, y_pred_best_xgb)

# print("Best XGBoost Model Accuracy with Feature Engineering after SMOTE:", accuracy_best_xgb)
# print("Best XGBoost Classification Report with Feature Engineering after SMOTE:\n", report_best_xgb)
# print("Best Parameters found by RandomizedSearchCV:", random_search.best_params_)



############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# import matplotlib.pyplot as plt
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from xgboost import plot_importance, XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Train the best XGBoost model
# best_xgb = XGBClassifier(subsample=0.8, n_estimators=300, min_child_weight=3, max_depth=6, learning_rate=0.2, gamma=0.3, colsample_bytree=0.8, random_state=42)
# best_xgb.fit(X_train_smote, y_train_smote)

# # Plot feature importance
# plot_importance(best_xgb, max_num_features=10)
# plt.show()

# # Define base models for stacking
# rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42)
# mlp = MLPClassifier(random_state=42, max_iter=300)

# # Define the stacking ensemble
# estimators = [
#     ('rf', rf),
#     ('xgb', best_xgb),
#     ('mlp', mlp)
# ]
# stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# # Train the stacking model
# stacking_clf.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred_stack = stacking_clf.predict(X_test)
# accuracy_stack = accuracy_score(y_test, y_pred_stack)
# report_stack = classification_report(y_test, y_pred_stack)

# print("Stacking Model Accuracy after SMOTE:", accuracy_stack)
# print("Stacking Classification Report after SMOTE:\n", report_stack)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################


# from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
# from hyperopt.pyll.base import scope
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from imblearn.over_sampling import SMOTE
# import pandas as pd
# import numpy as np

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Define the objective function for Hyperopt
# def objective(params):
#     # Convert necessary parameters to int
#     params['n_estimators'] = int(params['n_estimators'])
#     params['max_depth'] = int(params['max_depth'])
#     params['min_child_weight'] = int(params['min_child_weight'])
    
#     model = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
#     model.fit(X_train_smote, y_train_smote)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     return {'loss': -accuracy, 'status': STATUS_OK}

# # Define the parameter space for Hyperopt
# param_space = {
#     'n_estimators': scope.int(hp.quniform('n_estimators', 100, 500, 50)),
#     'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
#     'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
#     'subsample': hp.uniform('subsample', 0.7, 1.0),
#     'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
#     'gamma': hp.uniform('gamma', 0, 0.5),
#     'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6, 1))
# }

# # Run Hyperopt for Bayesian Optimization
# trials = Trials()
# best_params = fmin(fn=objective, space=param_space, algo=tpe.suggest, max_evals=50, trials=trials, rstate=np.random.default_rng(42))

# # Convert the best parameters to the correct types
# best_params['n_estimators'] = int(best_params['n_estimators'])
# best_params['max_depth'] = int(best_params['max_depth'])
# best_params['min_child_weight'] = int(best_params['min_child_weight'])

# # Train the best model with the optimized parameters
# best_xgb = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# best_xgb.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred_best_xgb = best_xgb.predict(X_test)
# accuracy_best_xgb = accuracy_score(y_test, y_pred_best_xgb)
# report_best_xgb = classification_report(y_test, y_pred_best_xgb)

# print("Best XGBoost Model Accuracy with Bayesian Optimization and Feature Engineering after SMOTE:", accuracy_best_xgb)
# print("Best XGBoost Classification Report with Bayesian Optimization and Feature Engineering after SMOTE:\n", report_best_xgb)
# print("Best Parameters found by Bayesian Optimization:", best_params)


############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from imblearn.over_sampling import SMOTE
# import pandas as pd
# import numpy as np




# # Define the best parameters found by Bayesian Optimization
# best_params = {
#     'colsample_bytree': 0.7430037706370214,
#     'gamma': 0.45041549854138485,
#     'learning_rate': 0.07671878441695622,
#     'max_depth': 7,
#     'min_child_weight': 3,
#     'n_estimators': 100,
#     'subsample': 0.9999339357441884
# }

# # Define custom class weights
# class_weights = {0: 1, 1: 1.5, 2: 1, 3: 2, 4: 1}

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Train the best XGBoost model with custom class weights
# best_xgb = XGBClassifier(**best_params, scale_pos_weight=class_weights, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
# best_xgb.fit(X_train_smote, y_train_smote)

# # Define base models for stacking
# rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42)
# mlp = MLPClassifier(random_state=42, max_iter=300)

# # Define the stacking ensemble
# estimators = [
#     ('rf', rf),
#     ('xgb', best_xgb),
#     ('mlp', mlp)
# ]
# stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# # Train the stacking model
# stacking_clf.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred_stack = stacking_clf.predict(X_test)
# accuracy_stack = accuracy_score(y_test, y_pred_stack)
# report_stack = classification_report(y_test, y_pred_stack)

# print("Stacking Model Accuracy with Custom Class Weights and SMOTE:", accuracy_stack)
# print("Stacking Classification Report with Custom Class Weights and SMOTE:\n", report_stack)


############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# # Define the best parameters found by Bayesian Optimization
# best_params = {
#     'colsample_bytree': 0.7430037706370214,
#     'gamma': 0.45041549854138485,
#     'learning_rate': 0.07671878441695622,
#     'max_depth': 7,
#     'min_child_weight': 3,
#     'n_estimators': 100,
#     'subsample': 0.9999339357441884
# }

# # Define custom class weights
# class_weights = {0: 1, 1: 1.5, 2: 1, 3: 2, 4: 1}

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# # Train the LightGBM model with custom class weights
# lgbm = LGBMClassifier(**best_params, class_weight=class_weights, random_state=42)
# lgbm.fit(X_train_smote, y_train_smote)

# # Predict and evaluate
# y_pred_lgbm = lgbm.predict(X_test)
# accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
# report_lgbm = classification_report(y_test, y_pred_lgbm)

# print("LightGBM Model Accuracy with Custom Class Weights and SMOTE:", accuracy_lgbm)
# print("LightGBM Classification Report with Custom Class Weights and SMOTE:\n", report_lgbm)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from lightgbm import LGBMClassifier
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from imblearn.combine import SMOTETomek
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# import pandas as pd

# # Define the best parameters found by Bayesian Optimization
# best_params = {
#     'colsample_bytree': 0.7430037706370214,
#     'gamma': 0.45041549854138485,
#     'learning_rate': 0.07671878441695622,
#     'max_depth': 7,
#     'min_child_weight': 3,
#     'n_estimators': 100,
#     'subsample': 0.9999339357441884
# }

# # Define custom class weights
# class_weights = {0: 1, 1: 1.5, 2: 1, 3: 2, 4: 1}

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTETomek to the training data
# smotetomek = SMOTETomek(random_state=42)
# X_train_smotetomek, y_train_smotetomek = smotetomek.fit_resample(X_train, y_train)

# # Train the LightGBM model with custom class weights
# lgbm = LGBMClassifier(**best_params, class_weight=class_weights, random_state=42)
# lgbm.fit(X_train_smotetomek, y_train_smotetomek)

# # Define base models for stacking
# rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42)
# mlp = MLPClassifier(random_state=42, max_iter=300)

# # Define the stacking ensemble
# estimators = [
#     ('rf', rf),
#     ('lgbm', lgbm),
#     ('mlp', mlp)
# ]
# stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# # Train the stacking model
# stacking_clf.fit(X_train_smotetomek, y_train_smotetomek)

# # Predict and evaluate
# y_pred_stack = stacking_clf.predict(X_test)
# accuracy_stack = accuracy_score(y_test, y_pred_stack)
# report_stack = classification_report(y_test, y_pred_stack)

# print("Stacking Model Accuracy with SMOTETomek and Custom Class Weights:", accuracy_stack)
# print("Stacking Classification Report with SMOTETomek and Custom Class Weights:\n", report_stack)

############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

# from catboost import CatBoostClassifier
# from sklearn.ensemble import StackingClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neural_network import MLPClassifier
# from imblearn.combine import SMOTETomek
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# import pandas as pd

# # Define custom class weights
# class_weights = {0: 1, 1: 1.5, 2: 1, 3: 2, 4: 1}

# # Load and preprocess the dataset
# file_path = './data/cleaned_robot_data.csv'
# robot_data = pd.read_csv(file_path)
# X = robot_data.drop(columns=['source'])
# y = robot_data['source']

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Create polynomial features and interaction terms
# poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# X_poly = poly.fit_transform(X_scaled)

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Apply SMOTETomek to the training data
# smotetomek = SMOTETomek(random_state=42)
# X_train_smotetomek, y_train_smotetomek = smotetomek.fit_resample(X_train, y_train)

# # Train the CatBoost model with custom class weights
# catboost = CatBoostClassifier(class_weights=[class_weights[i] for i in sorted(class_weights.keys())], random_state=42, verbose=0)
# catboost.fit(X_train_smotetomek, y_train_smotetomek)

# # Define base models for stacking
# rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=10, random_state=42)
# mlp = MLPClassifier(random_state=42, max_iter=300)

# # Define the stacking ensemble
# estimators = [
#     ('rf', rf),
#     ('catboost', catboost),
#     ('mlp', mlp)
# ]
# stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# # Train the stacking model
# stacking_clf.fit(X_train_smotetomek, y_train_smotetomek)

# # Predict and evaluate
# y_pred_stack = stacking_clf.predict(X_test)
# accuracy_stack = accuracy_score(y_test, y_pred_stack)
# report_stack = classification_report(y_test, y_pred_stack)

# print("Stacking Model Accuracy with CatBoost, SMOTETomek, and Custom Class Weights:", accuracy_stack)
# print("Stacking Classification Report with CatBoost, SMOTETomek, and Custom Class Weights:\n", report_stack)


############################################################################################################
#############################################    NEW METHODS     ###########################################
############################################################################################################

from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import pandas as pd

# Load and preprocess the dataset
file_path = './data/cleaned_robot_data.csv'
robot_data = pd.read_csv(file_path)
X = robot_data.drop(columns=['source'])
y = robot_data['source']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create polynomial features and interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Train the EasyEnsembleClassifier model
easy_ensemble = EasyEnsembleClassifier(random_state=42, n_estimators=10)
easy_ensemble.fit(X_train, y_train)

# Predict and evaluate
y_pred_easy_ensemble = easy_ensemble.predict(X_test)
accuracy_easy_ensemble = accuracy_score(y_test, y_pred_easy_ensemble)
report_easy_ensemble = classification_report(y_test, y_pred_easy_ensemble)

print("EasyEnsembleClassifier Model Accuracy:", accuracy_easy_ensemble)
print("EasyEnsembleClassifier Classification Report:\n", report_easy_ensemble)

# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE

# print(f"Numpy version: {np.__version__}")
# print(f"Scikit-learn version: {RandomForestClassifier.__module__.split('.')[0]} version {RandomForestClassifier.__version__}")
# print(f"Imbalanced-learn version: {SMOTE.__module__.split('.')[0]} version {SMOTE.__version__}")