# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report
# import matplotlib.pyplot as plt

# def load_and_preprocess_data(data_path, sample_fracs, random_state=42):
#     df = pd.read_csv(data_path, header=0, dtype=str)
#     col_names = ['source'] + [f'num{i}' for i in range(1, 11)]
#     df.columns = col_names
#     df = df.iloc[1:].copy()
#     df[df.columns[1:]] = df[df.columns[1:]].astype(float)
#     df['source'] = df['source'].astype('category').cat.codes
#     df = df.sample(frac=1, random_state=random_state)
#     df_sample = pd.concat([df[df['source'] == robot_id].sample(frac=sample_fracs[robot_id], random_state=random_state) for robot_id in sample_fracs])
#     return df_sample.drop('source', axis=1), df_sample['source']

# def feature_engineering(df):
#     df['mean'] = df.mean(axis=1)
#     df['std'] = df.std(axis=1)
#     df['min'] = df.min(axis=1)
#     df['max'] = df.max(axis=1)
#     return df

# if __name__ == '__main__':
#     data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
#     sample_fracs = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1}
#     X, y = load_and_preprocess_data(data_path, sample_fracs)
#     X = feature_engineering(X)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     smote = SMOTE(random_state=42)
#     X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

#     # Hyperparameter tuning for RandomForestClassifier with more regularization
#     rf_params = {
#         'n_estimators': [50, 100],
#         'max_depth': [3, 5],  # Reduced max depth
#         'min_samples_split': [5, 10],  # Increased min_samples_split
#         'min_samples_leaf': [4, 8],  # Added min_samples_leaf
#         'max_features': ['sqrt', 'log2'],
#         'class_weight': ['balanced']
#     }    
#     # # Hyperparameter tuning for RandomForestClassifier
#     # rf_params = {
#     #     'n_estimators': [100, 200],
#     #     'max_depth': [3, 4],  # Further reduced max depth
#     #     'min_samples_split': [5, 10],  # Increased min_samples_split
#     #     'min_samples_leaf': [4, 8],  # Added min_samples_leaf
#     #     'max_features': ['sqrt', 'log2'],
#     #     'class_weight': ['balanced']
#     # }
#     rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1, verbose=2)
#     rf_grid_search.fit(X_train_balanced, y_train_balanced)
#     best_rf = rf_grid_search.best_estimator_



#     # Hyperparameter tuning for XGBoost with more regularization
#     xgb_params = {
#         'n_estimators': [50, 100],
#         'max_depth': [3, 5],  # Reduced max depth
#         'learning_rate': [0.01, 0.05],
#         'reg_alpha': [0.5, 1],  # Increased L1 regularization
#         'reg_lambda': [0.5, 1]  # Increased L2 regularization
#     }


#     # # Hyperparameter tuning for XGBoost
#     # xgb_params = {
#     #     'n_estimators': [100, 200],
#     #     'max_depth': [3, 4],  # Further reduced max depth
#     #     'learning_rate': [0.01, 0.1],
#     #     'reg_alpha': [0.5, 1],  # Further increased L1 regularization
#     #     'reg_lambda': [0.5, 1]  # Further increased L2 regularization
#     # }
#     xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42), xgb_params, cv=5, n_jobs=-1, verbose=2)
#     xgb_grid_search.fit(X_train_balanced, y_train_balanced)
#     best_xgb = xgb_grid_search.best_estimator_

#     # Ensemble model
#     ensemble = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb)], voting='soft')
#     ensemble.fit(X_train_balanced, y_train_balanced)

#     # Cross-validation
#     cv_scores = cross_val_score(ensemble, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
#     print(f'Cross-Validation Accuracy Scores: {cv_scores}')
#     print(f'Mean Cross-Validation Accuracy: {cv_scores.mean()}')
#     print(f'Standard Deviation of Cross-Validation Accuracy: {cv_scores.std()}')

#     # Learning curves
#     train_sizes, train_scores, test_scores = learning_curve(
#         ensemble, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy', n_jobs=-1,
#         train_sizes=np.linspace(0.1, 1.0, 10)
#     )

#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)

#     plt.figure()
#     plt.title('Learning Curves')
#     plt.xlabel('Training examples')
#     plt.ylabel('Score')
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color='r')
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1,
#                      color='g')
#     plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
#              label='Training score')
#     plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
#              label='Cross-validation score')

#     plt.legend(loc='best')
#     plt.show()

#     # Evaluate the ensemble model
#     y_pred_ensemble = ensemble.predict(X_test_scaled)

#     ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
#     ensemble_report = classification_report(y_test, y_pred_ensemble)

#     print(f'Ensemble Model Accuracy: {ensemble_accuracy}')
#     print('Ensemble Model Classification Report:')
#     print(ensemble_report)

##################################################################################################################################################

# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from xgboost import XGBClassifier
# from imblearn.over_sampling import SMOTE
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# def load_and_preprocess_data(data_path, sample_fracs, random_state=42):
#     df = pd.read_csv(data_path, header=0, dtype=str)
#     col_names = ['source'] + [f'num{i}' for i in range(1, 11)]
#     df.columns = col_names
#     df = df.iloc[1:].copy()
#     df[df.columns[1:]] = df[df.columns[1:]].astype(float)
#     df['source'] = df['source'].astype('category').cat.codes
#     df = df.sample(frac=1, random_state=random_state)
#     df_sample = pd.concat([df[df['source'] == robot_id].sample(frac=sample_fracs[robot_id], random_state=random_state) for robot_id in sample_fracs])
#     return df_sample.drop('source', axis=1), df_sample['source']

# def feature_engineering(df):
#     df['mean'] = df.mean(axis=1)
#     df['std'] = df.std(axis=1)
#     df['min'] = df.min(axis=1)
#     df['max'] = df.max(axis=1)
#     return df

# if __name__ == '__main__':
#     data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
#     sample_fracs = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
#     X, y = load_and_preprocess_data(data_path, sample_fracs)
#     X = feature_engineering(X)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     smote = SMOTE(random_state=42)
#     X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

#     # Hyperparameter tuning for RandomForestClassifier with more regularization
#     rf_params = {
#         'n_estimators': [50, 100],
#         'max_depth': [3, 5],
#         'min_samples_split': [5, 10],
#         'min_samples_leaf': [4, 8],
#         'max_features': ['sqrt', 'log2'],
#         'class_weight': ['balanced']
#     }
#     rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, n_jobs=-1, verbose=2)
#     rf_grid_search.fit(X_train_balanced, y_train_balanced)
#     best_rf = rf_grid_search.best_estimator_

#     # Hyperparameter tuning for XGBoost with more regularization
#     xgb_params = {
#         'n_estimators': [50, 100],
#         'max_depth': [3, 5],
#         'learning_rate': [0.01, 0.05],
#         'reg_alpha': [0.5, 1],
#         'reg_lambda': [0.5, 1]
#     }
#     xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42), xgb_params, cv=5, n_jobs=-1, verbose=2)
#     xgb_grid_search.fit(X_train_balanced, y_train_balanced)
#     best_xgb = xgb_grid_search.best_estimator_

#     # Ensemble model
#     ensemble = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb)], voting='soft')
#     ensemble.fit(X_train_balanced, y_train_balanced)

#     # Cross-validation
#     cv_scores = cross_val_score(ensemble, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
#     print(f'Cross-Validation Accuracy Scores: {cv_scores}')
#     print(f'Mean Cross-Validation Accuracy: {cv_scores.mean()}')
#     print(f'Standard Deviation of Cross-Validation Accuracy: {cv_scores.std()}')

#     # Learning curves
#     train_sizes, train_scores, test_scores = learning_curve(
#         ensemble, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy', n_jobs=-1,
#         train_sizes=np.linspace(0.1, 1.0, 10)
#     )

#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)

#     plt.figure()
#     plt.title('Learning Curves')
#     plt.xlabel('Training examples')
#     plt.ylabel('Score')
#     plt.grid()

#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color='r')
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1,
#                      color='g')
#     plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
#              label='Training score')
#     plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
#              label='Cross-validation score')

#     plt.legend(loc='best')
#     plt.show()

#     # Evaluate the ensemble model
#     y_pred_ensemble = ensemble.predict(X_test_scaled)
#     y_pred_prob = ensemble.predict_proba(X_test_scaled)

#     ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
#     ensemble_report = classification_report(y_test, y_pred_ensemble)
#     cm = confusion_matrix(y_test, y_pred_ensemble)

#     print(f'Ensemble Model Accuracy: {ensemble_accuracy}')
#     print('Ensemble Model Classification Report:')
#     print(ensemble_report)
#     print('Confusion Matrix:')
#     print(cm)

#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(data_path, sample_fracs, random_state=42):
    df = pd.read_csv(data_path, header=0, dtype=str)
    col_names = ['source'] + [f'num{i}' for i in range(1, 11)]
    df.columns = col_names
    df = df.iloc[1:].copy()
    df[df.columns[1:]] = df[df.columns[1:]].astype(float)
    df['source'] = df['source'].astype('category').cat.codes
    df = df.sample(frac=1, random_state=random_state)
    df_sample = pd.concat([df[df['source'] == robot_id].sample(frac=sample_fracs[robot_id], random_state=random_state) for robot_id in sample_fracs])
    return df_sample.drop('source', axis=1), df_sample['source']

def feature_engineering(df):
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['min'] = df.min(axis=1)
    df['max'] = df.max(axis=1)
    return df

def tune_hyperparameters(X_train, y_train, model, params):
    grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def plot_learning_curves(model, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')
    plt.legend(loc='best')
    plt.show()

def plot_roc_curve(y_test, y_pred_prob, n_classes):
    plt.figure()
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_prob, n_classes):
    plt.figure()
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_test == i, y_pred_prob[:, i])
        avg_precision = average_precision_score(y_test == i, y_pred_prob[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AP = {avg_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

def plot_feature_importance(model, X_train):
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X_train.shape[1]), indices)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

def plot_permutation_importance(model, X_train, y_train):
    result = permutation_importance(model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()

    plt.figure()
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=sorted_idx)
    plt.title("Permutation Importances (train set)")
    plt.show()

def main():
    data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
    sample_fracs = {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
    X, y = load_and_preprocess_data(data_path, sample_fracs)
    X = feature_engineering(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Hyperparameter tuning for RandomForestClassifier
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [4, 8],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced']
    }
    best_rf = tune_hyperparameters(X_train_balanced, y_train_balanced, RandomForestClassifier(random_state=42), rf_params)

    # Hyperparameter tuning for XGBoost
    xgb_params = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.05],
        'reg_alpha': [0.5, 1],
        'reg_lambda': [0.5, 1]
    }
    best_xgb = tune_hyperparameters(X_train_balanced, y_train_balanced, XGBClassifier(random_state=42), xgb_params)

    # Ensemble model
    ensemble = VotingClassifier(estimators=[('rf', best_rf), ('xgb', best_xgb)], voting='soft')
    ensemble.fit(X_train_balanced, y_train_balanced)

    # Cross-validation
    cv_scores = cross_val_score(ensemble, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
    print(f'Cross-Validation Accuracy Scores: {cv_scores}')
    print(f'Mean Cross-Validation Accuracy: {cv_scores.mean()}')
    print(f'Standard Deviation of Cross-Validation Accuracy: {cv_scores.std()}')

    # Learning curves
    plot_learning_curves(ensemble, X_train_balanced, y_train_balanced)

    # Evaluate the ensemble model
    y_pred_ensemble = ensemble.predict(X_test_scaled)
    y_pred_prob = ensemble.predict_proba(X_test_scaled)

    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    ensemble_report = classification_report(y_test, y_pred_ensemble)
    cm = confusion_matrix(y_test, y_pred_ensemble)

    print(f'Ensemble Model Accuracy: {ensemble_accuracy}')
    print('Ensemble Model Classification Report:')
    print(ensemble_report)
    print('Confusion Matrix:')
    print(cm)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plot ROC Curve
    n_classes = len(np.unique(y))
    plot_roc_curve(y_test, y_pred_prob, n_classes)

    # Plot Precision-Recall Curve
    plot_precision_recall_curve(y_test, y_pred_prob, n_classes)

    # Plot Feature Importance for RandomForest
    plot_feature_importance(best_rf, X_train)

    # Plot
    plot_permutation_importance(ensemble, X_train_scaled, y_train)

if __name__ == '__main__':
    main()