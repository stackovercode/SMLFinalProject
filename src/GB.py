# # gb_model.py
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# import os
# import sys
# from itertools import cycle

# class GradientBoosting:
#     def __init__(self, plot_dir='./plot/GB'):
#         self.plot_dir = plot_dir
#         os.makedirs(self.plot_dir, exist_ok=True)
#         self.model = None

#     @staticmethod
#     def show_progress(label, phase, current, total):
#         prog = int((current / total) * 100)
#         full = prog // 3
#         sys.stdout.write(f"\r{label} ({phase}): {prog}%  [{'█' * full}{' ' * (30 - full)}]")
#         sys.stdout.flush()

#     def load_data(self, X_train, X_test, y_train, y_test):
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test

#         # Apply SMOTE to the training data
#         smote = SMOTE(random_state=42)
#         self.X_train_smote, self.y_train_smote = smote.fit_resample(X_train, y_train)
    
#     def build_model(self):
#         self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    
#     def train_model(self):
#         print("Start fitting Gradient Boosting Model...")
#         n_estimators = self.model.get_params()['n_estimators']
#         for i in range(1, n_estimators + 1):
#             self.show_progress("Gradient Boosting Model", "Training", i, n_estimators)
#             self.model.set_params(n_estimators=i)
#             self.model.fit(self.X_train_smote, self.y_train_smote)
#         print()  

#     def evaluate_model(self):
#         total_steps = 5
#         current_step = 1

#         y_pred = self.model.predict(self.X_test)
#         self.show_progress("Gradient Boosting Model", "Evaluating", current_step, total_steps)
#         current_step += 1

#         accuracy = accuracy_score(self.y_test, y_pred)
#         report = classification_report(self.y_test, y_pred)
#         self.show_progress("Gradient Boosting Model", "Evaluating", current_step, total_steps)
#         current_step += 1
#         print("Gradient Boosting Model Accuracy with Feature Engineering after SMOTE:", accuracy)
#         print("Gradient Boosting Classification Report with Feature Engineering after SMOTE:\n", report)
        
#         # Save metrics to a CSV file
#         metrics_df = pd.DataFrame({
#             'Metric': ['Precision', 'Recall', 'F1 Score'],
#             'Value': [
#                 precision_score(self.y_test, y_pred, average='macro', zero_division=0),
#                 recall_score(self.y_test, y_pred, average='macro', zero_division=0),
#                 f1_score(self.y_test, y_pred, average='macro', zero_division=0)
#             ]
#         })
#         metrics_df.to_csv(os.path.join(self.plot_dir, 'evaluation_metrics.csv'), index=False)
#         self.show_progress("Gradient Boosting Model", "Evaluating", current_step, total_steps)
#         current_step += 1

#         # Additional plots
#         self.plot_roc_curve(y_pred_proba=self.model.predict_proba(self.X_test))
#         self.plot_confusion_matrix(y_pred)
#         self.plot_feature_importance()

#         print()  # For newline after progress bar

#     def plot_roc_curve(self, y_pred_proba):
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#         n_classes = len(np.unique(self.y_test))
#         for i in range(n_classes):
#             fpr[i], tpr[i], _ = roc_curve(self.y_test, y_pred_proba[:, i], pos_label=i)
#             roc_auc[i] = auc(fpr[i], tpr[i])
#         all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#         mean_tpr = np.zeros_like(all_fpr)
#         for i in range(n_classes):
#             mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#         mean_tpr /= n_classes
#         plt.figure(figsize=(10, 8))
#         colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
#         for i, color in zip(range(n_classes), colors):
#             plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
#         plt.plot([0, 1], [0, 1], 'k--', lw=2)
#         plt.plot(all_fpr, mean_tpr, color='navy', linestyle='-', lw=2, label='Mean ROC (area = %0.2f)' % auc(all_fpr, mean_tpr))
#         plt.xlim([-0.05, 1.05])
#         plt.ylim([-0.05, 1.05])
#         plt.xlabel('False Positive Rate', fontsize=12)
#         plt.ylabel('True Positive Rate', fontsize=12)
#         plt.title('ROC Curve for Multi-Class Classification', fontsize=14)
#         plt.legend(loc="lower right", prop={'size': 10})
#         plt.grid(alpha=0.4)
#         plt.savefig(os.path.join(self.plot_dir, 'ROC_plot.png'))
#         plt.close()

#     def plot_confusion_matrix(self, y_pred):
#         cm = confusion_matrix(self.y_test, y_pred)
#         plt.figure(figsize=(10, 7))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.xlabel('Predicted Labels')
#         plt.ylabel('True Labels')
#         plt.title('Confusion Matrix')
#         plt.savefig(os.path.join(self.plot_dir, 'confusion_plot.png'))
#         plt.close()

#     def plot_feature_importance(self):
#         feature_importance = self.model.feature_importances_
#         sorted_idx = np.argsort(feature_importance)
#         plt.figure(figsize=(10, 7))
#         plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
#         plt.yticks(range(len(sorted_idx)), sorted_idx)
#         plt.xlabel('Feature Importance')
#         plt.title('Feature Importance')
#         plt.savefig(os.path.join(self.plot_dir, 'feature_importance.png'))
#         plt.close()


import json
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
from itertools import cycle

class GradientBoosting:
    def __init__(self, plot_dir='./plot/GB'):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.model = None

    @staticmethod
    def show_progress(label, phase, current, total):
        prog = int((current / total) * 100)
        full = prog // 3
        sys.stdout.write(f"\r{label} ({phase}): {prog}%  [{'█' * full}{' ' * (30 - full)}]")
        sys.stdout.flush()

    def load_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(X_train, y_train)
    
    def build_model(self):
        self.model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    
    def train_model(self):
        print("Start fitting Gradient Boosting Model...")
        n_estimators = self.model.get_params()['n_estimators']
        for i in range(1, n_estimators + 1):
            self.show_progress("Gradient Boosting Model", "Training", i, n_estimators)
            self.model.set_params(n_estimators=i)
            self.model.fit(self.X_train_smote, self.y_train_smote)
        print()  

    def evaluate_model(self):
        total_steps = 5
        current_step = 1

        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        self.show_progress("Gradient Boosting Model", "Evaluating", current_step, total_steps)
        current_step += 1

        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='macro', zero_division=0)
        self.show_progress("Gradient Boosting Model", "Evaluating", current_step, total_steps)
        current_step += 1
        print("Gradient Boosting Model Accuracy with Feature Engineering after SMOTE:", accuracy)
        print("Gradient Boosting Classification Report with Feature Engineering after SMOTE:\n", classification_report(self.y_test, y_pred))
        
        # Save metrics to JSON file
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        with open(os.path.join(self.plot_dir, 'metrics_gb.json'), 'w') as f:
            json.dump(metrics, f)
        self.show_progress("Gradient Boosting Model", "Evaluating", current_step, total_steps)
        current_step += 1

        # Save ROC data
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(np.unique(self.y_test))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test, y_pred_proba[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        roc_data = {
            'fpr': {str(i): fpr[i].tolist() for i in fpr},
            'tpr': {str(i): tpr[i].tolist() for i in tpr},
            'roc_auc': {str(i): roc_auc[i] for i in roc_auc},
            'all_fpr': all_fpr.tolist(),
            'mean_tpr': mean_tpr.tolist()
        }
        with open(os.path.join(self.plot_dir, 'roc_data_gb.json'), 'w') as f:
            json.dump(roc_data, f)

        # Additional plots
        self.plot_roc_curve(y_pred_proba)
        self.plot_confusion_matrix(y_pred)
        self.plot_feature_importance()

        print()  # For newline after progress bar

    def plot_roc_curve(self, y_pred_proba):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = len(np.unique(self.y_test))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test, y_pred_proba[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.plot(all_fpr, mean_tpr, color='navy', linestyle='-', lw=2, label='Mean ROC (area = %0.2f)' % auc(all_fpr, mean_tpr))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve for Multi-Class Classification', fontsize=14)
        plt.legend(loc="lower right", prop={'size': 10})
        plt.grid(alpha=0.4)
        plt.savefig(os.path.join(self.plot_dir, 'ROC_plot.png'))
        plt.close()

    def plot_confusion_matrix(self, y_pred):
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.plot_dir, 'confusion_plot.png'))
        plt.close()

    def plot_feature_importance(self):
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.figure(figsize=(10, 7))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), sorted_idx)
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance')
        plt.savefig(os.path.join(self.plot_dir, 'feature_importance.png'))
        plt.close()
