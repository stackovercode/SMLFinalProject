import json
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
from itertools import cycle, product
from sklearn.model_selection import KFold
import time

class GradientBoosting:
    def __init__(self, plot_dir='./plot/GB'):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.model = None
        self.best_params_ = None
        self.best_score_ = None
        self.n_classes = None

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
        self.n_classes = len(np.unique(y_test))
        
        # Print dataset sizes
        print(f"Training data shape: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        print(f"Testing data shape: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
        
        # Save y_test to a JSON file
        with open(os.path.join(self.plot_dir, 'test_labels.json'), 'w') as f:
            json.dump(y_test.tolist(), f)


    def hyperparameter_tuning(self):
        # param_grid = {
        #     'n_estimators': [100, 200, 300],
        #     'learning_rate': [0.01, 0.1, 0.2],
        #     'max_depth': [3, 4, 5],
        #     'subsample': [0.8, 1.0, 1.2],
        #     'min_samples_split': [2, 5, 10]
        # }
        param_grid = {
            'n_estimators': [150, 225],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.5, 1.0],
            'min_samples_split': [3, 6]
        }

        total_steps = np.prod([len(v) for v in param_grid.values()]) * 3  # Number of combinations times number of CV folds
        current_step = 1

        best_score = -np.inf
        best_params = None
        all_results = []


        print("Start time for GB hyperparameter tuning: ")        
        start_time = time.time()  # Start timing


        kf = KFold(n_splits=3)

        param_list = list(product(*param_grid.values()))
        for params in param_list:
            param_dict = dict(zip(param_grid.keys(), params))
            fold_scores = []
            for train_index, val_index in kf.split(self.X_train):
                X_fold_train, X_fold_val = self.X_train[train_index], self.X_train[val_index]
                y_fold_train, y_fold_val = self.y_train[train_index], self.y_train[val_index]

                model = GradientBoostingClassifier(
                    n_estimators=param_dict['n_estimators'],
                    learning_rate=param_dict['learning_rate'],
                    max_depth=param_dict['max_depth'],
                    subsample=param_dict['subsample'],
                    min_samples_split=param_dict['min_samples_split'],
                    random_state=42
                )
                model.fit(X_fold_train, y_fold_train)
                score = model.score(X_fold_val, y_fold_val)
                fold_scores.append(score)

                self.show_progress("Gradient Boosting", "Hyperparameter Tuning", current_step, total_steps)
                current_step += 1

            mean_score = np.mean(fold_scores)
            all_results.append((param_dict, mean_score))
            if mean_score > best_score:
                best_score = mean_score
                best_params = param_dict

        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"\nTotal time for GB hyperparameter tuning: hours={elapsed_time/3600:.2f}, minutes={elapsed_time/60:.2f}, seconds={elapsed_time:.2f}")
        

        self.best_params_ = best_params
        self.best_score_ = best_score


        results_df = pd.DataFrame(all_results, columns=['params', 'mean_score'])
        results_df.to_csv(os.path.join(self.plot_dir, 'grid_search_results.csv'), index=False)
        self.plot_hyperparameter_performance(results_df)

        #print(f"Best parameters found: {self.best_params_}")  # Print the best parameters
        print(f"Best: {self.best_score_} using {self.best_params_}")
        self.model = GradientBoostingClassifier(**self.best_params_, random_state=42)

    def plot_hyperparameter_performance(self, results_df):
        results = pd.DataFrame(results_df['params'].tolist())
        results['mean_score'] = results_df['mean_score']

        for param in results.columns[:-1]:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=param, y='mean_score', data=results)
            plt.title(f'Hyperparameter Tuning - {param}')
            plt.ylabel('Mean Test Score')
            plt.xlabel(param)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'{param}_performance.png'))
            plt.close()

    def set_params(self, params):
        self.model = GradientBoostingClassifier(**params, random_state=42)
        self.best_params_ = params

    def train_model(self):
        print("Start fitting Gradient Boosting Model with best parameters...")
        print("Start time for GB traning: ")        
        start_time = time.time()  # Start timing
        
        self.model.fit(self.X_train, self.y_train)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal time for GB training: hours={elapsed_time/3600:.2f}, minutes={elapsed_time/60:.2f}, seconds={elapsed_time:.2f}")

        print()  # For newline after progress bar

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
        print("Gradient Boosting Model Accuracy:", accuracy)
        print("Gradient Boosting Classification Report:\n", classification_report(self.y_test, y_pred))
        
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

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        roc_data = {
            'fpr': {str(i): fpr[i].tolist() for i in fpr},
            'tpr': {str(i): tpr[i].tolist() for i in tpr},
            'roc_auc': {str(i): roc_auc[i] for i in roc_auc},
            'macro_fpr': fpr["macro"].tolist(),
            'macro_tpr': tpr["macro"].tolist(),
            'macro_roc_auc': roc_auc["macro"]
        }
        with open(os.path.join(self.plot_dir, 'roc_data_gb.json'), 'w') as f:
            json.dump(roc_data, f)

        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_roc_curve(fpr, tpr, roc_auc)
        self.plot_confusion_matrix(cm, classes=np.unique(self.y_test))
        self.plot_feature_importance()

        print()  # For newline after progress bar

        # # Save ROC data
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # n_classes = len(np.unique(self.y_test))
        # for i in range(n_classes):
        #     fpr[i], tpr[i], _ = roc_curve(self.y_test, y_pred_proba[:, i], pos_label=i)
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(n_classes):
        #     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # mean_tpr /= n_classes

        # roc_data = {
        #     'fpr': {str(i): fpr[i].tolist() for i in fpr},
        #     'tpr': {str(i): tpr[i].tolist() for i in tpr},
        #     'roc_auc': {str(i): roc_auc[i] for i in roc_auc},
        #     'all_fpr': all_fpr.tolist(),
        #     'mean_tpr': mean_tpr.tolist()
        # }
        # with open(os.path.join(self.plot_dir, 'roc_data_gb.json'), 'w') as f:
        #     json.dump(roc_data, f)


        # mean_tpr /= self.n_classes
        # fpr["macro"] = all_fpr
        # tpr["macro"] = mean_tpr
        # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # # y_pred = np.argmax(y_pred_proba, axis=1)
        # # y_test_class = np.argmax(self.y_test, axis=1)
        # cm = confusion_matrix(self.y_test, y_pred)
        # #np.save(os.path.join(self.plot_dir, 'gb_confusion_matrix.npy'), cm) 
    
        # # Additional plots
        # #self.plot_roc_curve(y_pred_proba)
        # self.plot_roc_curve(fpr, tpr, roc_auc)
        # self.plot_confusion_matrix(cm, classes=np.unique(self.y_test))
        # self.plot_feature_importance()

        # print()  # For newline after progress bar
    

    # def plot_roc_curve(self, y_pred_proba):
    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     n_classes = len(np.unique(self.y_test))
    #     for i in range(n_classes):
    #         fpr[i], tpr[i], _ = roc_curve(self.y_test, y_pred_proba[:, i], pos_label=i)
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    #     mean_tpr = np.zeros_like(all_fpr)
    #     for i in range(n_classes):
    #         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    #     mean_tpr /= n_classes
    #     plt.figure(figsize=(10, 8))
    #     colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
    #     for i, color in zip(range(n_classes), colors):
    #         plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    #     plt.plot([0, 1], [0, 1], 'k--', lw=2)
    #     plt.plot(all_fpr, mean_tpr, color='navy', lw=2, linestyle='--', label='Mean ROC curve (area = {0:0.2f})'.format(auc(all_fpr, mean_tpr)))
    #     plt.xlim([-0.05, 1.05])
    #     plt.ylim([-0.05, 1.05])
    #     # plt.xlabel('False Positive Rate', fontsize=plt.rcParams['axes.labelsize'])
    #     # plt.ylabel('True Positive Rate', fontsize=plt.rcParams['axes.labelsize'])
    #     # plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=plt.rcParams['axes.titlesize'])
    #     # plt.legend(loc="lower right",  prop={'size': plt.rcParams['legend.fontsize']})
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic (ROC)')
    #     plt.legend(loc="lower right")
    #     plt.grid(alpha=0.4)
    #     plt.savefig(os.path.join(self.plot_dir, 'roc_curves.png'))
    #     plt.close()
    
    # def plot_roc_curve(self, fpr, tpr, roc_auc):
    #     plt.figure(figsize=(10, 8))
    #     colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    #     for i, color in zip(range(self.n_classes), colors):
    #         plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
    #     plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='--', lw=2, label=f'Macro-average ROC curve (area = {roc_auc["macro"]:0.2f})')
    #     plt.plot([0, 1], [0, 1], 'k--', lw=2)
    #     plt.xlim([-0.05, 1.05])
    #     plt.ylim([-0.05, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic (ROC)')
    #     plt.legend(loc="lower right")
    #     plt.tight_layout()
    #     plt.grid(alpha=0.4)
    #     plt.savefig(os.path.join(self.plot_dir, 'roc_curves.png'))
    #     plt.close()


    def plot_roc_curve(self, fpr, tpr, roc_auc):
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
        plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='--', lw=2, label=f'Macro-average ROC curve (area = {roc_auc["macro"]:0.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.grid(alpha=0.4)
        plt.savefig(os.path.join(self.plot_dir, 'roc_curves.png'))
        plt.close()
        
    def plot_confusion_matrix(self,cm, classes):
        #cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_feature_importance(self):
        feature_importance = self.model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'feature_importance.png'))
        plt.close()

    def get_feature_importance(self):
        feature_importance = self.model.feature_importances_
        return feature_importance
