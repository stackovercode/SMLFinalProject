
    # def evaluate_model(self):
    #     total_steps = 5
    #     current_step = 1

    #     y_pred_proba = self.model.predict(self.X_test)
    #     self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
    #     current_step += 1

    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     all_fpr = np.unique(np.concatenate([roc_curve(self.y_test[:, i], y_pred_proba[:, i])[0] for i in range(self.n_classes)]))
    #     mean_tpr = np.zeros_like(all_fpr)
    #     for i in range(self.n_classes):
    #         fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred_proba[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        
    #     self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
    #     current_step += 1

    #     mean_tpr /= self.n_classes
    #     fpr["macro"] = all_fpr
    #     tpr["macro"] = mean_tpr
    #     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    #     y_pred = np.argmax(y_pred_proba, axis=1)
    #     y_test_class = np.argmax(self.y_test, axis=1)
    #     cm = confusion_matrix(y_test_class, y_pred)

    #     self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
    #     current_step += 1

    #     precision = precision_score(y_test_class, y_pred, average='macro')
    #     recall = recall_score(y_test_class, y_pred, average='macro')
    #     f1 = f1_score(y_test_class, y_pred, average='macro')
    #     accuracy = np.mean(y_test_class == y_pred)
    #     metrics = {'precision': precision, 'recall': recall, 'f1_score': f1, 'accuracy': accuracy}

    #     self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
    #     current_step += 1

    #     self.plot_confusion_matrix(cm, classes=np.unique(y_test_class))
    #     self.plot_roc_curves(fpr, tpr, roc_auc)
        
    #     print()
    #     print(f"Precision: {precision:.4f}")
    #     print(f"Recall: {recall:.4f}")
    #     print(f"F1 Score: {f1:.4f}")
    #     print(f"Accuracy: {accuracy:.4f}")

    # def plot_confusion_matrix(self, cm, classes):
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    #     plt.xlabel('Predicted Label')
    #     plt.ylabel('True Label')
    #     plt.title('Confusion Matrix')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.plot_dir, 'confusion_matrix.png'))
    #     plt.close()

    # def plot_roc_curves(self, fpr, tpr, roc_auc):
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
    #     plt.savefig(os.path.join(self.plot_dir, 'roc_curves.png'))
    #     plt.close()

    # def plot_training_history(self):
    #     if self.history is None:
    #         print("No training history found.")
    #         return

    #     plt.figure(figsize=(10, 8))
    #     plt.plot(self.history.history['loss'], label='Training Loss')
    #     plt.plot(self.history.history['val_loss'], label='Validation Loss')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Loss')
    #     plt.title('Training and Validation Loss')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.plot_dir, 'training_history.png'))
    #     plt.close()
        
    #     plt.figure(figsize=(10, 8))
    #     plt.plot(self.history.history['accuracy'], label='Training Accuracy')
    #     plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Accuracy')
    #     plt.title('Training and Validation Accuracy')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.plot_dir, 'training_accuracy.png'))
    #     plt.close()
        
        
    # def evaluate_model(self):
    #     total_steps = 5
    #     current_step = 1

    #     y_pred_proba = self.model.predict(self.X_test)
    #     self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
    #     current_step += 1

    #     fpr = dict()
    #     tpr = dict()
    #     roc_auc = dict()
    #     all_fpr = np.unique(np.concatenate([roc_curve(self.y_test[:, i], y_pred_proba[:, i])[0] for i in range(self.n_classes)]))
    #     mean_tpr = np.zeros_like(all_fpr)
    #     for i in range(self.n_classes):
    #         fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred_proba[:, i])
    #         roc_auc[i] = auc(fpr[i], tpr[i])
    #         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    #     mean_tpr /= self.n_classes

    #     roc_data = {
    #         'fpr': {str(i): fpr[i].tolist() for i in fpr},
    #         'tpr': {str(i): tpr[i].tolist() for i in tpr},
    #         'roc_auc': {str(i): roc_auc[i] for i in roc_auc},
    #         'all_fpr': all_fpr.tolist(),
    #         'mean_tpr': mean_tpr.tolist()
    #     }
    #     with open(os.path.join(self.plot_dir, 'roc_data_nn.json'), 'w') as f:
    #         json.dump(roc_data, f)
        
    #     self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
    #     current_step += 1
        
    #     test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
    #     self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
    #     current_step += 1
    #     print('Test loss:', test_loss)
    #     print('Test accuracy:', test_acc)

    #     y_pred_classes = np.argmax(y_pred_proba, axis=1)
    #     y_test_classes = np.argmax(self.y_test, axis=1)
        
    #     cm = confusion_matrix(y_test_classes, y_pred_classes)
    #     # np.save(os.path.join(self.plot_dir, 'nn_confusion_matrix.npy'), cm) 
        
    #     plt.figure(figsize=(10, 7))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #     plt.xlabel('Predicted Labels')
    #     plt.ylabel('True Labels')
    #     plt.title('Confusion Matrix')
    #     plt.savefig(os.path.join(self.plot_dir, 'confusion_plot.png'))
    #     plt.close()
        
    #     self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
    #     current_step += 1

    #     precision = precision_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
    #     recall = recall_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
    #     f1 = f1_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
    #     metrics = {
    #         'accuracy': test_acc,
    #         'precision': precision,
    #         'recall': recall,
    #         'f1_score': f1
    #     }
    #     with open(os.path.join(self.plot_dir, 'metrics_nn.json'), 'w') as f:
    #         json.dump(metrics, f)

    #     print(f'Precision: {precision}')
    #     print(f'Recall: {recall}')
    #     print(f'F1 Score: {f1}')

    # def plot_training_history(self):
    #     plt.figure()
    #     plt.plot(self.history.history['accuracy'], label='accuracy')
    #     plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.ylim([0, 1])
    #     plt.title('Training History')
    #     plt.legend(loc='lower right')
    #     plt.savefig(os.path.join(self.plot_dir, 'training_history.png'))
    #     plt.close()



# import tensorflow as tf
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os
# import sys
# from itertools import cycle
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
# import json
# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import ParameterGrid, KFold
# import time
# import warnings

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all non-error messages
# warnings.filterwarnings("ignore", message="Do not pass an `input_shape`/`input_dim` argument to a layer")
# warnings.filterwarnings("ignore")

# # def create_model(activation='relu', optimizer='adam', learning_rate=0.001, dropout_rate=0.2, input_shape=5, n_classes=5):
# def create_model(activation='tanh', optimizer='sgd', learning_rate=0.001, dropout_rate=0.5, input_shape=5, n_classes=5):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(input_shape,)))
#     model.add(tf.keras.layers.Dense(128, activation=activation))
#     model.add(tf.keras.layers.Dropout(dropout_rate))
#     model.add(tf.keras.layers.Dense(64, activation=activation))
#     model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     if optimizer == 'adam':
#         optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     elif optimizer == 'sgd':
#         optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

#     return model


# class NeuralNetwork:
#     def __init__(self, plot_dir='./plot/NN'):
#         self.plot_dir = plot_dir
#         os.makedirs(self.plot_dir, exist_ok=True)
#         self.model = None
#         self.best_params_ = None
#         self.history = None
#         print("Available devices:")
#         self.set_gpu()

#     def set_gpu(self):
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             try:
#                 tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#                 tf.config.experimental.set_memory_growth(gpus[0], True)
#                 print("Using GPU:", gpus[0])
#             except RuntimeError as e:
#                 print(e)

#     @staticmethod
#     def show_progress(label, phase, current, total):
#         prog = int((current / total) * 100)
#         full = prog // 3
#         sys.stdout.write(f"\r{label} ({phase}): {prog}%  [{'█' * full}{' ' * (30 - full)}]")
#         sys.stdout.flush()

#     def load_data(self, X_train, X_test, y_train, y_test):
#         global input_shape, n_classes
#         y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
#         y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
#         self.n_classes = y_train_bin.shape[1]

#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train_bin
#         self.y_test = y_test_bin
#         input_shape = self.X_train.shape[1]
#         n_classes = self.n_classes
#         # Print dataset sizes
#         print(f"Training data shape: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
#         print(f"Testing data shape: X_test={self.X_test.shape}, y_test={self.y_test.shape}")

#         # Save y_test to a JSON file
#         with open(os.path.join(self.plot_dir, 'test_labels.json'), 'w') as f:
#             json.dump(y_test.tolist(), f)

#     def hyperparameter_tuning(self):
#         param_grid = {
#             'model__activation': ['leaky_relu', 'tanh'],
#             'model__optimizer': ['adam', 'sgd'],
#             'model__learning_rate': [0.002, 0.005],
#             'model__dropout_rate': [0.1, 0.2],
#             'batch_size': [32, 64],
#             'epochs': [100, 150]
#         }

#         total_steps = np.prod([len(v) for v in param_grid.values()]) * 3  # Number of combinations times number of CV folds
#         current_step = 1
#         print(f"Start training with {total_steps} steps and current step: {current_step}")

#         best_score = -np.inf
#         best_params = None
#         all_results = []

#         model = KerasClassifier(
#             model=create_model,
#             verbose=0,
#             input_shape=input_shape,
#             n_classes=n_classes
#         )

#         start_time = time.time()  # Start timing

#         # K-Fold cross-validation
#         kf = KFold(n_splits=3)
#         param_list = list(ParameterGrid(param_grid))

#         for params in param_list:
#             fold_scores = []
#             for train_index, val_index in kf.split(self.X_train):
#                 X_fold_train, X_fold_val = self.X_train[train_index], self.X_train[val_index]
#                 y_fold_train, y_fold_val = self.y_train[train_index], self.y_train[val_index]

#                 model = create_model(
#                     activation=params['model__activation'],
#                     optimizer=params['model__optimizer'],
#                     learning_rate=params['model__learning_rate'],
#                     dropout_rate=params['model__dropout_rate'],
#                     input_shape=input_shape,
#                     n_classes=n_classes
#                 )
#                 model.fit(X_fold_train, y_fold_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

#                 score = model.evaluate(X_fold_val, y_fold_val, verbose=0)[1]
#                 fold_scores.append(score)

#                 self.show_progress("Neural Network", "Hyperparameter Tuning", current_step, total_steps)
#                 current_step += 1

#             mean_score = np.mean(fold_scores)
#             all_results.append((params, mean_score))
#             if mean_score > best_score:
#                 best_score = mean_score
#                 best_params = params

#         end_time = time.time()

#         elapsed_time = end_time - start_time
#         print(f"\nTotal time for hyperparameter tuning: {elapsed_time:.2f} seconds")
#         print(f"Best: {best_score} using {best_params}")

#         self.best_params_ = best_params
#         model_params = {k.replace('model__', ''): v for k, v in self.best_params_.items() if 'model__' in k}
#         self.best_params_ = model_params
#         self.model = create_model(**self.best_params_, input_shape=input_shape, n_classes=n_classes)

#         results_df = pd.DataFrame(all_results, columns=['params', 'mean_score'])
#         results_df.to_csv(os.path.join(self.plot_dir, 'grid_search_results.csv'), index=False)
#         self.plot_hyperparameter_performance(results_df)

#     def plot_hyperparameter_performance(self, results_df):

#         def strip_model_prefix(param_dict):
#             return {k.replace('model__', ''): v for k, v in param_dict.items()}

#         results_df['params'] = results_df['params'].apply(strip_model_prefix)
#         results = pd.DataFrame(results_df['params'].tolist())
#         results['mean_score'] = results_df['mean_score']

#         expected_params = ['activation', 'optimizer', 'dropout_rate', 'batch_size', 'epochs']
#         for param in expected_params:
#             if param not in results.columns:
#                 print(f"Warning: {param} not found in results. Skipping plot for {param}.")
#                 continue
#             plt.figure(figsize=(10, 6))
#             sns.boxplot(x=param, y='mean_score', data=results)
#             plt.title(f'Hyperparameter Tuning - {param}')
#             plt.ylabel('Mean Test Score')
#             plt.xlabel(param.replace('_', ' ').capitalize())
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             plt.savefig(os.path.join(self.plot_dir, f'{param}_performance.png'))
#             plt.close()

#     def build_model(self):
#         self.model = create_model(input_shape=input_shape, n_classes=n_classes)

#     def set_params(self, params):
#         model_params = {k.replace('model__', ''): v for k, v in params.items() if 'model__' in k}
#         self.model = create_model(**model_params, input_shape=input_shape, n_classes=n_classes)
#         self.best_params_ = model_params

#     def train_model(self, epochs=100):
#         if not self.model:
#             self.build_model()
#         checkpoint = tf.keras.callbacks.ModelCheckpoint('NN_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#         total_steps = epochs
#         current_step = 1
#         print("Start training Neural Network Model...")
#         for epoch in range(epochs):
#             self.show_progress("Neural Network", "Training", current_step, total_steps)
#             current_step += 1
#             self.history = self.model.fit(self.X_train, self.y_train, epochs=1, validation_split=0.2, callbacks=[checkpoint, early_stopping], verbose=0)

#         print()  # For newline after progress bar

#     def evaluate_model(self):
#         total_steps = 5
#         current_step = 1

#         y_pred_proba = self.model.predict(self.X_test)
#         self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
#         current_step += 1

#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#         all_fpr = np.unique(np.concatenate([roc_curve(self.y_test[:, i], y_pred_proba[:, i])[0] for i in range(self.n_classes)]))
#         mean_tpr = np.zeros_like(all_fpr)
#         for i in range(self.n_classes):
#             fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred_proba[:, i])
#             roc_auc[i] = auc(fpr[i], tpr[i])
#             mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

#         mean_tpr /= self.n_classes
#         fpr["macro"] = all_fpr
#         tpr["macro"] = mean_tpr
#         roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#         roc_data = {
#             'fpr': {str(i): fpr[i].tolist() for i in fpr},
#             'tpr': {str(i): tpr[i].tolist() for i in tpr},
#             'roc_auc': {str(i): roc_auc[i] for i in roc_auc},
#             'all_fpr': all_fpr.tolist(),
#             'mean_tpr': mean_tpr.tolist()
#         }
#         with open(os.path.join(self.plot_dir, 'roc_data_nn.json'), 'w') as f:
#             json.dump(roc_data, f)

#         self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
#         current_step += 1

#         test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
#         self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
#         current_step += 1
#         print('Test loss:', test_loss)
#         print('Test accuracy:', test_acc)

#         y_pred_classes = np.argmax(y_pred_proba, axis=1)
#         y_test_classes = np.argmax(self.y_test, axis=1)
#         cm = confusion_matrix(y_test_classes, y_pred_classes)

#         plt.figure(figsize=(10, 7))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.xlabel('Predicted Labels')
#         plt.ylabel('True Labels')
#         plt.title('Confusion Matrix')
#         plt.savefig(os.path.join(self.plot_dir, 'confusion_plot.png'))
#         plt.close()

#         self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
#         current_step += 1

#         precision = precision_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
#         recall = recall_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
#         f1 = f1_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
#         metrics = {
#             'accuracy': test_acc,
#             'precision': precision,
#             'recall': recall,
#             'f1_score': f1
#         }
#         with open(os.path.join(self.plot_dir, 'metrics_nn.json'), 'w') as f:
#             json.dump(metrics, f)

#         print(f'Precision: {precision}')
#         print(f'Recall: {recall}')
#         print(f'F1 Score: {f1}')

#         self.plot_confusion_matrix(cm, classes=np.unique(y_test_classes))
#         self.plot_roc_curves(fpr, tpr, roc_auc)

#         print()
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1 Score: {f1:.4f}")
#         print(f"Accuracy: {accuracy:.4f}")

#     def plot_confusion_matrix(self, cm, classes):
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
#         plt.xlabel('Predicted Label')
#         plt.ylabel('True Label')
#         plt.title('Confusion Matrix')
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.plot_dir, 'confusion_matrix.png'))
#         plt.close()

#     def plot_roc_curves(self, fpr, tpr, roc_auc):
#         plt.figure(figsize=(10, 8))
#         colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#         for i, color in zip(range(self.n_classes), colors):
#             plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
#         plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='--', lw=2, label=f'Macro-average ROC curve (area = {roc_auc["macro"]:0.2f})')
#         plt.plot([0, 1], [0, 1], 'k--', lw=2)
#         plt.xlim([-0.05, 1.05])
#         plt.ylim([-0.05, 1.05])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic (ROC)')
#         plt.legend(loc="lower right")
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.plot_dir, 'roc_curves.png'))
#         plt.close()

#     def plot_training_history(self):
#         if self.history is None:
#             print("No training history found.")
#             return

#         plt.figure(figsize=(10, 8))
#         plt.plot(self.history.history['loss'], label='Training Loss')
#         plt.plot(self.history.history['val_loss'], label='Validation Loss')
#         plt.xlabel('Epochs')
#         plt.ylabel('Loss')
#         plt.title('Training and Validation Loss')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.plot_dir, 'training_history.png'))
#         plt.close()

#         plt.figure(figsize=(10, 8))
#         plt.plot(self.history.history['accuracy'], label='Training Accuracy')
#         plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
#         plt.xlabel('Epochs')
#         plt.ylabel('Accuracy')
#         plt.title('Training and Validation Accuracy')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.plot_dir, 'training_accuracy.png'))
#         plt.close()



import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, classification_report
import json
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import ParameterGrid, KFold
import time
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all non-error messages
warnings.filterwarnings("ignore", message="Do not pass an `input_shape`/`input_dim` argument to a layer")
warnings.filterwarnings("ignore")

def create_model(activation='tanh', optimizer='adam', learning_rate=0.005, dropout_rate=0.2, input_shape=5, n_classes=5):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(input_shape,)))
    model.add(tf.keras.layers.Dense(64, activation=activation))  # Reduced number of neurons
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(32, activation=activation))  # Reduced number of neurons
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd': 
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

class NeuralNetwork:
    def __init__(self, plot_dir='./plot/NN'):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.model = None
        self.best_params_ = None
        self.history = None
        print("Available devices:")
        self.set_gpu()
        
    def set_gpu(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("Using GPU:", gpus[0])
            except RuntimeError as e:
                print(e)

    @staticmethod
    def show_progress(label, phase, current, total):
        prog = int((current / total) * 100)
        full = prog // 3
        sys.stdout.write(f"\r{label} ({phase}): {prog}%  [{'█' * full}{' ' * (30 - full)}]")
        sys.stdout.flush()
    
    def load_data(self, X_train, X_test, y_train, y_test):
        global input_shape, n_classes
        y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        self.n_classes = y_train_bin.shape[1]
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train_bin
        self.y_test = y_test_bin
        input_shape = self.X_train.shape[1]
        n_classes = self.n_classes
        # Print dataset sizes
        print(f"Training data shape: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        print(f"Testing data shape: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
        
        # Save y_train and y_test to JSON files
        with open(os.path.join(self.plot_dir, 'test_labels.json'), 'w') as f:
            json.dump(y_test.tolist(), f)

    def hyperparameter_tuning(self):
        param_grid = {
            'model__activation': ['relu', 'tanh'],  # Reduced number of activations
            'model__optimizer': ['adam', 'sgd'],
            'model__learning_rate': [0.005, 0.01],
            'model__dropout_rate': [0.1, 0.2],
            'batch_size': [32],
            'epochs': [30]  # Reduced number of epochs for tuning
        }
        
        total_steps = np.prod([len(v) for v in param_grid.values()]) * 3  # Number of combinations times number of CV folds
        current_step = 1
        print(f"Start training with {total_steps} steps and current step: {current_step}")

        best_score = -np.inf
        best_params = None
        all_results = []
        
        model = KerasClassifier(
            model=create_model,
            verbose=0,
            input_shape=input_shape,
            n_classes=n_classes
        )
        
        print("Start time for NN hyperparameter tuning: ")        
        start_time = time.time()  # Start timing
        
        kf = KFold(n_splits=3)
        param_list = list(ParameterGrid(param_grid))
        
        for params in param_list:
            fold_scores = []
            for train_index, val_index in kf.split(self.X_train):
                X_fold_train, X_fold_val = self.X_train[train_index], self.X_train[val_index]
                y_fold_train, y_fold_val = self.y_train[train_index], self.y_train[val_index]
                
                model = create_model(
                    activation=params['model__activation'],
                    optimizer=params['model__optimizer'],
                    dropout_rate=params['model__dropout_rate'],
                    input_shape=input_shape,
                    n_classes=n_classes
                )
                model.fit(X_fold_train, y_fold_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                
                score = model.evaluate(X_fold_val, y_fold_val, verbose=0)[1]
                fold_scores.append(score)

                self.show_progress("Neural Network", "Hyperparameter Tuning", current_step, total_steps)
                current_step += 1
                
            mean_score = np.mean(fold_scores)
            all_results.append((params, mean_score))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal time for hyperparameter tuning: minutes={elapsed_time/60:.2f}, seconds={elapsed_time:.2f}")
        print(f"Best: {best_score} using {best_params}")

        self.best_params_ = best_params
        model_params = {k.replace('model__', ''): v for k, v in self.best_params_.items() if 'model__' in k}
        self.best_params_ = model_params
        self.model = create_model(**self.best_params_, input_shape=input_shape, n_classes=n_classes)

        results_df = pd.DataFrame(all_results, columns=['params', 'mean_score'])
        results_df.to_csv(os.path.join(self.plot_dir, 'grid_search_results.csv'), index=False)
        self.plot_hyperparameter_performance(results_df)
    
    def plot_hyperparameter_performance(self, results_df):
        
        def strip_model_prefix(param_dict):
            return {k.replace('model__', ''): v for k, v in param_dict.items()}

        results_df['params'] = results_df['params'].apply(strip_model_prefix)
        results = pd.DataFrame(results_df['params'].tolist())
        results['mean_score'] = results_df['mean_score']

        expected_params = ['activation', 'optimizer', 'dropout_rate', 'batch_size', 'epochs']
        for param in expected_params:
            if param not in results.columns:
                print(f"Warning: {param} not found in results. Skipping plot for {param}.")
                continue
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=param, y='mean_score', data=results)
            plt.title(f'Hyperparameter Tuning - {param}')
            plt.ylabel('Mean Test Score')
            plt.xlabel(param.replace('_', ' ').capitalize())
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'{param}_performance.png'))
            plt.close()
            
    def build_model(self):
        self.model = create_model(input_shape=input_shape, n_classes=n_classes)

    def set_params(self, params):
        model_params = {k.replace('model__', ''): v for k, v in params.items() if 'model__' in k}
        self.model = create_model(**model_params, input_shape=input_shape, n_classes=n_classes)
        self.best_params_ = model_params
    
    def train_model(self, epochs=50):  # Reduced number of epochs for training
        print("Start time for NN training: ")        
        start_time = time.time()  # Start timing

        if not self.model:
            self.build_model()
        checkpoint = tf.keras.callbacks.ModelCheckpoint('NN_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Reduced patience

        total_steps = epochs
        current_step = 1
        print("Start training Neural Network Model...")
        
        self.history = self.model.fit(
            self.X_train, 
            self.y_train, 
            epochs=epochs, 
            validation_split=0.2, 
            callbacks=[checkpoint, early_stopping], 
            verbose=0
        )
            
        print()  # For newline after progress bar
        
        print(f"Training completed. Plotting training history...")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\nTotal time for NN training: hours={elapsed_time/3600:.2f}, minutes={elapsed_time/60:.2f}, seconds={elapsed_time:.2f}")
        
        self.plot_training_history()
        print("Training history plotted.")

    def evaluate_model(self):
        total_steps = 5
        current_step = 1

        y_pred_proba = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        print("Error handling: (Predicted class distribution)")
        print(np.bincount(y_pred_classes))

        self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
        current_step += 1

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        all_fpr = np.unique(np.concatenate([roc_curve(self.y_test[:, i], y_pred_proba[:, i])[0] for i in range(self.n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        roc_data = {
            'fpr': {str(i): fpr[i].tolist() for i in fpr},
            'tpr': {str(i): tpr[i].tolist() for i in tpr},
            'roc_auc': {str(i): roc_auc[i] for i in roc_auc},
            'all_fpr': all_fpr.tolist(),
            'mean_tpr': mean_tpr.tolist()
        }
        
        with open(os.path.join(self.plot_dir, 'roc_data_nn.json'), 'w') as f:
            json.dump(roc_data, f)
            
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
        current_step += 1
            
        mean_tpr /= self.n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        y_pred = np.argmax(y_pred_proba, axis=1)
        y_test_class = np.argmax(self.y_test, axis=1)
        cm = confusion_matrix(y_test_class, y_pred)

        self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
        current_step += 1

        precision = precision_score(y_test_class, y_pred, average='macro')
        recall = recall_score(y_test_class, y_pred, average='macro')
        f1 = f1_score(y_test_class, y_pred, average='macro')
        accuracy = np.mean(y_test_class == y_pred)
        
        print("Neural Network Classification Report:\n", classification_report(y_test_class, y_pred))
        
        metrics = {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        with open(os.path.join(self.plot_dir, 'metrics_nn.json'), 'w') as f:
            json.dump(metrics, f)
            
        self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
        current_step += 1

        self.plot_confusion_matrix(cm, classes=np.unique(y_test_class))
        self.plot_roc_curves(fpr, tpr, roc_auc)
        
        print()
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

    def plot_confusion_matrix(self, cm, classes):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_roc_curves(self, fpr, tpr, roc_auc):
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
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

    def plot_training_history(self):
        plt.figure(figsize=(10, 8))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.ylim(0, max(max(self.history.history['loss']), max(self.history.history['val_loss'])))
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_history.png'))
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.ylim(0, 1)  # Accuracy values typically range from 0 to 1
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_accuracy.png'))
        plt.close()
