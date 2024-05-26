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
# from sklearn.model_selection import GridSearchCV

# import warnings
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all non-error messages
# # Suppress specific warnings
# warnings.filterwarnings("ignore", message="Do not pass an `input_shape`/`input_dim` argument to a layer")
# # Suppress all warnings (if needed, use with caution)
# warnings.filterwarnings("ignore")


# def create_model(activation='relu', optimizer='adam', dropout_rate=0.2, input_shape=5, n_classes=5):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(input_shape,)))
#     model.add(tf.keras.layers.Dense(128, activation=activation, input_shape=(input_shape,)))
#     model.add(tf.keras.layers.Dropout(dropout_rate))
#     model.add(tf.keras.layers.Dense(64, activation=activation))
#     model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# class NeuralNetwork:
#     def __init__(self, plot_dir='./plot/NN'):
#         self.plot_dir = plot_dir
#         os.makedirs(self.plot_dir, exist_ok=True)
#         self.model = None
#         print("Available devices:")
#         self.set_gpu()
#         #print(tf.config.list_physical_devices())
        
#     def set_gpu(self):
#         # Check available GPUs
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             try:
#                 # Set GPU as the default device
#                 tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#                 tf.config.experimental.set_memory_growth(gpus[0], True)
#                 print("Using GPU:", gpus[0])
#             except RuntimeError as e:
#                 # Memory growth must be set before GPUs have been initialized
#                 print(e)
#     @staticmethod
#     def show_progress(label, phase, current, total):
#         prog = int((current / total) * 100)
#         full = prog // 3
#         sys.stdout.write(f"\r{label} ({phase}): {prog}%  [{'█' * full}{' ' * (30 - full)}]")
#         sys.stdout.flush()
    
#     def load_data(self, X_train, X_test, y_train, y_test):
#         global input_shape, n_classes
#         # Binarize the labels for multi-class classification
#         y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
#         y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
#         self.n_classes = y_train_bin.shape[1]
        
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train_bin
#         self.y_test = y_test_bin
#         input_shape = self.X_train.shape[1]
#         n_classes = self.n_classes
#         print("Input shape:", input_shape)
#         print("Number of classes:", self.n_classes)


#     # def hyperparameter_tuning(self):
#     #     total_steps = 96  # Based on the grid size: 2 * 2 * 2 * 2 * 2 = 32
#     #     current_step = 1
        
#     #     model = KerasClassifier(
#     #         model=create_model,
#     #         verbose=0,
#     #         input_shape=input_shape,
#     #         n_classes=n_classes
#     #     )
        
#     #     param_grid = {
#     #         'model__activation': ['relu', 'tanh'],
#     #         'model__optimizer': ['adam', 'rmsprop'],
#     #         'model__dropout_rate': [0.2, 0.3],
#     #         'batch_size': [32, 64],
#     #         'epochs': [50, 100]
#     #     }
        
#     #     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
#     #     for params in grid.param_grid:
#     #         self.show_progress("Neural Network", "Hyperparameter Tuning", current_step, total_steps)
#     #         current_step += 1
#     #     grid_result = grid.fit(self.X_train, self.y_train)
        
#     #     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     #     self.best_params_ = grid_result.best_params_
#     #     self.model = grid_result.best_estimator_
        
#     #     # Save the grid search results
#     #     results_df = pd.DataFrame(grid_result.cv_results_)
#     #     results_df.to_csv(os.path.join(self.plot_dir, 'grid_search_results.csv'), index=False)
#     #     self.plot_hyperparameter_performance(results_df)
#     def hyperparameter_tuning(self):
#         param_grid = {
#             'model__activation': ['relu', 'tanh'],
#             'model__optimizer': ['adam', 'rmsprop'],
#             'model__dropout_rate': [0.2, 0.3],
#             'batch_size': [32, 64],
#             'epochs': [50, 100]
#         }
        
#         total_steps = np.prod([len(v) for v in param_grid.values()]) * 3  # Number of combinations times number of CV folds
#         current_step = 1
        
#         model = KerasClassifier(
#             model=create_model,
#             verbose=0,
#             input_shape=input_shape,
#             n_classes=n_classes
#         )
        
#         grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, return_train_score=True)
        
#         start_time = time.time()  # Start timing
        
#         for params in ParameterGrid(param_grid):
#             self.show_progress("Neural Network", "Hyperparameter Tuning", current_step, total_steps)
#             current_step += 1
        
#         grid_result = grid.fit(self.X_train, self.y_train)
#         end_time = time.time()  # End timing
        
#         elapsed_time = end_time - start_time
#         print(f"\nTotal time for hyperparameter tuning: {elapsed_time:.2f} seconds")

#         print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#         self.best_params_ = grid_result.best_params_
#         self.model = grid_result.best_estimator_
        
#     def plot_hyperparameter_performance(self, results_df):
#         # Plot the performance of each parameter combination
#         for param in ['param_model__activation', 'param_model__optimizer', 'param_model__dropout_rate', 'param_batch_size', 'param_epochs']:
#             plt.figure(figsize=(10, 6))
#             sns.boxplot(x=param, y='mean_test_score', data=results_df)
#             plt.title(f'Hyperparameter Tuning - {param}')
#             plt.ylabel('Mean Test Score')
#             plt.xlabel(param.replace('param_', '').replace('__', ' '))
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             plt.savefig(os.path.join(self.plot_dir, f'{param}_performance.png'))
#             plt.close()

#     def build_model(self):
#         self.model = create_model(input_shape=input_shape, n_classes=n_classes)

    
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
        
#         self.model.load_weights('NN_best_model.keras')
#         print()  # For newline after progress bar

#     def evaluate_model(self):
#         total_steps = 5
#         current_step = 1

#         # Predict probabilities on the test set
#         y_pred_proba = self.model.predict(self.X_test)
#         self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
#         current_step += 1
#         # Compute ROC curve and ROC area for each class
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#         all_fpr = np.unique(np.concatenate([roc_curve(self.y_test[:, i], y_pred_proba[:, i])[0] for i in range(self.n_classes)]))
#         # Interpolate all ROC curves at these points
#         mean_tpr = np.zeros_like(all_fpr)
#         for i in range(self.n_classes):
#             fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred_proba[:, i])
#             roc_auc[i] = auc(fpr[i], tpr[i])
#             mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
#         # Finally average it and compute AUC
#         mean_tpr /= self.n_classes

#         # Save ROC data
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
        
#         # Evaluate the model on the test set
#         test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)  # Pass both features and labels
#         self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
#         current_step += 1
#         print('Test loss:', test_loss)
#         print('Test accuracy:', test_acc)
#         # Predict self.create_model
#         y_pred_classes = np.argmax(y_pred_proba, axis=1)
#         y_test_classes = np.argmax(self.y_test, axis=1)
        
#         # Generate confusion matrix
#         cm = confusion_matrix(y_test_classes, y_pred_classes)
#         plt.figure(figsize=(10, 7))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.xlabel('Predicted Labels')
#         plt.ylabel('True Labels')
#         plt.title('Confusion Matrix')
#         plt.savefig(os.path.join(self.plot_dir, 'confusion_plot.png'))
#         plt.close()
#         #plt.show()
#         self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
#         current_step += 1
#         # Save metrics
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

#     def plot_training_history(self):
#         plt.figure()
#         plt.plot(self.history.history['accuracy'], label='accuracy')
#         plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.ylim([0, 1])
#         plt.title('Training History')
#         plt.legend(loc='lower right')
#         plt.savefig(os.path.join(self.plot_dir, 'training_history.png'))
#         #plt.show()
#         plt.close()



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
# from sklearn.model_selection import GridSearchCV, ParameterGrid
# import time


# import warnings
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all non-error messages
# # Suppress specific warnings
# warnings.filterwarnings("ignore", message="Do not pass an `input_shape`/`input_dim` argument to a layer")
# # Suppress all warnings (if needed, use with caution)
# warnings.filterwarnings("ignore")


# def create_model(activation='relu', optimizer='adam', dropout_rate=0.2, input_shape=5, n_classes=5):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.Input(shape=(input_shape,)))
#     model.add(tf.keras.layers.Dense(128, activation=activation))
#     model.add(tf.keras.layers.Dropout(dropout_rate))
#     model.add(tf.keras.layers.Dense(64, activation=activation))
#     model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#     return model


# class NeuralNetwork:
#     def __init__(self, plot_dir='./plot/NN'):
#         self.plot_dir = plot_dir
#         os.makedirs(self.plot_dir, exist_ok=True)
#         self.model = None
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
#         print("Input shape:", input_shape)
#         print("Number of classes:", self.n_classes)


#     def hyperparameter_tuning(self):
#         param_grid = {
#             'model__activation': ['relu', 'tanh'],
#             'model__optimizer': ['adam', 'rmsprop'],
#             'model__dropout_rate': [0.2, 0.3],
#             'batch_size': [32, 64],
#             'epochs': [50, 100]
#         }
        
#         total_steps = np.prod([len(v) for v in param_grid.values()]) * 3  # Number of combinations times number of CV folds
#         current_step = 1
        
#         model = KerasClassifier(
#             model=create_model,
#             verbose=0,
#             input_shape=input_shape,
#             n_classes=n_classes
#         )
        
#         grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, return_train_score=True)
        
#         start_time = time.time()  # Start timing
        
#         for params in ParameterGrid(param_grid):
#             self.show_progress("Neural Network", "Hyperparameter Tuning", current_step, total_steps)
#             current_step += 1
        
#         grid_result = grid.fit(self.X_train, self.y_train)
#         end_time = time.time()  # End timing
        
#         elapsed_time = end_time - start_time
#         print(f"\nTotal time for hyperparameter tuning: {elapsed_time:.2f} seconds")

#         print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#         self.best_params_ = grid_result.best_params_
#         self.model = grid_result.best_estimator_
        
#         # Save the grid search results
#         results_df = pd.DataFrame(grid_result.cv_results_)
#         results_df.to_csv(os.path.join(self.plot_dir, 'grid_search_results.csv'), index=False)
#         self.plot_hyperparameter_performance(results_df)
    
#     def plot_hyperparameter_performance(self, results_df):
#         # Plot the performance of each parameter combination
#         for param in ['param_model__activation', 'param_model__optimizer', 'param_model__dropout_rate', 'param_batch_size', 'param_epochs']:
#             plt.figure(figsize=(10, 6))
#             sns.boxplot(x=param, y='mean_test_score', data=results_df)
#             plt.title(f'Hyperparameter Tuning - {param}')
#             plt.ylabel('Mean Test Score')
#             plt.xlabel(param.replace('param_', '').replace('__', ' '))
#             plt.xticks(rotation=45)
#             plt.tight_layout()
#             plt.savefig(os.path.join(self.plot_dir, f'{param}_performance.png'))
#             plt.close()
            
#     def build_model(self):
#         self.model = create_model(input_shape=input_shape, n_classes=n_classes)

    
#     # def train_model(self, epochs=100):
#     #     if not self.model:
#     #         self.build_model()
#     #     checkpoint = tf.keras.callbacks.ModelCheckpoint('NN_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
#     #     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#     #     total_steps = epochs
#     #     current_step = 1
#     #     print("Start training Neural Network Model...")
#     #     for epoch in range(epochs):
#     #         self.show_progress("Neural Network", "Training", current_step, total_steps)
#     #         current_step += 1
#     #         self.history = self.model.fit(self.X_train, self.y_train, epochs=1, validation_split=0.2, callbacks=[checkpoint, early_stopping], verbose=0)
        
#     #     self.model.load_weights('NN_best_model.keras')
#     #     print()  # For newline after progress bar

#     def train_model(self, epochs=100):
#             if not self.model:
#                 self.build_model()
#             checkpoint = tf.keras.callbacks.ModelCheckpoint('NN_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
#             early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#             total_steps = epochs
#             current_step = 1
#             print("Start training Neural Network Model...")
#             for epoch in range(epochs):
#                 self.show_progress("Neural Network", "Training", current_step, total_steps)
#                 current_step += 1
#                 self.history = self.model.fit(self.X_train, self.y_train, epochs=1, validation_split=0.2, callbacks=[checkpoint, early_stopping], verbose=0)
            
#             self.model.load_weights('NN_best_model.keras')
#             print()  # For newline after progress bar

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

#     def plot_training_history(self):
#         plt.figure()
#         plt.plot(self.history.history['accuracy'], label='accuracy')
#         plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.ylim([0, 1])
#         plt.title('Training History')
#         plt.legend(loc='lower right')
#         plt.savefig(os.path.join(self.plot_dir, 'training_history.png'))
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
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import json
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold
import time
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress all non-error messages
warnings.filterwarnings("ignore", message="Do not pass an `input_shape`/`input_dim` argument to a layer")
warnings.filterwarnings("ignore")


def create_model(activation='relu', optimizer='adam', dropout_rate=0.2, input_shape=5, n_classes=5):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(input_shape,)))
    model.add(tf.keras.layers.Dense(128, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(64, activation=activation))
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
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
        print("Input shape:", input_shape)
        print("Number of classes:", self.n_classes)

    def hyperparameter_tuning(self):
        param_grid = {
            'model__activation': ['relu', 'tanh'],
            'model__optimizer': ['adam', 'rmsprop'],
            'model__dropout_rate': [0.2, 0.3],
            'batch_size': [32, 64],
            'epochs': [50, 100]
        }
        
        # total_steps = np.prod([len(v) for v in param_grid.values()]) * 3  # Number of combinations times number of CV folds
        # current_step = 1
        # total_steps = np.prod([len(v) for v in param_grid.values()]) * 3  # Number of combinations times number of CV folds
        # current_step = 1
        total_steps = np.prod([len(v) for v in param_grid.values()]) * 3  # Number of combinations times number of CV folds
        current_step = 1

        best_score = -np.inf
        best_params = None
        all_results = []
        
        model = KerasClassifier(
            model=create_model,
            verbose=0,
            input_shape=input_shape,
            n_classes=n_classes
        )
        
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, return_train_score=True)
        
        start_time = time.time()  # Start timing
        
        # for params in ParameterGrid(param_grid):
        #     self.show_progress("Neural Network", "Hyperparameter Tuning", current_step, total_steps)
        #     current_step += 1
        
        # param_list = list(ParameterGrid(param_grid))
        # for params in param_list:
        #     self.show_progress("Neural Network", "Hyperparameter Tuning", current_step, total_steps)
        #     current_step += 1
        #     grid.set_params(estimator__batch_size=params['batch_size'])
        #     grid.set_params(estimator__epochs=params['epochs'])
        #     grid.set_params(estimator__model__activation=params['model__activation'])
        #     grid.set_params(estimator__model__optimizer=params['model__optimizer'])
        #     grid.set_params(estimator__model__dropout_rate=params['model__dropout_rate'])
        #     grid.fit(self.X_train, self.y_train)
            
        # grid_result = grid.fit(self.X_train, self.y_train)
        
        
        # Grid search with KFold cross-validation and progress bar implementation
        
        # This initializes the K-Fold cross-validation with 3 splits. 
        # K-Fold cross-validation is a method used to evaluate the performance of a model by splitting the dataset into k subsets, training the model on k-1 of them, and validating it on the remaining one. 
        # This process is repeated k times, with each subset used exactly once as the validation data.
        kf = KFold(n_splits=3)

        # Creates a list of all possible combinations of hyperparameters sepecified in the param_grid
        param_list = list(ParameterGrid(param_grid))
        
        # This outer loop iterates over each combination of hyperparameters. 
        # The inner loop then performs the k-fold cross-validation for the current set of hyperparameters.
        for params in param_list:
            fold_scores = []
            for train_index, val_index in kf.split(self.X_train):
                #  The training and validation data are extracted based on the current fold indices.
                X_fold_train, X_fold_val = self.X_train[train_index], self.X_train[val_index]
                y_fold_train, y_fold_val = self.y_train[train_index], self.y_train[val_index]

                # The model is created using the current set of hyperparameters and then trained on the training data of the current fold.
                model = create_model(
                    activation=params['model__activation'],
                    optimizer=params['model__optimizer'],
                    dropout_rate=params['model__dropout_rate'],
                    input_shape=input_shape,
                    n_classes=n_classes
                )
                model.fit(X_fold_train, y_fold_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
                
                # After training, the model is evaluated on the validation data of the current fold, and the accuracy score is stored in fold_scores.
                score = model.evaluate(X_fold_val, y_fold_val, verbose=0)[1]  # Evaluate and get accuracy
                fold_scores.append(score)

                # Update progress
                self.show_progress("Neural Network", "Hyperparameter Tuning", current_step, total_steps)
                current_step += 1

            # The mean score across all folds for the current set of hyperparameters is calculated and stored. 
            # If this mean score is better than the best score seen so far, the best score and best parameters are updated.
            mean_score = np.mean(fold_scores)
            all_results.append((params, mean_score))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        end_time = time.time()  # End timing

        elapsed_time = end_time - start_time
        print(f"\nTotal time for hyperparameter tuning: {elapsed_time:.2f} seconds")
        print(f"Best: {best_score} using {best_params}")

        self.best_params_ = best_params
        
        # # # Extract only the parameters for creating the model
        # # model_params = {k.replace('model__', ''): v for k, v in self.best_params_.items() if 'model__' in k}
        # self.model = create_model(**self.best_params_, input_shape=input_shape, n_classes=n_classes)
        
        # Extract only the parameters for creating the model
        model_params = {k.replace('model__', ''): v for k, v in self.best_params_.items() if 'model__' in k}
        self.best_params_ = model_params
        self.model = create_model(**self.best_params_, input_shape=input_shape, n_classes=n_classes)


        # Save the results
        results_df = pd.DataFrame(all_results, columns=['params', 'mean_score'])
        results_df.to_csv(os.path.join(self.plot_dir, 'grid_search_results.csv'), index=False)
        self.plot_hyperparameter_performance(results_df)
        
        # end_time = time.time()  # End timing
        
        # elapsed_time = end_time - start_time
        # print(f"\nTotal time for hyperparameter tuning: {elapsed_time:.2f} seconds")

        # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # self.best_params_ = grid_result.best_params_
        
        #         # Extract only the parameters for creating the model
        # model_params = {k.replace('model__', ''): v for k, v in self.best_params_.items() if 'model__' in k}
        # self.model = create_model(**model_params, input_shape=input_shape, n_classes=n_classes)
        
        # #self.model = create_model(**self.best_params_, input_shape=input_shape, n_classes=n_classes)
        
        # # Save the grid search results
        # results_df = pd.DataFrame(grid_result.cv_results_)
        # results_df.to_csv(os.path.join(self.plot_dir, 'grid_search_results.csv'), index=False)
        # self.plot_hyperparameter_performance(results_df)
    
    # def plot_hyperparameter_performance(self, results_df):
    #     # Plot the performance of each parameter combination
    #     for param in ['param_model__activation', 'param_model__optimizer', 'param_model__dropout_rate', 'param_batch_size', 'param_epochs']:
    #         plt.figure(figsize=(10, 6))
    #         sns.boxplot(x=param, y='mean_test_score', data=results_df)
    #         plt.title(f'Hyperparameter Tuning - {param}')
    #         plt.ylabel('Mean Test Score')
    #         plt.xlabel(param.replace('param_', '').replace('__', ' '))
    #         plt.xticks(rotation=45)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(self.plot_dir, f'{param}_performance.png'))
    #         plt.close()
    
    # def plot_hyperparameter_performance(self, results_df):
    #     results_df['param_model__activation'] = results_df['param_model__activation'].astype(str)
    #     results_df['param_model__optimizer'] = results_df['param_model__optimizer'].astype(str)
    #     results_df['param_model__dropout_rate'] = results_df['param_model__dropout_rate'].astype(float)
    #     results_df['param_batch_size'] = results_df['param_batch_size'].astype(int)
    #     results_df['param_epochs'] = results_df['param_epochs'].astype(int)

    #     plt.figure(figsize=(12, 8))
    #     sns.lineplot(data=results_df, x='param_epochs', y='mean_test_score', hue='param_model__activation', style='param_model__optimizer', markers=True, dashes=False)
    #     plt.title('Hyperparameter Tuning Performance')
    #     plt.xlabel('Epochs')
    #     plt.ylabel('Mean Accuracy')
    #     plt.legend(loc='best')
    #     plt.savefig(os.path.join(self.plot_dir, 'hyperparameter_performance.png'))
    #     plt.close()
            
    # def plot_hyperparameter_performance(self, results_df):
    #     results = pd.DataFrame(results_df['params'].tolist())
    #     results['mean_score'] = results_df['mean_score']

    #     for param in ['activation', 'optimizer', 'dropout_rate', 'batch_size', 'epochs']:
    #         plt.figure(figsize=(10, 6))
    #         sns.boxplot(x=param, y='mean_score', data=results)
    #         plt.title(f'Hyperparameter Tuning - {param}')
    #         plt.ylabel('Mean Test Score')
    #         plt.xlabel(param.replace('_', ' ').capitalize())
    #         plt.xticks(rotation=45)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(self.plot_dir, f'{param}_performance.png'))
    #         plt.close()
    
    def plot_hyperparameter_performance(self, results_df):
        # Convert results_df into a proper DataFrame
        results = pd.DataFrame(results_df['params'].tolist())
        results['mean_score'] = results_df['mean_score']

        # Check for presence of all necessary columns before plotting
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
    
    def train_model(self, epochs=100):
        if not self.model:
            self.build_model()
        checkpoint = tf.keras.callbacks.ModelCheckpoint('NN_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

        total_steps = epochs
        current_step = 1
        print("Start training Neural Network Model...")
        for epoch in range(epochs):
            self.show_progress("Neural Network", "Training", current_step, total_steps)
            current_step += 1
            self.history = self.model.fit(self.X_train, self.y_train, epochs=1, validation_split=0.2, callbacks=[checkpoint, early_stopping], verbose=0)
        
        print()  # For newline after progress bar

    def evaluate_model(self):
        total_steps = 5
        current_step = 1

        y_pred_proba = self.model.predict(self.X_test)
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
        mean_tpr /= self.n_classes

        roc_data = {
            'fpr': {str(i): fpr[i].tolist() for i in fpr},
            'tpr': {str(i): tpr[i].tolist() for i in tpr},
            'roc_auc': {str(i): roc_auc[i] for i in roc_auc},
            'all_fpr': all_fpr.tolist(),
            'mean_tpr': mean_tpr.tolist()
        }
        with open(os.path.join(self.plot_dir, 'roc_data_nn.json'), 'w') as f:
            json.dump(roc_data, f)
        
        self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
        current_step += 1
        
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
        current_step += 1
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        y_test_classes = np.argmax(self.y_test, axis=1)
        
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.plot_dir, 'confusion_plot.png'))
        plt.close()
        
        self.show_progress("Neural Network", "Evaluating", current_step, total_steps)
        current_step += 1

        precision = precision_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
        f1 = f1_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
        metrics = {
            'accuracy': test_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        with open(os.path.join(self.plot_dir, 'metrics_nn.json'), 'w') as f:
            json.dump(metrics, f)

        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

    def plot_training_history(self):
        plt.figure()
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.title('Training History')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.plot_dir, 'training_history.png'))
        plt.close()
