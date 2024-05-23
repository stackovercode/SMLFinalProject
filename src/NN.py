# import tensorflow as tf
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os
# from itertools import cycle
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
# from sklearn.model_selection import train_test_split

# class NeuralNetwork:
#     def __init__(self, data_path, processed_data_path, plot_dir='./plot/NN'):
#         self.data_path = data_path
#         self.processed_data_path = processed_data_path
#         self.plot_dir = plot_dir
#         os.makedirs(self.plot_dir, exist_ok=True)
#         self.model = None

#         print("Available devices:")
#         print(tf.config.list_physical_devices())

#     # def load_and_preprocess_data(self):
#     #     # Load the processed data
#     #     df = pd.read_csv(self.processed_data_path)

#     #     # Assuming the first column is the label and the rest are features
#     #     X = df.iloc[:, 1:].values
#     #     y = df.iloc[:, 0].values

#     #     # Binarize the labels for multi-class classification
#     #     y_bin = label_binarize(y, classes=np.unique(y))
#     #     self.n_classes = y_bin.shape[1]

#     #     # Split the data into training and testing sets
#     #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)
    
#     def load_data(self, X_train, X_test, y_train, y_test):
#         # Binarize the labels for multi-class classification
#         y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
#         y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
#         self.n_classes = y_train_bin.shape[1]

#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train_bin
#         self.y_test = y_test_bin

#     def build_model(self):
#         self.model = tf.keras.Sequential([
#             tf.keras.layers.Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(self.n_classes, activation='softmax')
#         ])
#         self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         self.model.summary()

#     def train_model(self, epochs=50):
#         checkpoint = tf.keras.callbacks.ModelCheckpoint('NN_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
#         early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
#         self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, validation_split=0.2, callbacks=[checkpoint, early_stopping])

#         # Load the best model
#         self.model.load_weights('NN_best_model.keras')


#     def evaluate_model(self):
#         # Predict probabilities on the test set
#         y_pred_proba = self.model.predict(self.X_test)

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

#         # Plot the ROC curve for each class and the mean ROC curve
#         plt.figure(figsize=(10, 8))
#         colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
#         for i, color in zip(range(self.n_classes), colors):
#             plt.plot(fpr[i], tpr[i], color=color, lw=2,
#                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

#         plt.plot([0, 1], [0, 1], 'k--', lw=2)
#         plt.plot(all_fpr, mean_tpr, color='navy', linestyle='-', lw=2, label='Mean ROC (area = %0.2f)' % auc(all_fpr, mean_tpr))

#         plt.xlim([-0.05, 1.05])
#         plt.ylim([-0.05, 1.05])
#         plt.xlabel('False Positive Rate', fontsize=12)
#         plt.ylabel('True Positive Rate', fontsize=12)
#         plt.title('ROC Curve for Multi-Class Classification', fontsize=14)
#         plt.legend(loc="lower right", prop={'size': 10})

#         for i in range(self.n_classes):
#             plt.text(fpr[i][-2], tpr[i][-2], f"Class {i}", fontsize=10)

#         plt.grid(alpha=0.4)
#         plt.savefig(os.path.join(self.plot_dir, 'ROC_plot.png'))
#         plt.show()

#         # Evaluate the model on the test set
#         test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)  # Pass both features and labels
#         print('Test accuracy:', test_acc)

#         # Predict classes
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
#         plt.show()

#         # Additional evaluation metrics
#         precision = precision_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
#         recall = recall_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
#         f1 = f1_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
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
#         plt.show()



import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

class NeuralNetwork:
    def __init__(self, plot_dir='./plot/NN'):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)
        self.model = None
        print("Available devices:")
        print(tf.config.list_physical_devices())

    def load_data(self, X_train, X_test, y_train, y_test):
        # Binarize the labels for multi-class classification
        y_train_bin = label_binarize(y_train, classes=np.unique(y_train))
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        self.n_classes = y_train_bin.shape[1]
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train_bin
        self.y_test = y_test_bin

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def train_model(self, epochs=50):
        checkpoint = tf.keras.callbacks.ModelCheckpoint('NN_best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, validation_split=0.2, callbacks=[checkpoint, early_stopping])
        # Load the best model
        self.model.load_weights('NN_best_model.keras')

    def evaluate_model(self):
        # Predict probabilities on the test set
        y_pred_proba = self.model.predict(self.X_test)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        all_fpr = np.unique(np.concatenate([roc_curve(self.y_test[:, i], y_pred_proba[:, i])[0] for i in range(self.n_classes)]))
        # Interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y_test[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= self.n_classes
        # Plot the ROC curve for each class and the mean ROC curve
        plt.figure(figsize=(10, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.plot(all_fpr, mean_tpr, color='navy', linestyle='-', lw=2, label='Mean ROC (area = %0.2f)' % auc(all_fpr, mean_tpr))
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve for Multi-Class Classification', fontsize=14)
        plt.legend(loc="lower right", prop={'size': 10})
        for i in range(self.n_classes):
            plt.text(fpr[i][-2], tpr[i][-2], f"Class {i}", fontsize=10)
        plt.grid(alpha=0.4)
        plt.savefig(os.path.join(self.plot_dir, 'ROC_plot.png'))
        #plt.show()
        # Evaluate the model on the test set
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)  # Pass both features and labels
        print('Test accuracy:', test_acc)
        # Predict classes
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        y_test_classes = np.argmax(self.y_test, axis=1)
        # Generate confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.plot_dir, 'confusion_plot.png'))
        #plt.show()
        # Additional evaluation metrics
        precision = precision_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
        recall = recall_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
        f1 = f1_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
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
        #plt.show()
