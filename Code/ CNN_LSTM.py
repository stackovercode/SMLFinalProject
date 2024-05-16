import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Input
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from numpy import interp 
from itertools import cycle

# Check available devices, look for GPU
print("Available devices:")
print(tf.config.list_physical_devices())

# Load and preprocess the dataset
data = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data)
df_sample = df.sample(frac=0.01, random_state=42)

# Define directory for saving plots
plot_dir = './TestingPlots/CNN_LSTM'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

ROC_plot = os.path.join(plot_dir, 'ROC_plot.png')
confusion_plot = os.path.join(plot_dir, 'confusion_plot.png')
training_history = os.path.join(plot_dir, 'training_history.png')

# Display information about the DataFrame
df_sample.info()

# Check for missing values
if df_sample.isna().any().any():
    print("Missing values detected. Handling missing values by dropping rows with NaNs.")
    df_sample.dropna(inplace=True)  # Remove rows with any NaN values

# Check for duplicate rows
if df_sample.duplicated().any():
    print("Duplicate rows detected. Removing duplicates.")
    df_sample.drop_duplicates(inplace=True)

# Assuming the first column is the label and the rest are features
X = df_sample.iloc[:, 1:].values
y = df_sample.iloc[:, 0].values

# Binarize the labels for multi-class classification
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN input (samples, time steps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=0.3, random_state=42)

# Define the CNN + LSTM model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, activation='relu'),
    Dropout(0.2),
    Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model with more epochs and save the best model
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[checkpoint, early_stopping])

# Load the best model
model.load_weights('best_model.keras')

# Predict probabilities on the test set
y_pred_proba = model.predict(X_test)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
all_fpr = np.unique(np.concatenate([roc_curve(y_test[:, i], y_pred_proba[:, i])[0] for i in range(n_classes)]))

# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

# Plot the ROC curve for each class and the mean ROC curve
plt.figure(figsize=(10, 8))  # Increase figure size for better readability
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive'])  # Add more colors for more classes
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Add a diagonal line representing random guessing

plt.plot(all_fpr, mean_tpr, color='navy', linestyle='-', lw=2,
         label='Mean ROC (area = %0.2f)' % auc(all_fpr, mean_tpr))

plt.xlim([-0.05, 1.05])  # Adjust the limits for better visual clarity
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)  # Increase label font size
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve for Multi-Class Classification', fontsize=14)  # Add a comprehensive title
plt.legend(loc="lower right", prop={'size': 10})  # Adjust legend font size

# Add text annotations to show class labels
for i in range(n_classes):
    plt.text(fpr[i][-2], tpr[i][-2], f"Class {i}", fontsize=10)  # No need to use colors here

plt.grid(alpha=0.4)  # Add a subtle grid

plt.savefig(ROC_plot)  # Save the figure
plt.show()  # Display the figure

# Predict classes
y_pred_classes = np.argmax(y_pred_proba, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(confusion_plot)
plt.show()

# Additional evaluation metrics
precision = precision_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
recall = recall_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
f1 = f1_score(y_test_classes, y_pred_classes, average='macro', zero_division=0)
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Plot training history
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Training History')
plt.savefig(training_history)
plt.legend(loc='lower right')
plt.show()
