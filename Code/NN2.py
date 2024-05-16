import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, roc_curve, auc

from scipy import interp
from itertools import cycle

# Check available devices, look for GPU
print("Available devices:")
print(tf.config.list_physical_devices())


# Load and prepare the dataset
data = '4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data)
df_sample = df.sample(frac=0.1, random_state=42)
X = df_sample.iloc[:, 1:].values
y = df_sample.iloc[:, 0].values
y_bin = label_binarize(y, classes=np.unique(y))
n_classes = y_bin.shape[1]


# Randomly sample 10% of the dataset
df_sample = df.sample(frac=0.1, random_state=42)

# Separate features and target
X = df_sample.iloc[:, 1:].values  # Assuming the first column is the label
y = df_sample.iloc[:, 0].values

# Normalize the features and split the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=0.3, random_state=42)


# # Split into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define and train the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Predict probabilities
y_pred_proba = model.predict(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
all_fpr = np.unique(np.concatenate([roc_curve(y_test[:, i], y_pred_proba[:, i])[0] for i in range(n_classes)]))


# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

# Plot the ROC curve
plt.figure()
plt.plot(all_fpr, mean_tpr, color='blue', label='Mean ROC (area = %0.2f)' % auc(all_fpr, mean_tpr))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# Predict classes
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()  