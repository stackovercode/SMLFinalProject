import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.calibration import calibration_curve
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
import os
from sklearn.decomposition import PCA
from tqdm import tqdm  # Add this import for the progress bar

# Load dataset with dtype specification and error handling
data = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data, header=0, on_bad_lines='skip')

# Define directory for saving plots
plot_dir = './svm2'
os.makedirs(plot_dir, exist_ok=True)

ROC_plot = os.path.join(plot_dir, 'ROC_plot.png')
confusion_plot = os.path.join(plot_dir, 'confusion_plot.png')
training_plot = os.path.join(plot_dir, 'training.png')
feature_plot = os.path.join(plot_dir, 'feature_plot.png')
cumulative_gains_plot = os.path.join(plot_dir, 'cumulative_gains_plot.png')
lift_chart_plot = os.path.join(plot_dir, 'lift_chart_plot.png')
calibration_curve_plot = os.path.join(plot_dir, 'calibration_curve_plot.png')
precision_recall_plot = os.path.join(plot_dir, 'precision_recall_plot.png')
training_history_plot = os.path.join(plot_dir, 'training_history_plot.png')

# Assign column names
col_names = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10', 'class']
df.columns = col_names

# Convert numerical columns to float
for col in col_names[:-1]:  # Exclude 'class' which is the target
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert class to categorical integer labels
df['class'] = pd.Categorical(df['class']).codes  # Convert to categorical codes if they are not numeric

# Handle missing data if any
df.dropna(inplace=True)

# Sample the data to reduce size for quicker processing
df_sample = df.sample(frac=0.05, random_state=42)

# Remove classes with less than 2 samples
class_counts = df_sample['class'].value_counts()
df_sample = df_sample[df_sample['class'].isin(class_counts[class_counts >= 2].index)]

# Split dataset into features and target variable
X = df_sample.drop('class', axis=1).values
y = df_sample['class'].values

# Apply SMOTE to handle imbalanced classes
#smote = SMOTE(random_state=42, k_neighbors=1)
#X_res, y_res = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

# Standardize the data for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optionally apply PCA
use_pca = False  # Set to False to disable PCA
if use_pca:
    pca = PCA(n_components=4)  # Adjust the number of components as needed
    X_train_scaled = pca.fit_transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)

# Train SVM with progress bar
svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)  # Use probability=True for ROC curve
print("Training SVM...")
for i in tqdm(range(1, 2)):  # Modify this loop if you need to iterate multiple times
    svm.fit(X_train_scaled, y_train)

# Evaluate SVM
y_pred_svm = svm.predict(X_test_scaled)
print('SVM Accuracy:', accuracy_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# Plot Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(confusion_plot)
plt.show()

# Binarize the output labels for ROC curve
y_test_binarized = label_binarize(y_test, classes=np.arange(len(np.unique(y))))
n_classes = y_test_binarized.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
y_score = svm.predict_proba(X_test_scaled)

# Safely calculate ROC curves
for i in tqdm(range(n_classes)):
    if np.sum(y_test_binarized[:, i]) > 0:
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        fpr[i] = np.array([0, 1])
        tpr[i] = np.array([0, 1])
        roc_auc[i] = 0.5
        print(f"Warning: No positive samples in class {i}, setting AUC to 0.5")

# Plot ROC curve for each class
plt.figure(figsize=(10, 7))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(ROC_plot)
plt.show()

# Learning Curve
plt.figure(figsize=(10, 8))
train_sizes, train_scores, test_scores = learning_curve(SVC(kernel='rbf', C=1, gamma='scale', probability=True), X, y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 5))
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.1)
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.savefig(training_history_plot)
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score[:, i])
    avg_precision = average_precision_score(y_test_binarized[:, i], y_score[:, i])
    plt.plot(recall, precision, lw=2, label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, avg_precision))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Multi-Class Classification')
plt.legend(loc="lower left")
plt.grid(alpha=0.4)
plt.savefig(precision_recall_plot)
plt.show()

# Cumulative Gains and Lift Charts
def plot_cumulative_gains(y_true, y_probas, title='Cumulative Gains Chart'):
    n_classes = y_probas.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        sorted_proba = np.sort(y_probas[:, i])[::-1]
        cumulative_gains = np.cumsum(sorted_proba) / np.sum(sorted_proba)
        random_gains = np.linspace(0, 1, len(cumulative_gains))

        ax.plot(cumulative_gains, label=f'Class {i}')
        ax.plot(random_gains, linestyle='--', label=f'Random Class {i}')

    ax.set_title(title)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Cumulative Gains')
    ax.legend()
    plt.savefig(cumulative_gains_plot)
    plt.show()

def plot_lift_chart(y_true, y_probas, title='Lift Chart'):
    n_classes = y_probas.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(n_classes):
        sorted_proba = np.sort(y_probas[:, i])[::-1]
        cumulative_gains = np.cumsum(sorted_proba) / np.sum(sorted_proba)
        random_gains = np.linspace(0, 1, len(cumulative_gains))
        random_gains[random_gains == 0] = 1e-6  # Prevent division by zero
        lift = cumulative_gains / random_gains

        ax.plot(lift, label=f'Class {i}')

    ax.set_title(title)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Lift')
    ax.legend()
    plt.savefig(lift_chart_plot)
    plt.show()

# Call the functions to plot the charts
plot_cumulative_gains(y_test, y_score)
plot_lift_chart(y_test, y_score)

# Calibration Curve
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    prob_true, prob_pred = calibration_curve(y_test_binarized[:, i], y_score[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, lw=2, label='Calibration curve of class {0}'.format(i))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve for Multi-Class Classification')
plt.legend(loc="lower right")
plt.grid(alpha=0.4)
plt.savefig(calibration_curve_plot)
plt.show()

# Feature Importance for SVM using PCA components (as SVM does not provide feature importances directly)

# Select features for PCA
pca_features = ['num9', 'num10']

# Apply PCA on the selected features
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Train SVM with PCA components
svm_pca = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
svm_pca.fit(X_train_pca, y_train)

# Get feature importances from PCA components
feature_scores = pd.Series(pca.explained_variance_ratio_, index=['PC1', 'PC2']).sort_values(ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.savefig(feature_plot)
plt.show()
