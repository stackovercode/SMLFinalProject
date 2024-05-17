import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os

# Load dataset
data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data_path, header=None, dtype={i: 'str' for i in range(11)})

# Assign column names
col_names = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10', 'class']
df.columns = col_names

# Convert numerical columns to float and handle missing data
df[col_names[:-1]] = df[col_names[:-1]].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# Convert class to categorical integer labels
df['class'] = pd.Categorical(df['class']).codes

# Stratified sampling of the data (5%)
df_sample = df.groupby('class', group_keys=True).apply(lambda x: x.sample(frac=0.05, random_state=42)).reset_index(drop=True)

# Split dataset into features and target variable
X = df_sample.drop('class', axis=1)
y = df_sample['class']

# Define directory for saving plots
plot_dir = './TestingPlots/svm'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

ROC_plot = os.path.join(plot_dir, 'ROC_plot.png')
confusion_plot = os.path.join(plot_dir, 'confusion_plot.png')


# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
    # ... (the rest of your code for model training, evaluation, and plotting)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize the data for SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)  
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
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # Plot ROC curve for each class in test set only
    plt.figure(figsize=(10, 7))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
    for i, color in zip(range(n_classes), colors):
        if np.sum(y_test_binarized[:, i]) > 0:  # Only plot if class is present
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig(ROC_plot)
    plt.show()
