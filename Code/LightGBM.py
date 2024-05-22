import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import os
import gc

# Load dataset
data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data_path, header=0, dtype={i: 'str' for i in range(11)})

# Assign column names
col_names = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10', 'class']
df.columns = col_names

# Convert numerical columns to float and handle missing data
df[col_names[:-1]] = df[col_names[:-1]].apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

# Convert class to categorical integer labels
df['class'] = pd.Categorical(df['class']).codes

# Sample the data to reduce size for quicker processing
df_sample = df.sample(frac=0.01, random_state=42)  # Adjust sample size as needed

# Split dataset into features and target variable and reset indexes
X = df_sample.drop('class', axis=1).reset_index(drop=True)
y = df_sample['class'].reset_index(drop=True)

# Define directory for saving plots
plot_dir = './TestingPlots/lightgbm'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

ROC_plot = os.path.join(plot_dir, 'ROC_plot.png')
confusion_plot = os.path.join(plot_dir, 'confusion_plot.png')

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_fpr = []
all_tpr = []
all_roc_auc = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert the data to LightGBM dataset format
    d_train = lgb.Dataset(X_train_scaled, label=y_train)
    d_test = lgb.Dataset(X_test_scaled, label=y_test, reference=d_train)

    # Define model parameters
    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'device': 'gpu',
    }

    # Train the model (using callbacks for early stopping)
    model = lgb.train(
        params, d_train, num_boost_round=100, valid_sets=[d_test],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]  # Early stopping callback
    )

    # Evaluate the model
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print('LightGBM Accuracy:', accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(confusion_plot)
    # plt.show()

    # Binarize the output labels for ROC curve
    y_test_binarized = label_binarize(y_test, classes=np.arange(len(np.unique(y))))
    n_classes = y_test_binarized.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Safely calculate ROC curves
    for i in range(n_classes):
        if np.sum(y_test_binarized[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i] = np.array([0, 1])
            tpr[i] = np.array([0, 1])
            roc_auc[i] = 0.5
            print(f"Warning: No positive samples in class {i}, setting AUC to 0.5")

    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)

    # Clean up memory
    del X_train, X_test, y_train, y_test, d_train, d_test, model
    gc.collect()

# Plot ROC curve for each class
plt.figure(figsize=(10, 7))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
for i in range(n_classes):
    for fpr, tpr, roc_auc in zip(all_fpr, all_tpr, all_roc_auc):
        plt.plot(fpr[i], tpr[i], lw=2, alpha=0.3)  # plot each fold
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr[i], tpr[i]) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
    mean_roc_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color=colors[i], lw=2, label=f'ROC curve of class {i} (area = {mean_roc_auc:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(ROC_plot)
# plt.show()
