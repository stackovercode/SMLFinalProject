import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import learning_curve
import os
from tqdm import tqdm
import sys
from time import time

# Function to show progress
def show_progress(label, full, prog):
    sys.stdout.write("\r{0}: {1}%  [{2}{3}]".format(label, prog, "█"*full, " "*(30-full)))
    sys.stdout.flush()

def perform_grid_search(X_train, y_train):
    # Create a Random Forest Classifier
    rfc = RandomForestClassifier(random_state=0, n_jobs=-1)

    # Define the parameter grid
    # param_grid = {
    #     'n_estimators': [50, 100, 150],
    #     'max_depth': [10, 15, 20],
    #     'min_samples_split': [10, 15, 20],
    #     'min_samples_leaf': [5, 10, 15]
    # }
# Define the parameter grid with reduced complexity
    param_grid = {
        'n_estimators': [25, 50, 75, 200],  # More options for number of trees
        'max_depth': [5, 10, 15, 20],         # More options for maximum depth
        'min_samples_split': [10, 15, 20, 25],  # More options for minimum samples to split
        'min_samples_leaf': [2, 5, 10, 15],     # More options for minimum samples per leaf
        'max_features': ['auto', 'sqrt', 'log2'],  # Different options for number of features to consider at each split
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=3)

    # Fit the grid search to the data with progress display
    n_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])
    start_time = time()

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.current = 0

        def update(self, increment=1):
            self.current += increment
            full = int(30 * self.current / self.total)
            prog = int(100 * self.current / self.total)
            show_progress("Grid Search Progress", full, prog)

    progress_bar = ProgressBar(n_combinations * 5)  # 5-fold cross-validation

    # Custom callback to update the progress bar
    def on_fit_start(estimator, *args, **kwargs):
        progress_bar.update()

    # Hook the progress bar into the grid search process
    grid_search.fit(X_train, y_train)

    elapsed_time = time() - start_time
    print(f"\nGrid Search completed in {elapsed_time:.2f} seconds")

    # Print the best parameters and best score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best accuracy score: ", grid_search.best_score_)

    return grid_search.best_estimator_

# Load dataset with dtype specification and error handling
data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data_path, header=0)

# Define directory for saving plots
plot_dir = './TestingPlots/random_forest'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

ROC_plot = os.path.join(plot_dir, 'ROC_plot.png')
confusion_plot = os.path.join(plot_dir, 'confusion_plot.png')
training = os.path.join(plot_dir, 'training.png')
training_history_plot = os.path.join(plot_dir, 'training_history_plot.png')
feature_plot = os.path.join(plot_dir, 'feature_plot.png')
cumulative_gains_plot = os.path.join(plot_dir, 'cumulative_gains_plot.png')
lift_chart_plot = os.path.join(plot_dir, 'lift_chart_plot.png')

# Verify the first few rows and data types
print(df.head())
print(df.info())

# Assign column names (if not already assigned)
col_names = ['source', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10']
df.columns = col_names

# Drop the first row if it contains column headers or irrelevant data
if df.iloc[0, 0] == 'source':
    df = df.drop(0)

# Convert numerical columns to float and handle missing data
for col in col_names[1:]:  # Exclude 'source' which is categorical
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert 'source' to categorical integer labels
df['source'] = pd.Categorical(df['source']).codes

# Handle missing data if any
df.dropna(inplace=True)

# Verify the cleaned data
print(df.head())
print(df.info())

# Sample the data to reduce size for quicker processing
df_sample = df.sample(frac=0.5, random_state=42)

# Split dataset into features and target variable
X = df_sample.drop('source', axis=1)
y = df_sample['source']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Uncomment the next two lines to perform grid search and find the best estimator
# best_rfc = perform_grid_search(X_train, y_train)
# best_rfc.fit(X_train, y_train)

# Use the best hyperparameters from the previous grid search
best_params = {'max_depth': 20, 'min_samples_leaf': 5, 'min_samples_split': 20, 'n_estimators': 150}
best_rfc = RandomForestClassifier(**best_params, random_state=0, n_jobs=-1)

# Train the model with the best parameters and show progress
print("Training the model...")
with tqdm(total=best_rfc.n_estimators) as pbar:
    for i in range(best_rfc.n_estimators):
        best_rfc.set_params(n_estimators=i+1)
        best_rfc.fit(X_train, y_train)
        pbar.update(1)

# Predict on the test set
y_pred = best_rfc.predict(X_test)
y_pred_proba = best_rfc.predict_proba(X_test)

# Model Evaluation
print('Model accuracy score with best hyperparameters: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC Curve
n_classes = y_pred_proba.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()
all_fpr = np.unique(np.concatenate([roc_curve(y_test == i, y_pred_proba[:, i])[0] for i in range(n_classes)]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= n_classes

plt.figure(figsize=(10, 8))
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
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

for i in range(n_classes):
    plt.text(fpr[i][-2], tpr[i][-2], f"Class {i}", fontsize=10)

plt.grid(alpha=0.4)
plt.savefig(ROC_plot)
#plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig(confusion_plot)
#plt.show()

# Additional evaluation metrics
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Training History (Random Forest doesn't have epochs, this is a placeholder if needed)
history = {'accuracy': [], 'val_accuracy': []}
for i in range(1, 51):
    best_rfc.set_params(n_estimators=i * 10)
    best_rfc.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, best_rfc.predict(X_train))
    val_accuracy = accuracy_score(y_test, best_rfc.predict(X_test))
    history['accuracy'].append(train_accuracy)
    history['val_accuracy'].append(val_accuracy)

plt.figure()
plt.plot(range(1, 51), history['accuracy'], label='accuracy')
plt.plot(range(1, 51), history['val_accuracy'], label='val_accuracy')
plt.xlabel('Number of Trees (in tens)')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Training History')
plt.savefig(training)
plt.legend(loc='lower right')
#plt.show()


# Feature Importance
feature_scores = pd.Series(best_rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.savefig(feature_plot)
#plt.show()

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X, y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 5))
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
#plt.show()



# Precision-Recall Curve
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test == i, y_pred_proba[:, i])
    avg_precision = average_precision_score(y_test == i, y_pred_proba[:, i])
    plt.plot(recall, precision, lw=2, label='Precision-Recall curve of class {0} (area = {1:0.2f})'.format(i, avg_precision))

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Multi-Class Classification')
plt.legend(loc="lower left")
plt.grid(alpha=0.4)
precision_recall_plot = os.path.join(plot_dir, 'precision_recall_plot.png')
plt.savefig(precision_recall_plot)
#plt.show()

# Cumulative Gains and Lift Charts
def plot_cumulative_gains(y_true, y_probas, title='Cumulative Gains Chart'):
    from sklearn.preprocessing import label_binarize
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
    #plt.show()

def plot_lift_chart(y_true, y_probas, title='Lift Chart'):
    from sklearn.preprocessing import label_binarize
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
    #plt.show()
    
# Call the functions to plot the charts
plot_cumulative_gains(y_test, y_pred_proba)
plot_lift_chart(y_test, y_pred_proba)

# Calibration Curve
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    prob_true, prob_pred = calibration_curve(y_test == i, y_pred_proba[:, i], n_bins=10)
    plt.plot(prob_pred, prob_true, lw=2, label='Calibration curve of class {0}'.format(i))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve for Multi-Class Classification')
plt.legend(loc="lower right")
plt.grid(alpha=0.4)
calibration_curve_plot = os.path.join(plot_dir, 'calibration_curve_plot.png')
plt.savefig(calibration_curve_plot)
#plt.show()
