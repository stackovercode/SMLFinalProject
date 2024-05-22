import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import itertools
from tqdm import tqdm
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

def load_and_preprocess_data(data_path, sample_fracs, random_state=42):
    """Loads, cleans, and samples the dataset."""
    df = pd.read_csv(data_path, header=None, dtype=str)
    col_names = ['source'] + [f'num{i}' for i in range(1, 11)]
    df.columns = col_names
    df = df.iloc[1:].copy()
    df[df.columns[1:]] = df[df.columns[1:]].astype(float)
    df['source'] = df['source'].astype('category')

    # Convert labels to integers
    df['source'] = df['source'].astype(int)

    # Print the number of samples for each robot before sampling
    print("Number of samples per robot before sampling:")
    print(df['source'].value_counts())

    # Shuffle and sample
    df = df.sample(frac=1, random_state=random_state)
    sampled_dfs = []
    for robot_id, frac in sample_fracs.items():
        robot_samples = df[df['source'] == robot_id]
        if frac > 0:
            sampled_df = robot_samples.sample(frac=frac, random_state=random_state)
            print(f"Samples for robot {robot_id} after sampling: {len(sampled_df)}")
            sampled_dfs.append(sampled_df)

    df_sample = pd.concat(sampled_dfs, ignore_index=True)

    # Check if the sample is empty
    if df_sample.empty:
        raise ValueError("The sampled dataset is empty. Please check the sample fractions.")

    return df_sample.drop('source', axis=1), df_sample['source']

def evaluate_robot_performance(robot_id, y_test, y_pred_gb):
    mask = (y_test == robot_id)
    if sum(mask) > 0:
        y_test_robot = y_test[mask]  
        y_pred_robot = y_pred_gb[mask] 
        labels = sorted(set(y_test_robot) | set(y_pred_robot))
        return (
            robot_id,
            confusion_matrix(y_test_robot, y_pred_robot, labels=[0, 1, 2, 3, 4]),  # Ensure all labels are included
            classification_report(y_test_robot, y_pred_robot, labels=labels, zero_division=0),
        )
    else:
        return robot_id, None, None

def evaluate_robot(args):
    robot_id, y_test, y_pred_gb = args
    return evaluate_robot_performance(robot_id, y_test, y_pred_gb)

def plot_confusion_matrix(cm, robot_id, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

if __name__ == '__main__':  # Main guard to prevent multiprocessing errors on Windows/macOS
    data_path = './4_Classification of Robots from their conversation sequence_Set2.csv'

    sample_fracs = {
    '0': 0.1,  # Adjust sample fractions for debugging
    '1': 0.1,  
    '2': 0.1,  
    '3': 0.1,  
    '4': 0.1   
    }

    X, y = load_and_preprocess_data(data_path, sample_fracs)

    # Debugging: Check if the data is correctly loaded and not empty
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Unique labels in y:", y.unique())

    if X.empty or y.empty:
        raise ValueError("The dataset is empty after preprocessing. Please check the sample fractions.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gb_accuracy = []
    robot_results = defaultdict(list)

    # Fixed parameters for the Gradient Boosting model
    fixed_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3
    }

    for fold, (train_index, test_index) in enumerate(tqdm(skf.split(X, y), desc="Cross-Validation")): 
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Commenting out grid search
        # grid_search = HalvingGridSearchCV(GradientBoostingClassifier(), param_grid, cv=3, n_jobs=-1)
        # grid_search.fit(X_train_scaled, y_train)
        # best_params[fold] = grid_search.best_params_
        # print(f"Best parameters for fold {fold}: {grid_search.best_params_}")

        # Train the final Gradient Boosting model with fixed parameters
        gb = GradientBoostingClassifier(**fixed_params)
        gb.fit(X_train_scaled, y_train)

        # Predict and evaluate
        y_pred_gb = gb.predict(X_test_scaled)
        gb_accuracy.append(accuracy_score(y_test, y_pred_gb))

        # Evaluate per robot for this fold
        for robot_id in sorted(y_test.unique()):
            mask = (y_test == robot_id)
            if mask.any():
                y_test_robot = y_test[mask]
                y_pred_robot = y_pred_gb[mask]
                robot_results[robot_id].append((y_test_robot, y_pred_robot))

    # Define directory for saving plots
    plot_dir = './TestingPlots/GB_V3'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Overall Gradient Boosting evaluation across folds
    mean_gb_accuracy = np.mean(gb_accuracy)
    print('Mean Gradient Boosting Accuracy:', mean_gb_accuracy)

    # Plot and save accuracy across folds
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(gb_accuracy) + 1), gb_accuracy, marker='o')
    plt.title('Gradient Boosting Accuracy Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, len(gb_accuracy) + 1))
    plt.grid(True)
    accuracy_plot_path = os.path.join(plot_dir, 'gb_accuracy_across_folds.png')
    plt.savefig(accuracy_plot_path)
    plt.close()

    # Detailed evaluation per robot (aggregate across folds)
    results = []
    with tqdm(total=len(robot_results), desc="Evaluating Robots") as pbar:  
        for result in process_map(
                evaluate_robot,
                [
                    (robot_id, np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results]))
                    for robot_id, results in robot_results.items()
                ],
                chunksize=1,
        ):
            results.append(result)
            pbar.update()

    # Print the results and plot confusion matrices
    for robot_id, cm, report in results:
        if cm is not None:  # Only print if there are results
            print(f"Performance for robot {robot_id}:")
            print(cm)
            print(report)

            # Save the classification report
            report_path = os.path.join(plot_dir, f'classification_report_robot_{robot_id}.txt')
            with open(report_path, 'w') as f:
                f.write(report)

            # Plot and save confusion matrix for each robot
            plt.figure(figsize=(8, 6))
            plot_confusion_matrix(cm, robot_id, classes=['0', '1', '2', '3', '4'], 
                                  title=f'Confusion Matrix for Robot {robot_id}')
            confusion_plot_path = os.path.join(plot_dir, f'confusion_matrix_robot_{robot_id}.png')
            plt.savefig(confusion_plot_path)
            plt
