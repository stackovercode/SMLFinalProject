import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
import sys

class ModelComparison:
    def __init__(self, nn_dir='./plot/NN', gb_dir='./plot/GB', output_dir='./plot/comparison'):
        self.nn_dir = nn_dir
        self.gb_dir = gb_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.total_steps = 4  # Number of steps in the comparison process

    @staticmethod
    def show_progress(label, phase, current, total):
        prog = int((current / total) * 100)
        full = prog // 3
        sys.stdout.write(f"\r{label} ({phase}): {prog}%  [{'█' * full}{' ' * (30 - full)}]")
        sys.stdout.flush()

    def load_metrics(self):
        with open(os.path.join(self.nn_dir, 'metrics_nn.json'), 'r') as f:
            self.nn_metrics = json.load(f)
        with open(os.path.join(self.gb_dir, 'metrics_gb.json'), 'r') as f:
            self.gb_metrics = json.load(f)

    def load_roc_data(self):
        with open(os.path.join(self.nn_dir, 'roc_data_nn.json'), 'r') as f:
            self.nn_roc_data = json.load(f)
        with open(os.path.join(self.gb_dir, 'roc_data_gb.json'), 'r') as f:
            self.gb_roc_data = json.load(f)

    def compare_roc_curves(self):
        plt.figure(figsize=(10, 8))
        
        # Plot NN ROC curve
        for i in range(len(self.nn_roc_data['fpr'])):
            plt.plot(self.nn_roc_data['fpr'][str(i)], self.nn_roc_data['tpr'][str(i)], lw=2,
                     label=f'NN ROC curve of class {i} (area = {self.nn_roc_data["roc_auc"][str(i)]:0.2f})')
        
        # Plot GB ROC curve
        for i in range(len(self.gb_roc_data['fpr'])):
            plt.plot(self.gb_roc_data['fpr'][str(i)], self.gb_roc_data['tpr'][str(i)], lw=2, linestyle='--',
                     label=f'GB ROC curve of class {i} (area = {self.gb_roc_data["roc_auc"][str(i)]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve Comparison', fontsize=14)
        plt.legend(loc="lower right", prop={'size': 10})
        plt.grid(alpha=0.4)
        plt.savefig(os.path.join(self.output_dir, 'ROC_curve_comparison.png'))
        plt.show()

    def compare_metrics(self):
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Neural Network': [
                self.nn_metrics['accuracy'],
                self.nn_metrics['precision'],
                self.nn_metrics['recall'],
                self.nn_metrics['f1_score']
            ],
            'Gradient Boosting': [
                self.gb_metrics['accuracy'],
                self.gb_metrics['precision'],
                self.gb_metrics['recall'],
                self.gb_metrics['f1_score']
            ]
        })
        print(metrics_df)
        metrics_df.to_csv(os.path.join(self.output_dir, 'metrics_comparison.csv'), index=False)

    def run_comparison(self):
        current_step = 1
        
        self.show_progress("Model Comparison", "Loading Metrics", current_step, self.total_steps)
        self.load_metrics()
        current_step += 1
        
        self.show_progress("Model Comparison", "Loading ROC Data", current_step, self.total_steps)
        self.load_roc_data()
        current_step += 1
        
        self.show_progress("Model Comparison", "Comparing ROC Curves", current_step, self.total_steps)
        self.compare_roc_curves()
        current_step += 1
        
        self.show_progress("Model Comparison", "Comparing Metrics", current_step, self.total_steps)
        self.compare_metrics()
        current_step += 1

        self.show_progress("Model Comparison", "Completed", current_step, self.total_steps)
        print("\nComparison of models complete.")

if __name__ == "__main__":
    comparator = ModelComparison()
    comparator.run_comparison()
