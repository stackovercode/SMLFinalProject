import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Load dataset
data = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data, header=None)

# Define directory for saving plots
plot_dir = './TestingPlots/linLogReg_MLX'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

ROC_plot = os.path.join(plot_dir, 'ROC_plot.png')
confusion_plot = os.path.join(plot_dir, 'confusion_plot.png')
Learning_Curves = os.path.join(plot_dir, 'Learning_Curves.png')

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
df_sample = df.sample(frac=0.1, random_state=42)

# Split dataset into features and target variable
X = df_sample.drop('class', axis=1).values
y = df_sample['class'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 5)  # 5 classes

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

# Move model to MPS device if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = LogisticRegressionModel(X_train_tensor.shape[1]).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    with torch.no_grad():
        train_outputs = model(X_train_tensor.to(device))
        _, train_predicted = torch.max(train_outputs.data, 1)
        train_accuracy = (train_predicted.cpu() == y_train_tensor).sum().item() / y_train_tensor.size(0)
        train_accuracies.append(train_accuracy)
    
    # Calculate validation accuracy
    with torch.no_grad():
        val_outputs = model(X_test_tensor.to(device))
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_accuracy = (val_predicted.cpu() == y_test_tensor).sum().item() / y_test_tensor.size(0)
        val_accuracies.append(val_accuracy)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted.cpu() == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy: {accuracy:.4f}')
    
# Confusion Matrix
cm = confusion_matrix(y_test, predicted.cpu())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2, 3, 4])
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix')
plt.savefig(confusion_plot)
#plt.show()

# Binarize the output
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
n_classes = y_test_bin.shape[1]

# Calculate probabilities
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.savefig(ROC_plot)
#plt.show()

# Plot learning curves
plt.figure()
plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Learning Curves')
plt.legend()
plt.savefig(Learning_Curves)
#plt.show()
