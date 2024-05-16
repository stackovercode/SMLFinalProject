import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# Load dataset
data = '4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data, header=None)

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
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.to(device))
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted.cpu() == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Accuracy: {accuracy:.4f}')
    

