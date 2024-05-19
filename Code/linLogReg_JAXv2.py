import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from jax import random
import optax  # A gradient processing and optimization library for JAX
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset, skipping the initial header row
data = './4_Classification of Robots from their conversation sequence_Set2.csv'
df = pd.read_csv(data, header=0)

# Check for the correct column headers
print(df.head())
print(df.info())

# Assign column names if necessary
col_names = ['source', 'num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8', 'num9', 'num10', 'class']
if df.columns[0] != 'source':  # Assuming first column should be 'source'
    df.columns = col_names

# Drop the first row if it contains column headers or irrelevant data
if df.iloc[0, 0] == 'source':
    df = df.drop(0)

# Convert numerical columns to float
for col in col_names[1:-1]:  # Exclude 'source' and 'class' which are categorical
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert class to categorical integer labels
df['class'] = pd.Categorical(df['class']).codes  # Convert to categorical codes if they are not numeric

# Handle missing data if any
df.dropna(inplace=True)

# Sample the data to reduce size for quicker processing
df_sample = df.sample(frac=0.1, random_state=42)

# Split dataset into features and target variable
X = df_sample.drop(['source', 'class'], axis=1).values
y = df_sample['class'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to JAX arrays
X_train_jax = jnp.array(X_train_scaled)
y_train_jax = jnp.array(y_train)
X_test_jax = jnp.array(X_test_scaled)
y_test_jax = jnp.array(y_test)

# Initialize parameters
key = random.PRNGKey(0)
input_dim = X_train_jax.shape[1]
output_dim = len(np.unique(y))
params = {
    "weights": random.normal(key, (input_dim, output_dim)),
    "bias": jnp.zeros(output_dim)
}

# Define logistic regression model
def model(params, X):
    return jnp.dot(X, params["weights"]) + params["bias"]

# Define loss function (cross-entropy)
def loss_fn(params, X, y):
    logits = model(params, X)
    one_hot = jnp.eye(output_dim)[y]
    return -jnp.mean(jnp.sum(one_hot * logits - jnp.log(1 + jnp.exp(logits)), axis=1))

# Compile with JIT
loss_fn_jit = jit(loss_fn)
grad_fn = jit(grad(loss_fn))

# Training loop
learning_rate = 0.01
num_epochs = 100
optimizer = optax.sgd(learning_rate)
opt_state = optimizer.init(params)

train_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    grads = grad_fn(params, X_train_jax, y_train_jax)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    train_loss = loss_fn_jit(params, X_train_jax, y_train_jax)
    train_losses.append(train_loss)
    
    logits = model(params, X_test_jax)
    predictions = jnp.argmax(logits, axis=1)
    accuracy = jnp.mean(predictions == y_test_jax)
    val_accuracies.append(accuracy)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Validation Accuracy: {accuracy:.4f}')

# Evaluation and plotting
plt.figure()
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.title('Training Curve')
plt.savefig(Learning_Curves)

# Confusion Matrix
logits = model(params, X_test_jax)
predictions = jnp.argmax(logits, axis=1)
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig(confusion_plot)

# ROC Curve
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
probabilities = jax.nn.softmax(logits)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(output_dim):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue', 'green', 'red']
for i, color in zip(range(output_dim), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend(loc="lower right")
plt.savefig(ROC_plot)
