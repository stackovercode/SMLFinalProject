import numpy as np
import pandas as pd

# Define the number of sequences (rows) to generate
num_sequences = 1000

# Define the number of features
num_features = 10

# Initialize an empty list to hold the rows of the dataset
data = []

# Define the properties of the number sequences for each robot
robot_properties = {
    0: {"mean": 0, "std": 1},   # Normal distribution with mean 0 and std 1
    1: {"low": 0, "high": 10},  # Uniform distribution between 0 and 10
    2: {"mean": 5, "std": 2},   # Normal distribution with mean 5 and std 2
    3: {"low": -5, "high": 5},  # Uniform distribution between -5 and 5
    4: {"mean": 10, "std": 3}   # Normal distribution with mean 10 and std 3
}

# Generate the dataset
for i in range(num_sequences):
    # Randomly select a robot (source)
    robot = np.random.choice(list(robot_properties.keys()))
    
    # Generate the feature values based on the robot's properties
    if "std" in robot_properties[robot]:
        # Normal distribution
        features = np.random.normal(
            loc=robot_properties[robot]["mean"],
            scale=robot_properties[robot]["std"],
            size=num_features
        )
    else:
        # Uniform distribution
        features = np.random.uniform(
            low=robot_properties[robot]["low"],
            high=robot_properties[robot]["high"],
            size=num_features
        )
    
    # Append the source and features to the dataset
    data.append([robot] + list(features))

# Convert to a pandas DataFrame
columns = ["Source"] + [f"num{i+1}" for i in range(num_features)]
df = pd.DataFrame(data, columns=columns)

# Display the first few rows of the dataset
print(df.head())
