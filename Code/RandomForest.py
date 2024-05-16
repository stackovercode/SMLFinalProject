
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve


# Load dataset with dtype specification and error handling
data = '4_Classification of Robots from their conversation sequence_Set2.csv'
data_types = {i: float for i in range(10)}  # Columns 0-9 as floats
data_types[10] = 'category'  # Column 10 as category

df = pd.read_csv(data, header=None)

# try:
#     df = pd.read_csv(data, header=None)
# except ValueError as e:
#     print(f"Error loading data: {e}")

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
X = df_sample.drop('class', axis=1)
y = df_sample['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Train the model
rfc.fit(X_train, y_train)

# Predict on the test set
y_pred = rfc.predict(X_test)

# Model Evaluation
print('Model accuracy score with 100 decision-trees : {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance
feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), X, y, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))

# Calculate training and test mean and std
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plotting
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="g")
plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.title("Learning Curve")
plt.legend(loc="best")
plt.show()


# df.shape

# """# View top rows of dataset"""

# # preview the dataset
# df.head()

# """We can see that the column names are renamed. Now, the columns have meaningful names.

# # View summary of dataset
# """

# df.info()

# """# Rename column names
# We can see that the dataset does not have proper column names. The columns are merely labelled as 0,1,2.... and so on. We should give proper names to the columns. I will do it as follows:-
# """

# col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# # vhigh,vhigh,2,2,small,med,unacc

# df.columns = col_names

# col_names

# # let's again preview the dataset

# df.head()

# """We can see that the column names are renamed. Now, the columns have meaningful names.

# # View summary of dataset
# """

# df.info()

# """### Frequency distribution of values in variables
# Now, we will check the frequency counts of categorical variables.
# """

# col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# for col in col_names:

#     print(df[col].value_counts())

# """We can see that the doors and persons are categorical in nature. So, I will treat them as categorical variables.

# ### Summary of variables
# * There are 7 variables in the dataset. All the variables are of categorical data type.
# * These are given by buying, maint, doors, persons, lug_boot, safety and class.
# * class is the target variable.


# ## Explore class variable
# """

# df['class'].value_counts()

# """The class target variable is ordinal in nature.

# ### Missing values in variables
# """

# # check missing values in variables

# df.isnull().sum()

# """We can see that there are no missing values in the dataset. I have checked the frequency distribution of values previously. It also confirms that there are no missing values in the dataset.

# # **Declare feature vector and target variable**
# """

# X = df.drop(['class'], axis=1)

# y = df['class']

# """# **Split data into separate training and test set**"""

# # split data into training and testing sets

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# # check the shape of X_train and X_test

# X_train.shape, X_test.shape

# """# **Feature Engineering**

# **Feature Engineering** is the process of transforming raw data into useful features that help us to understand our model better and increase its predictive power. I will carry out feature engineering on different types of variables.

# First, I will check the data types of variables again.
# """

# # check data types in X_train

# X_train.dtypes

# """### **Encode categorical variables**

# Now, we will encode the categorical variables.
# """

# X_train.head()

# """We can see that all the variables are ordinal categorical data type."""

# # import category encoders
# # !pip install category_encoders
# import category_encoders as ce

# # encode categorical variables with ordinal encoding

# encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])


# X_train = encoder.fit_transform(X_train)

# X_test = encoder.transform(X_test)

# X_train.head()

# X_test.head()

# """We now have training and test set ready for model building.

# # **Random Forest Classifier model with default parameters**
# """

# # import Random Forest classifier

# from sklearn.ensemble import RandomForestClassifier

# # instantiate the classifier

# rfc = RandomForestClassifier(random_state=0)

# # fit the model

# rfc.fit(X_train, y_train)

# # Predict the Test set results

# y_pred = rfc.predict(X_test)

# # Check accuracy score

# from sklearn.metrics import accuracy_score

# print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# """Here, y_test are the true class labels and y_pred are the predicted class labels in the test-set.

# Here, I have build the Random Forest Classifier model with default parameter of n_estimators = 10. So, I have used 10 decision-trees to build the model. Now, I will increase the number of decision-trees and see its effect on accuracy.

# # **Random Forest Classifier model with parameter n_estimators=100**
# """

# # instantiate the classifier with n_estimators = 100

# rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)

# # fit the model to the training set

# rfc_100.fit(X_train, y_train)

# # Predict on the test set results

# y_pred_100 = rfc_100.predict(X_test)

# # Check accuracy score

# print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))

# """The model accuracy score with 10 decision-trees is 0.9247 but the same with 100 decision-trees is 0.9457. So, as expected accuracy increases with number of decision-trees in the model.

# # **<font color='red'> <b>Task: Evaluate the training with different estimators and different random state values.</b> </font>**

# # **Find important features with Random Forest model**

# Until now, we have used all the features given in the model. Now, I will select only the important features, build the model using these features and see its effect on accuracy.

# First, we will create the Random Forest model as follows:-
# """

# # create the classifier with n_estimators = 100

# clf = RandomForestClassifier(n_estimators=100, random_state=0)

# # fit the model to the training set

# clf.fit(X_train, y_train)

# """Now, use the feature importance variable to see feature importance scores."""

# # view the feature scores

# feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# feature_scores

# """We can see that the most important feature is safety and least important feature is doors.

# # **Visualize feature scores of the features**

# Now, visualize the feature scores with matplotlib and seaborn.
# """

# # Creating a seaborn bar plot

# sns.barplot(x=feature_scores, y=feature_scores.index)

# # Add labels to the graph

# plt.xlabel('Feature Importance Score')

# plt.ylabel('Features')

# # Add title to the graph

# plt.title("Visualizing Important Features")

# # Visualize the graph

# plt.show()

# """# **Build Random Forest model on selected features**

# Now, drop the least important feature doors from the model, rebuild the model and check its effect on accuracy.
# """

# # declare feature vector and target variable

# X = df.drop(['class', 'doors'], axis=1)

# y = df['class']

# # split data into training and testing sets

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

# """# **<font color='red'> <b>Task: Evaluate the testing with different split size and different random state values.</b> </font>**

# Now, build the random forest model and check accuracy.
# """

# # encode categorical variables with ordinal encoding

# encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'persons', 'lug_boot', 'safety'])

# X_train = encoder.fit_transform(X_train)

# X_test = encoder.transform(X_test)

# # instantiate the classifier with n_estimators = 100

# clf = RandomForestClassifier(random_state=0)

# # fit the model to the training set

# clf.fit(X_train, y_train)

# # Predict on the test set results

# y_pred = clf.predict(X_test)

# # Check accuracy score

# print('Model accuracy score with doors variable removed : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# """* I have removed the doors variable from the model, rebuild it and checked its accuracy. The accuracy of the model with doors variable removed is 0.9264. The accuracy of the model with all the variables taken into account is 0.9247. So, we can see that the model accuracy has been improved with doors variable removed from the model.

# * Furthermore, the second least important model is lug_boot. If I remove it from the model and rebuild the model, then the accuracy was found to be 0.8546. It is a significant drop in the accuracy. So, I will not drop it from the model.

# * Now, based on the above analysis we can conclude that our classification model accuracy is very good. Our model is doing a very good job in terms of predicting the class labels.

# * But, it does not give the underlying distribution of values. Also, it does not tell anything about the type of errors our classifer is making.

# * We have another tool called Confusion matrix that comes to our rescue.

# # **Confusion matrix**
# """

# # Print the Confusion Matrix and slice it into four pieces

# from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test, y_pred)

# print('Confusion matrix\n\n', cm)

# """# **<font color='red'> <b>Task: Plot the confusion matrix</b> </font>**

# # **Classification Report**

# Classification report is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model. I have described these terms in later.

# We can print a classification report as follows:-
# """

# from sklearn.metrics import classification_report

# print(classification_report(y_test, y_pred))

# """
# # **<font color='red'> <b>Task: Implement the same process for the dataset that were previously given to you.</b> </font>**"""
