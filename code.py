import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
iris = sns.load_dataset('iris')

# Display information about the dataset
print("Iris Dataset Information:")
print(iris.info())

# Visualize pairwise relationships between features, colored by species
sns.pairplot(iris, hue='species')
plt.title('Pairplot of Iris Dataset by Species')
plt.show()

# Filter data for 'setosa' species
setosa = iris[iris['species'] == 'setosa']

# Create a 2D density plot of sepal width vs. sepal length for setosa
sns.kdeplot(data=setosa, x='sepal_width', y='sepal_length', cmap='magma', fill=True)
plt.title('Density Plot of Sepal Width vs. Sepal Length for Setosa')
plt.show()

# Split data into features (X) and target (y)
X = iris.drop('species', axis=1)
y = iris['species']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Create an initial SVM model with default parameters
model = SVC()

# Train the SVM model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model's performance
print("\nInitial SVM Model Performance:")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Define a grid of hyperparameters to search over
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

# Use GridSearchCV to find the best hyperparameters
grid = GridSearchCV(SVC(), param_grid, verbose=3, refit=True)

# Train the GridSearchCV object on the training data
grid.fit(X_train, y_train)

# Get the best hyperparameters found by GridSearchCV
print("\nBest Hyperparameters:", grid.best_params_)

# Make predictions on the test data using the best model
grid_predictions = grid.predict(X_test)

# Evaluate the performance of the model with the best hyperparameters
print("\nOptimized SVM Model Performance:")
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
