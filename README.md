# Iris Flower Classification with SVM

This project demonstrates how to build and optimize a Support Vector Machine (SVM) model to classify iris flowers into their respective species (setosa, versicolor, virginica) based on their features.

## Contents

* `code.py`: Python script containing the data loading, preprocessing, model training, and evaluation steps.
* `LICENSE`: MIT license file granting permissions for using and modifying the code.

## Dataset

The famous Iris flower dataset is used, which is included in the seaborn library.

## Libraries Used

* pandas
* numpy
* matplotlib
* sklearn
* seaborn

## How to Use

1. Make sure you have the required libraries installed (`pip install pandas numpy matplotlib scikit-learn seaborn`).
2. Run the `code.py` script. The output will include:
    * Pairplot visualization of the dataset.
    * Density plot of sepal width vs. sepal length for setosa species.
    * Classification report of the initial SVM model.
    * Best hyperparameters found using GridSearchCV.
    * Classification report of the optimized SVM model.-

## Results

The optimized SVM model achieves a high accuracy(98%) in classifying iris flowers, demonstrating the effectiveness of this technique.
