# Iris Flower Classification

## Project Overview

This project is developed as part of the CodeAlpha Data Science Internship. The objective is to classify Iris flowers into three species using a Machine Learning model.

## Objective

To predict the species of an Iris flower based on its measurements:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The three possible species are:

* Iris-setosa
* Iris-versicolor
* Iris-virginica

## Dataset

The project uses the **Iris.csv** dataset, which contains flower measurements and their corresponding species.

## Technologies Used

* Python
* Pandas
* Scikit-learn

## Machine Learning Algorithm

* Decision Tree Classifier

## Steps Performed

1. Load the Iris dataset.
2. Remove the Id column.
3. Select input features and target variable.
4. Split the dataset into training and testing sets.
5. Train the Decision Tree model.
6. Evaluate the model accuracy.
7. Accept user input for flower measurements.
8. Predict the Iris flower species.

## Files Included

* iris_classification.py
* Iris.csv
* README.md
* requirements.txt

## How to Run

1. Install the required libraries:

   ```
   pip install pandas scikit-learn
   ```

2. Run the program:

   ```
   python iris_classification.py
   ```

3. Enter the flower measurements when prompted.

## Sample Output

Model Accuracy: 100.0%

Example Input:

* Sepal Length: 5.1
* Sepal Width: 3.5
* Petal Length: 1.4
* Petal Width: 0.2

Predicted Iris Flower Species:
Iris-setosa

## Conclusion

The model successfully classifies Iris flowers based on their measurements and demonstrates the basic concept of classification in machine learning.
