# Unfinished project with the aim to set up a 
# data visualization automation process
# for machine learning from datasets that can
# distinguish the type of regression necessary


# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston, load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_boston()  # or load_iris() for classification problem
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name="target")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a dictionary of models to be used
models = {
    "Linear Regression": LinearRegression(),
    "Logistic Regression": LogisticRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Bayesian Linear Regression": BayesianRidge(),
    "Polynomial Regression": LinearRegression()
}

# Define a dictionary of model hyperparameters to be used
params = {
    "Polynomial Regression": {"degree": 2}
}

# Define a list of features to be plotted
features = list(X.columns)

# Loop through the models and plot the features
for model_name, model in models.items():
    # Fit the model on the training data
    if model_name == "Polynomial Regression":
        degree = params[model_name]["degree"]
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    
    # Calculate the predictions on the testing data
    y_pred = model.predict(X_test)
    
    # Plot the features
    for feature in features:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_test[feature], y_test, color='black')
        plt.plot(X_test[feature], y_pred, color='blue', linewidth=3)
        plt.xlabel(feature)
        plt.ylabel('target')
        plt.title(f"{model_name} - {feature}")
        plt.show()

# Havent been able to make it work yet :/