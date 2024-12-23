import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.python.eager.context import execution_mode


def main():
    file_path = "diamonds.csv"
    predict_price(file_path, False)

def preprocess_data(file_path, is_selected_columns=False):
    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Data loaded from {file_path}, shape: {df.shape}")   
    # Drop unnecessary columns
    dataset = df.drop(["Unnamed: 0"], axis=1)

    # Drop rows with missing values or invalid measurements
    dataset = dataset.dropna()

    dataset = dataset[(dataset["x"] > 0) & (dataset["y"] > 0) & (dataset["z"] > 0)]

    # Encode categorical features
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    encoder_cut = OrdinalEncoder(categories=[cut_order])
    encoder_clarity = OrdinalEncoder(categories=[clarity_order])

    dataset['cut'] = encoder_cut.fit_transform(dataset[['cut']])
    dataset['clarity'] = encoder_clarity.fit_transform(dataset[['clarity']])
    dataset = pd.get_dummies(dataset, columns=['color'], drop_first=True)

    # Outlier removal based on IQR
    numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
    for col in numeric_columns:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

    if (is_selected_columns):
        dataset = dataset[[ 'clarity', 'x', 'y', 'z', 'price']]

    X = dataset.drop('price', axis=1) # Features
    y = dataset['price'] # Target variable
    mean_price = y.mean()
    std_price = y.std()
    print(f"Mean price: {mean_price:.2f}")
    print(f"Standard deviation of prices: {std_price:.2f}")

    print("Data preprocessed. Shape of dataset after preprocessing:", dataset.shape)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Data preprocessed and split into training and testing sets.")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def predict_price(file_path, isSelectedColumns=False):
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(file_path, isSelectedColumns)
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    execution_time = time.time() - start_time
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Mean squared error: {mse:.2f}")
    print(f"R^2 score: {r2:.2f}")
    print(f"Mean absolute error: {mae:.2f}")







#main()