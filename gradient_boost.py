import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor, plot_importance

input_path = 'diamonds.csv'
output_path = 'results.txt'
train = True
test = True
feature_importance = False

def preprocess(path):
    dataset = pd.read_csv(path, index_col=0)
    dataset = dataset.reset_index(drop=True)

    # Data Cleaning
    dataset = dataset.dropna()
    dataset = dataset[(dataset["x"] > 0) & (dataset["y"] > 0) & (dataset["z"] > 0)]

    # Encoding
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    encoder_cut = OrdinalEncoder(categories=[cut_order])
    encoder_clarity = OrdinalEncoder(categories=[clarity_order])

    dataset['cut'] = encoder_cut.fit_transform(dataset[['cut']])
    dataset['clarity'] = encoder_clarity.fit_transform(dataset[['clarity']])
    dataset = pd.get_dummies(dataset, columns=['color'], drop_first=True)

    # Defining the features and target
    x = dataset.drop(columns=['price'])
    y = dataset['price']

    return train_test_split(x, y, test_size=0.2, random_state=42)

def train_XGBRegressor(x, y):
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100,    # Number of trees
                        learning_rate=0.1,    # Step size shrinkage
                        max_depth=3,          # Maximum tree depth
                        subsample=0.8,        # Subsample ratio
                        colsample_bytree=0.8, # Subsample ratio of columns
                        random_state=42)      # For reproducibility
    start_time = time.time()
    model.fit(x_train,
              y_train,
              eval_set=[(x_validation, y_validation)],
              verbose=True)
    end_time = time.time()
    return model, end_time - start_time

def test_regression_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    mean = np.mean(y_test)
    stddev = np.std(y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Price of Test Set: {mean:.2f}")
    print(f"Standard Deviation of Test Set: {stddev:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

def feature_importances(model):
    plot_importance(model)
    plt.show()

def main():
    # Preprocess Data
    x_train, x_test, y_train, y_test = preprocess(input_path)

    # Train the Model
    if train:
        model, execution_time = train_XGBRegressor(x_train, y_train)
        print(f"Execution time: {execution_time:.2f} seconds")
    with open(output_path, 'w') as f:
        sys.stdout = f
        # Test the Model
        if test:
            print("TRAIN RESULTS:\n")
            test_regression_model(model, x_train, y_train)
            print("\nTEST RESULTS:\n")
            test_regression_model(model, x_test, y_test)
        if feature_importance:
            print("\nFEATURE IMPORTANCES:\n")
            feature_importances(model)

if __name__ == '__main__':
    main()