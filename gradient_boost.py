import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor, plot_importance

input_path = 'diamonds.csv'
output_path = 'results.txt'
outlier_removal = True
train = True
test = True
feature_importance = False
cross_validate = False
four_features = False

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

    # Outlier Removal
    if outlier_removal:
        numeric_columns = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
        for col in numeric_columns:
            Q1 = dataset[col].quantile(0.25)
            Q3 = dataset[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

    # Keep only 4 "most important" features
    if four_features:
        dataset = dataset[['carat', 'x', 'y', 'z', 'price']]

    # Defining the features and target
    x = dataset.drop(columns=['price'])
    y = dataset['price']

    return train_test_split(x, y, test_size=0.2, random_state=42)

def train_XGBRegressor(x, y):
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=95,
                        learning_rate=0.087,
                        max_depth=9,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        importance_type='total_gain',
                        early_stopping_rounds=10,
                        random_state=42)
    start_time = time.time()
    model.fit(x_train,
              y_train,
              eval_set=[(x_validation, y_validation)],
              verbose=False)
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
    print(f"R² Score: {r2:.3f}")

def feature_importances(model):
    plot_importance(model)
    plt.show()

def cross_validation(x_train, y_train):
    param_grid = {'n_estimators': [95],
                  'learning_rate': [0.087],
                  'max_depth': [5, 6, 7, 8, 9, 10, 11, 12],
                  'subsample': [0.8],
                  'colsample_bytree': [0.8]}
    grid_search = GridSearchCV(estimator=XGBRegressor(random_state=42),
                               param_grid=param_grid,
                               scoring='neg_root_mean_squared_error',
                               refit='neg_root_mean_squared_error',
                               cv=5,
                               verbose=2)
    grid_search.fit(x_train, y_train)
    df = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(10, 6))
    plt.plot(df['param_max_depth'], -df['mean_test_score'], marker='o', label='Mean RMSE')
    plt.title('MAX DEPTH')
    plt.xlabel('max_depth')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.legend()
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
        # 5-Fold Cross Validation
        if cross_validate:
            cross_validation(x_train, y_train)
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