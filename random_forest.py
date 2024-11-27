import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

input_path = 'diamonds.csv'
output_path = 'results.txt'
sample_size = 50000

def preprocess(path):
    dataset = pd.read_csv(path, index_col=0)
    dataset = dataset.sample(n=sample_size, random_state=42)
    dataset = dataset.reset_index(drop=True)
    dataset.columns = dataset.columns.str.strip()
    dataset = dataset.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Encoding
    cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    encoder_cut = OrdinalEncoder(categories=[cut_order])
    encoder_clarity = OrdinalEncoder(categories=[clarity_order])

    dataset['cut'] = encoder_cut.fit_transform(dataset[['cut']])
    dataset['clarity'] = encoder_clarity.fit_transform(dataset[['clarity']])
    dataset = pd.get_dummies(dataset, columns=['color'], drop_first=True)

    x = dataset.drop(columns=['price'])
    y = dataset['price']

    return train_test_split(x, y, test_size=0.2, random_state=42)

def cross_validation(x_train, y_train):
    scores = dict()
    n_estimators_range = [100, 200, 300, 500, 1000]
    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error'
    }
    for n_estimators in n_estimators_range:
        model = RandomForestRegressor(n_estimators=n_estimators,  max_features="log2", random_state=42)

        cv_scores = cross_validate(
            estimator = model,
            X = x_train,
            y = y_train,
            cv = 5,
            scoring = scoring,
            n_jobs = -1
        )
    
        rmse_scores = (-cv_scores['test_neg_mse']) ** 0.5
        mae_scores = -cv_scores['test_neg_mae']
        scores[n_estimators] = (cv_scores['test_r2'].mean(), cv_scores['test_r2'].std(),
                                rmse_scores.mean(), rmse_scores.std(),
                                mae_scores.mean(), mae_scores.std())

    for n_estimators, (r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std) in scores.items():
        print(f"Number of trees: {n_estimators}")
        print(f"Mean R² Score: {r2_mean:.2f}")
        print(f"Standard Deviation of R² Scores: {r2_std:.2f}")
        print(f"Mean RMSE: {rmse_mean:.2f}")
        print(f"Standard Deviation of RMSE: {rmse_std:.2f}")
        print(f"Mean MAE: {mae_mean:.2f}")
        print(f"Standard Deviation of MAE: {mae_std:.2f}")
        print()

def random_forest_regressor(x_train, y_train):
    start_time = time.time()
    rf_model = RandomForestRegressor(n_estimators=100, max_features="log2", oob_score=True, random_state=42)
    rf_model.fit(x_train, y_train)
    end_time = time.time()

    return rf_model, end_time - start_time

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

def feature_importances(model, x_train):
    importances = model.feature_importances_
    feature_names = x_train.columns
    sorted_indices = importances.argsort()[::-1]
    
    color = 0
    color_features = list()
    for i in sorted_indices:
        if feature_names[i].startswith('color'):
            color += importances[i]
            color_features.append(i)

    importances = np.delete(importances, color_features)
    importances = np.append(importances, color)
    new_feature_names = feature_names.drop(feature_names[color_features])
    new_feature_names = new_feature_names.append(pd.Index(['color']))
    sorted_indices = importances.argsort()[::-1]

    for i in sorted_indices:
        print(f"{new_feature_names[i]}: {importances[i]:.4f}")

    # plt.barh(new_feature_names, importances)
    # plt.xlabel("Feature Importance")
    # plt.ylabel("Features")
    # plt.title("Random Forest Feature Importances")
    # plt.show()

def main():
    # Preprocess data
    x_train, x_test, y_train, y_test = preprocess(input_path)

    #Train the model
    model, execution_time = random_forest_regressor(x_train, y_train)
    print(f"Execution time: {execution_time:.2f} seconds")
    
    with open(output_path, 'w') as f:
        sys.stdout = f
        # cross_validation(x_train, y_train)

        # Test the model
        print("TRAIN RESULTS:\n")
        test_regression_model(model, x_train, y_train)
        print(f"OOB Score (R²): {model.oob_score_:.2f}")
    
        print("\nTEST RESULTS:\n")
        test_regression_model(model, x_test, y_test)
    
        # Analyze feature importances
        print("\nFEATURE IMPORTANCES:\n")
        feature_importances(model, x_train)

if __name__ == '__main__':
    main()