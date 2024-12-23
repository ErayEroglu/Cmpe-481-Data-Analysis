import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

input_path = 'diamonds.csv'
output_path = 'results.txt'
sample_size = 50000
outlier_removal = True
train = True
test = True
feature_importance = False
cross_validate = False
four_features = False

def preprocess(path):
    dataset = pd.read_csv(path, index_col=0)
    dataset = dataset.sample(n=sample_size, random_state=42)
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

        start_time = time.time()
        cv_scores = cross_validate(
            estimator = model,
            X = x_train,
            y = y_train,
            cv = 5,
            scoring = scoring,
            n_jobs = -1
        )
        end_time = time.time()
    
        rmse_scores = (-cv_scores['test_neg_mse']) ** 0.5
        mae_scores = -cv_scores['test_neg_mae']
        scores[n_estimators] = (cv_scores['test_r2'].mean(), cv_scores['test_r2'].std(),
                                rmse_scores.mean(), rmse_scores.std(),
                                mae_scores.mean(), mae_scores.std(), end_time-start_time)

    for n_estimators, (r2_mean, r2_std, rmse_mean, rmse_std, mae_mean, mae_std, time_) in scores.items():
        print(f"Number of trees: {n_estimators}")
        print(f"Mean R² Score: {r2_mean:.2f}")
        print(f"Standard Deviation of R² Scores: {r2_std:.2f}")
        print(f"Mean RMSE: {rmse_mean:.2f}")
        print(f"Standard Deviation of RMSE: {rmse_std:.2f}")
        print(f"Mean MAE: {mae_mean:.2f}")
        print(f"Standard Deviation of MAE: {mae_std:.2f}")
        print(f"Execution Time: {(time_):.2f} seconds")
        print()

def random_forest_regressor(x_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, max_features="log2", oob_score=True, random_state=42)
    start_time = time.time()
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
    print(f"R² Score: {r2:.3f}")

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

    plt.barh(new_feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Random Forest Feature Importances")
    plt.show()

def main():
    # Preprocess data
    x_train, x_test, y_train, y_test = preprocess(input_path)

    #Train the model
    if train:
        model, execution_time = random_forest_regressor(x_train, y_train)
        print(f"Execution time: {execution_time:.2f} seconds")
    
    with open(output_path, 'w') as f:
        sys.stdout = f
        if cross_validate:
            cross_validation(x_train, y_train)
    
        # Test the model
        if test:
            print("TRAIN RESULTS:\n")
            test_regression_model(model, x_train, y_train)
            print(f"OOB Score (R²): {model.oob_score_:.2f}")

            print("\nTEST RESULTS:\n")
            test_regression_model(model, x_test, y_test)
        
        # Analyze feature importances
        if feature_importance:
            print("\nFEATURE IMPORTANCES:\n")
            feature_importances(model, x_train)

# Plotting Cross Validation Results
def plot():
    trees = [100, 200, 300, 500, 1000]
    mean_r2 = [0.98, 0.98, 0.98, 0.98, 0.98]
    std_r2 = [0.00, 0.00, 0.00, 0.00, 0.00]
    mean_rmse = [607.99, 606.31, 605.80, 606.38, 605.56]
    std_rmse = [21.73, 20.90, 20.08, 20.31, 20.87]
    mean_mae = [303.84, 302.59, 302.38, 302.24, 301.47]
    std_mae = [8.62, 8.09, 7.93, 8.04, 8.05]
    execution_time = [4.83, 7.49, 10.53, 16.28, 54.62]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.errorbar(trees, mean_r2, yerr=std_r2, fmt='o-', label="Mean R²")
    plt.xlabel("Number of Trees")
    plt.ylabel("R²")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.errorbar(trees, mean_rmse, yerr=std_rmse, fmt='o-', color='orange', label="Mean RMSE")
    plt.xlabel("Number of Trees")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.errorbar(trees, mean_mae, yerr=std_mae, fmt='o-', color='green', label="Mean MAE")
    plt.xlabel("Number of Trees")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(trees, execution_time, 'o-', color='red', label="Execution Time")
    plt.xlabel("Number of Trees")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()