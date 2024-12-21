import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

def main():
    file_path = "diamonds.csv"
    predict_price(file_path)

def preprocess_data(file_path):
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

def predict_price(file_path):
    # Preprocess the data
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    model = LinearRegression()

    print("Performing 5-fold cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rmse_scores = []
    r2_scores = []
    mae_scores = []
    coefficients_list = []

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(X_fold_train, y_fold_train)
        y_fold_pred = model.predict(X_fold_val)

        rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
        r2 = r2_score(y_fold_val, y_fold_pred)
        mae = mean_absolute_error(y_fold_val, y_fold_pred)

        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mae_scores.append(mae)
        coefficients_list.append(model.coef_)

    # Compute average coefficients
    avg_coefficients = np.mean(coefficients_list, axis=0)
    coefficients_df = pd.DataFrame({'Feature': X_train.columns, 'Average Coefficient': avg_coefficients})

    # Display results
    print("\nCross-Validation Results:")
    print(f"Mean RMSE: {np.mean(rmse_scores):.2f}, Std: {np.std(rmse_scores):.2f}")
    print(f"Mean R²: {np.mean(r2_scores):.2f}, Std: {np.std(r2_scores):.2f}")
    print(f"Mean MAE: {np.mean(mae_scores):.2f}, Std: {np.std(mae_scores):.2f}")

    print("\nFeature Coefficients (Averaged Across Folds):")
    print(coefficients_df)

    # Plot cross-validation metrics
    folds = range(1, 6)
    plt.figure(figsize=(12, 8))

    # RMSE Plot
    plt.subplot(3, 1, 1)
    plt.plot(folds, rmse_scores, marker='o', linestyle='-', color='b', label='RMSE per fold')
    plt.axhline(y=np.mean(rmse_scores), color='r', linestyle='--', label='Mean RMSE')
    plt.title("5-Fold Cross-Validation RMSE")
    plt.xlabel("Fold Number")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)

    # R² Plot
    plt.subplot(3, 1, 2)
    plt.plot(folds, r2_scores, marker='o', linestyle='-', color='g', label='R² per fold')
    plt.axhline(y=np.mean(r2_scores), color='r', linestyle='--', label='Mean R²')
    plt.title("5-Fold Cross-Validation R²")
    plt.xlabel("Fold Number")
    plt.ylabel("R²")
    plt.legend()
    plt.grid(True)

    # MAE Plot
    plt.subplot(3, 1, 3)
    plt.plot(folds, mae_scores, marker='o', linestyle='-', color='purple', label='MAE per fold')
    plt.axhline(y=np.mean(mae_scores), color='r', linestyle='--', label='Mean MAE')
    plt.title("5-Fold Cross-Validation MAE")
    plt.xlabel("Fold Number")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig("linear_regression_metric.png", dpi=300)
    print("Cross-validation metrics plot saved as 'linear_regression_metrics.png'.")



#main()