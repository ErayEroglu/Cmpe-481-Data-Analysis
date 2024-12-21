from keras.src.layers import Dense
from keras.src.losses import mean_absolute_error
from keras.src.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error
from keras import Sequential, Input
from linear_regression import preprocess_data
import numpy as np
import time

def build_model(input_dim, layers=[64, 32, 16], learning_rate=0.001):
    model = Sequential()
    for neurons in layers:
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])
    return model

def sample_dataset(x, y, fraction=0.2, random_state=42):
    np.random.seed(random_state)
    sample_indices = np.random.choice(len(x), int(len(x) * fraction), replace=True)
    return x.iloc[sample_indices], y.iloc[sample_indices]

def predict_price(file_path):
    x_train, x_test, y_train, y_test = preprocess_data(file_path)
    input_dim = x_train.shape[1]
    x_train_sampled, y_train_sampled = sample_dataset(x_train, y_train, fraction=0.5)

    learning_rates = [0.01, 0.1, 0.2, 0.4, 1.0]
    batch_sizes = [16, 32, 64, 128, 256]
    epochs_list = [10, 20, 50, 100, 200]
    layers_list = [[64, 32], [64, 32, 16], [64, 32, 16, 8], [64, 32, 16, 8, 4], [64, 32, 16, 8, 4, 2]]
    results = {'learning_rate': [], 'batch_size': [], 'epochs': [], 'layers': []}

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_param(param_name, param_values, fixed_params):
        param_mae = []
        for value in param_values:
            fold_mae = []
            for train_index, val_index in kf.split(x_train_sampled):
                x_fold_train, x_fold_val = x_train_sampled.iloc[train_index], x_train_sampled.iloc[val_index]
                y_fold_train, y_fold_val = y_train_sampled.iloc[train_index], y_train_sampled.iloc[val_index]

                # Update parameters dynamically
                params = fixed_params.copy()
                params[param_name] = value

                model = build_model(input_dim, layers=params['layers'], learning_rate=params['learning_rate'])
                model.fit(x_fold_train, y_fold_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

                # Predict on the validation fold
                y_fold_pred = model.predict(x_fold_val).flatten()  # Ensure predictions are 1D
                fold_mae.append(mean_absolute_error(y_fold_val, y_fold_pred))  # Calculate MAE for this fold

            avg_mae = np.mean(fold_mae)  # Compute the average MAE for all folds
            param_mae.append(avg_mae)
        results[param_name] = param_mae

    fixed_params = {
        'learning_rate': 0.01,
        'batch_size': 32,
        'epochs': 50,
        'layers': [64, 32]
    }

    evaluate_param('learning_rate', learning_rates, fixed_params)
    evaluate_param('batch_size', batch_sizes, fixed_params)
    evaluate_param('epochs', epochs_list, fixed_params)
    evaluate_param('layers', layers_list, fixed_params)

    plt.figure(figsize=(16, 12))
    plt.subplot(2, 2, 1)
    plt.plot(learning_rates, results['learning_rate'], marker='o')
    plt.title('Learning Rate vs MAE')
    plt.xlabel('Learning Rate')
    plt.ylabel('MAE')

    plt.subplot(2, 2, 2)
    plt.plot(batch_sizes, results['batch_size'], marker='o')
    plt.title('Batch Size vs MAE')
    plt.xlabel('Batch Size')
    plt.ylabel('MAE')

    plt.subplot(2, 2, 3)
    plt.plot(epochs_list, results['epochs'], marker='o')
    plt.title('Epochs vs MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')

    plt.subplot(2, 2, 4)
    plt.plot(range(len(layers_list)), results['layers'], marker='o')
    plt.title('Layers vs MAE')
    plt.xlabel('Layer Configurations (Index)')
    plt.ylabel('MAE')
    plt.xticks(range(len(layers_list)), [str(l) for l in layers_list], rotation=45)

    plt.tight_layout()
    plt.show()
    plt.savefig("neural_network_metrics.png", dpi=300)

if __name__ == '__main__':
    predict_price("diamonds.csv")