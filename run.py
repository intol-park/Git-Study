import yaml
import os
import tensorflow as tf
from models import *
from datasets import *
import json
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def normalize_data(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_norm = (x_train - mean) / std
    x_test_norm = (x_test - mean) / std
    return x_train_norm, x_test_norm, mean, std

def main():
    # Load configurations
    regression_config = load_config('configs/regression.yaml')
    cnn1d_regression_config = load_config('configs/cnn1d_regression.yaml')
    print(regression_config)
    print(cnn1d_regression_config)

    # Initialize models
    regression_model = RegressionModel(regression_config)
    cnn1d_regression_model = CNN1DRegressionModel(cnn1d_regression_config)

    # Load dataset (using synthetic data for 1D CNN example)
    (x_train, y_train), (x_test, y_test) = load_dataset('boston_housing')

    # Normalize data
    x_train, x_test, x_mean, x_std = normalize_data(x_train, x_test)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    # Reshape data for 1D CNN (example only)
    x_train_cnn1d = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test_cnn1d = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Build models with input shape
    regression_model.build(x_train.shape[1:])
    cnn1d_regression_model.build(x_train_cnn1d.shape[1:])
    
    save_model_path = "results/saved_models"
    os.makedirs(save_model_path, exist_ok=True)  # Ensure the save directory exists

    # Compile and train regression model
    regression_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=regression_config['learning_rate']),
                             loss='mse',
                             metrics=['mae'])
    regression_model.fit(x_train, y_train, epochs=regression_config['epochs'], batch_size=regression_config['batch_size'])
    regression_model.save(os.path.join(save_model_path, 'regression.keras'))
    regression_save_config = regression_model.get_config()
    with open(os.path.join(save_model_path, 'regression_config.json'), 'w') as f:
        json.dump(regression_save_config, f)

    # Compile and train CNN1D regression model
    cnn1d_regression_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cnn1d_regression_config['learning_rate']),
                                   loss='mse',
                                   metrics=['mae'])
    cnn1d_regression_model.fit(x_train_cnn1d, y_train, epochs=cnn1d_regression_config['epochs'], batch_size=cnn1d_regression_config['batch_size'])
    cnn1d_regression_model.save(os.path.join(save_model_path, 'cnn1d_regression.keras'))
    cnn1d_save_config = cnn1d_regression_model.get_config()
    with open(os.path.join(save_model_path, 'cnn1d_regression_config.json'), 'w') as f:
        json.dump(cnn1d_save_config, f)

    # Load models
    loaded_regression_model = tf.keras.models.load_model(
        os.path.join(save_model_path, 'regression.keras'), custom_objects={"RegressionModel": RegressionModel})
    loaded_cnn1d_regression_model = tf.keras.models.load_model(
        os.path.join(save_model_path, 'cnn1d_regression.keras'), custom_objects={"CNN1DRegressionModel": CNN1DRegressionModel})


    # Evaluate loaded models
    loaded_regression_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=regression_config['learning_rate']),
                                    loss='mse',
                                    metrics=['mae'])
    loss, mae = loaded_regression_model.evaluate(x_test, y_test)
    print(f'Loaded regression model evaluation - Loss: {loss}, MAE: {mae}')

    loaded_cnn1d_regression_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cnn1d_regression_config['learning_rate']),
                                          loss='mse',
                                          metrics=['mae'])
    loss, mae = loaded_cnn1d_regression_model.evaluate(x_test_cnn1d, y_test)
    print(f'Loaded CNN1D regression model evaluation - Loss: {loss}, MAE: {mae}')

    # 예측 수행 및 결과 출력
    save_csv_path = "results/csv/"
    os.makedirs(save_csv_path, exist_ok=True)
    y_pred_regression = loaded_regression_model.predict(x_test)
    y_pred_cnn1d = loaded_cnn1d_regression_model.predict(x_test_cnn1d)

    # Un-normalize predictions
    y_pred_regression = y_pred_regression * y_std + y_mean
    y_pred_cnn1d = y_pred_cnn1d * y_std + y_mean
    y_test = y_test * y_std + y_mean

    # Calculate RMSE and R-squared for regression model
    rmse_regression = mean_squared_error(y_test, y_pred_regression, squared=False)
    r2_regression = r2_score(y_test, y_pred_regression)

    # Calculate RMSE and R-squared for CNN1D regression model
    rmse_cnn1d = mean_squared_error(y_test, y_pred_cnn1d, squared=False)
    r2_cnn1d = r2_score(y_test, y_pred_cnn1d)

    # Create DataFrame to store evaluation results
    results = pd.DataFrame({
        'True Values': y_test,
        'Regression Predictions': y_pred_regression.flatten(),
        'CNN1D Predictions': y_pred_cnn1d.flatten()
    })
    results['RMSE Regression'] = rmse_regression
    results['R2 Regression'] = r2_regression
    results['RMSE CNN1D'] = rmse_cnn1d
    results['R2 CNN1D'] = r2_cnn1d

    # Save results to a CSV file
    results.to_csv(os.path.join(save_csv_path, 'evaluation_results.csv'), index=False)

    print(results)

if __name__ == '__main__':
    main()
