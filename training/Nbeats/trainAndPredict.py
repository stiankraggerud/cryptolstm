import os
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib
import stockDataHandler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from nbeats_keras.model import NBeatsNet
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def prepare_data(data, columns_to_percent_return, columns_to_min_max_normalize, min_max_scalers=None):
    """Prepares the data for the model."""
    data = stockDataHandler.percent_return(data, columns_to_percent_return)
    
    if not min_max_scalers:
        data_min_max, min_max_scalers = stockDataHandler.min_max_normalization(data, columns_to_min_max_normalize)
    else:
        data_min_max, _ = stockDataHandler.min_max_normalization(data, columns_to_min_max_normalize, min_max_scalers)

    data[columns_to_min_max_normalize] = data_min_max[columns_to_min_max_normalize]
    data = data.dropna()

    return data, min_max_scalers

def generate_signals(data):
    """Generates trading signals based on given conditions."""
    for i in range(1, len(data)):
        if data['RSI'].iloc[i] < 40 and data['EMA12'].iloc[i] > data['EMA100'].iloc[i] and data['ADX'].iloc[i] > 20:
            data.loc[data.index[i], 'Signal'] = 'Buy'
        elif data['RSI'].iloc[i] > 60 and data['EMA12'].iloc[i] < data['EMA100'].iloc[i] and data['ADX'].iloc[i] > 25:
            data.loc[data.index[i], 'Signal'] = 'Sell'
        else:
            data.loc[data.index[i], 'Signal'] = 'Hold'
    
    data['Signal'].fillna('Hold', inplace=True)
    data.dropna()
    return data


def train_model():
    # Load and preprocess the data
    stock_data = stockDataHandler.LoadData('onehour2018_2023.csv')
    stockDataHandler.SetEMA(stock_data, 12, 'EMA12')
    stockDataHandler.SetEMA(stock_data, 200, 'EMA100')
    stockDataHandler.SetMacd(stock_data, 50)
    stockDataHandler.compute_ADX(stock_data,14)
    stockDataHandler.compute_RSI(stock_data,8)

    stockDataHandler.CleanData(stock_data)

    output_horizon = 1
    numberOfBars = 100

    stock_data = generate_signals(stock_data)

    # Split the data
    train_percent = 0.7
    val_percent = 0.15

    training_data_len = int(len(stock_data) * train_percent)
    val_data_len = int(len(stock_data) * val_percent)

    train_data = stock_data[:training_data_len]
    val_data = stock_data[training_data_len:training_data_len + val_data_len]
    test_data = stock_data[training_data_len + val_data_len:]


    columns_to_percent_return = ['Open', 'Close', 'High', 'Low', 'RSI', 'ADX']
    columns_to_min_max_normalize = ['Volume', 'EMA12', 'EMA100']

    train_data, min_max_scalers = prepare_data(train_data, columns_to_percent_return, columns_to_min_max_normalize)
    val_data, _ = prepare_data(val_data, columns_to_percent_return, columns_to_min_max_normalize, min_max_scalers)



    # columns_to_percent_return = ['Open', 'Close', 'High', 'Low', 'RSI', 'ADX']
    # columns_to_min_max_normalize = ['Volume', 'EMA12', 'EMA100']

    # # Apply transformations to training data
    # train_data = stockDataHandler.percent_return(train_data, columns_to_percent_return)
    # train_data_min_max, min_max_scalers = stockDataHandler.min_max_normalization(train_data, columns_to_min_max_normalize)
    # train_data[columns_to_min_max_normalize] = train_data_min_max[columns_to_min_max_normalize]
    # train_data = train_data.dropna()

    # Apply transformations to validation data
    val_data = stockDataHandler.percent_return(val_data, columns_to_percent_return)
    val_data_min_max, _ = stockDataHandler.min_max_normalization(val_data, columns_to_min_max_normalize, min_max_scalers)
    val_data[columns_to_min_max_normalize] = val_data_min_max[columns_to_min_max_normalize]
    val_data = val_data.dropna()

    # Create training dataset
    x_train = []
    y_train = []

    for i in range(numberOfBars, len(train_data) - output_horizon):
        x_train.append(train_data[columns_to_percent_return + columns_to_min_max_normalize].iloc[i - numberOfBars:i].values)
        y_train.append(train_data['Close'].iloc[i:i+output_horizon].values)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Create validation dataset
    x_val = []
    y_val = []

    for i in range(numberOfBars, len(val_data) - output_horizon):
        x_val.append(val_data[columns_to_percent_return + columns_to_min_max_normalize].iloc[i - numberOfBars:i].values)
        y_val.append(val_data['Close'].iloc[i:i+output_horizon].values)

    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # Create N-BEATS model
    import warnings
    import numpy as np
    from nbeats_keras.model import NBeatsNet

    model = NBeatsNet(backcast_length=numberOfBars, forecast_length=output_horizon,
                    stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=3,
                    thetas_dim=(4, 4), share_weights_in_stack=True, hidden_layer_units=256)

    model.compile(loss='mae', optimizer='adam')
    model.summary()



    # Train the model with validation data
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=100)
    joblib.dump(history, stockDataHandler.get_full_path('onehour_2018_2023_nbeats_history_100epoch.pkl'))

    
    newModelname = stockDataHandler.get_full_path('onehour_2018_2023_nbeats1.h5')
    model.save(newModelname)
    joblib.dump(min_max_scalers, stockDataHandler.get_full_path('min_max_scalers_nbeats.pkl'))

def load_model_and_predict():
    # Load previously saved model and scalers
    model_path = stockDataHandler.get_full_path('onehour_2018_2023_nbeats.h5')
    loaded_model = keras.models.load_model(model_path)

    scalers_path = stockDataHandler.get_full_path('min_max_scalers_nbeats.pkl')
    loaded_scalers = joblib.load(scalers_path)

    # Load and preprocess the test data
    stock_data = stockDataHandler.LoadData('onehour2018_2023.csv')
    stockDataHandler.SetEMA(stock_data, 12, 'EMA12')
    stockDataHandler.SetEMA(stock_data, 200, 'EMA100')
    stockDataHandler.SetMacd(stock_data, 50)
    stockDataHandler.compute_ADX(stock_data, 14)
    stockDataHandler.compute_RSI(stock_data, 8)
    stockDataHandler.CleanData(stock_data)

    output_horizon = 1
    numberOfBars = 100

    # Assuming the test data starts after training and validation data
    train_percent = 0.7
    val_percent = 0.15
    training_data_len = int(len(stock_data) * train_percent)
    val_data_len = int(len(stock_data) * val_percent)
    test_data = stock_data[training_data_len + val_data_len:]

    columns_to_percent_return = ['Open', 'Close', 'High', 'Low', 'RSI', 'ADX']
    columns_to_min_max_normalize = ['Volume', 'EMA12', 'EMA100']
    test_data, _ = prepare_data(test_data, columns_to_percent_return, columns_to_min_max_normalize, loaded_scalers)

    test_data.reset_index(inplace=True)
    test_data.rename(columns={'index': 'Original_Close'}, inplace=True)

    x_test_train = []  # For trening og prediksjon
    y_test = []  # For etterbehandling og analyse

    for i in range(numberOfBars, len(test_data) - output_horizon):
        x_test_train.append(test_data[columns_to_percent_return + columns_to_min_max_normalize].iloc[i - numberOfBars:i].values)
        y_test.append(test_data['Original_Close'].iloc[i:i + output_horizon].values)

    x_test_train = np.array(x_test_train)
    y_test = np.array(y_test)

    # Use 'loaded_model' instead of 'model' for prediction
    predicted_values = loaded_model.predict(x_test_train)
    
    # Flatten the arrays
    predicted_values_flat = np.squeeze(predicted_values)
    y_test_flat = np.squeeze(y_test)

    reconstructed_values = [y_test_flat[0]]  # Starter med den første faktiske verdien

    for i in range(1, len(predicted_values_flat)):
        reconstructed_value = reconstructed_values[-1] * (1 + (predicted_values_flat[i] / 100))
        reconstructed_values.append(reconstructed_value)

    # Visualization
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_flat[1:], label='True')
    plt.plot(reconstructed_values, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# MAIN
choice = '2' # input("Do you want to (1) train the model or (2) load a saved model and predict? Enter 1 or 2: ")

if choice == '1':
    train_model()
elif choice == '2':
    load_model_and_predict()
else:
    print("Invalid choice.")

