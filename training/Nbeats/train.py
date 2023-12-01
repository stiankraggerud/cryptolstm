import os
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib
import stockDataHandler
from sklearn.preprocessing import MinMaxScaler
import math



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

# Label generation
for i in range(1, len(stock_data)):
    if stock_data['RSI'].iloc[i] < 40 and stock_data['EMA12'].iloc[i] > stock_data['EMA100'].iloc[i] and stock_data['ADX'].iloc[i] > 20:
        stock_data.loc[stock_data.index[i], 'Signal'] = 'Buy'
    elif stock_data['RSI'].iloc[i] > 60 and stock_data['EMA12'].iloc[i] < stock_data['EMA100'].iloc[i] and stock_data['ADX'].iloc[i] > 25:
        stock_data.loc[stock_data.index[i], 'Signal'] = 'Sell'
    else:
        stock_data.loc[stock_data.index[i], 'Signal'] = 'Hold'

stock_data['Signal'].fillna('Hold', inplace=True)
stock_data = stock_data.dropna()

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

# Apply transformations to training data
train_data = stockDataHandler.percent_return(train_data, columns_to_percent_return)
train_data_min_max, min_max_scalers = stockDataHandler.min_max_normalization(train_data, columns_to_min_max_normalize)
train_data[columns_to_min_max_normalize] = train_data_min_max[columns_to_min_max_normalize]
train_data = train_data.dropna()

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
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=1000)
joblib.dump(history, stockDataHandler.get_full_path('onehour_2018_2023_nbeats_history_1000epoch.pkl'))


import matplotlib.pyplot as plt

# Plotting training loss vs validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Save the model and scalers
newModelname = stockDataHandler.get_full_path('onehour_2018_2023_nbeats.h5')
model.save(newModelname)
joblib.dump(min_max_scalers, stockDataHandler.get_full_path('min_max_scalers_nbeats.pkl'))


# Teste modellen: 
# Prepare test data
test_data = stockDataHandler.percent_return(test_data, columns_to_percent_return)
test_data_min_max, _ = stockDataHandler.min_max_normalization(test_data, columns_to_min_max_normalize, min_max_scalers)
test_data[columns_to_min_max_normalize] = test_data_min_max[columns_to_min_max_normalize]

x_test = []
y_test = []

for i in range(numberOfBars, len(test_data) - output_horizon):
    x_test.append(test_data[columns_to_percent_return + columns_to_min_max_normalize].iloc[i - numberOfBars:i].values)
    y_test.append(test_data['Close'].iloc[i:i+output_horizon].values)

x_test = np.array(x_test)
y_test = np.array(y_test)

# Predicting using the trained model
predicted_values = model.predict(x_test)

# Convert percent predictions back to actual prices
last_known_price = test_data['Close'].iloc[numberOfBars - 1]
predicted_actual_prices = [last_known_price]

for pct_change in predicted_values:
    last_known_price += last_known_price * pct_change[0]  # Accessing the first element since pct_change might be a 2D array
    predicted_actual_prices.append(last_known_price)

predicted_actual_prices = np.array(predicted_actual_prices[1:])

# Plotting the actual vs predicted values with actual dates
dates = test_data.index[numberOfBars:len(test_data) - output_horizon].to_list()

plt.figure(figsize=(15, 6))
plt.plot(dates, y_test, color='blue', label='Actual Stock Price')
plt.plot(dates, predicted_actual_prices, color='red', linestyle='dashed', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-labels for better visibility
plt.tight_layout()  # Adjust layout for better visibility
plt.show()