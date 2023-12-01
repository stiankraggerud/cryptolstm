import os
import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.utils import custom_object_scope  
import stockDataHandler

import plotly.offline as pyo  
import plotly.io as pio  
pio.renderers.default = "browser"  
  
#modelName = 'lstm_model_2017-06_2023_train_v3_epoch-010_400_400_50_1.h5'  
modelName = 'lstm_model_2018-06_2023_train_v2_epoch-100.h5'  

def min_max_normalization(data, column_name):
    """
    Min-Max normalisering.
    """
    scaler = MinMaxScaler()
    normalized_column = scaler.fit_transform(data[[column_name]])
    return normalized_column

def root_mean_squared_error(y_true, y_pred):  
    return K.sqrt(K.mean(K.square(y_pred - y_true)))  
  
def rolling_window_normalization(data, column_name, window_size):  
    rolling_mean = data[column_name].rolling(window=window_size).mean()  
    rolling_std = data[column_name].rolling(window=window_size).std()  
    normalized_column = (data[column_name] - rolling_mean) / rolling_std  
    return normalized_column, rolling_mean, rolling_std  
  
def rolling_window_denormalization(normalized_data, rolling_mean, rolling_std):  
    denormalized_data = normalized_data * rolling_std + rolling_mean  
    return denormalized_data  
  
# Les inn data  
test_data = stockDataHandler.LoadData('onehour_2023-08-25-now.csv')  
  
stockDataHandler.SetEMA(test_data, 50, 'EMA50')  
stockDataHandler.SetEMA(test_data, 100, 'EMA100')  
stockDataHandler.SetMacd(test_data, 50)  
  
window_size = 20  
test_data['Close_normalized'], rolling_mean_close, rolling_std_close = rolling_window_normalization(test_data, 'Original_Close', window_size)  
test_data['Open_normalized'], rolling_mean_open, rolling_std_open = rolling_window_normalization(test_data, 'Original_Open', window_size)  
test_data['High_normalized'], rolling_mean_high, rolling_std_high = rolling_window_normalization(test_data, 'Original_High', window_size)  
test_data['Low_normalized'], rolling_mean_low, rolling_std_low = rolling_window_normalization(test_data, 'Original_Low', window_size)  
test_data['Volume_normalized'], _, _ = rolling_window_normalization(test_data, 'Volume', window_size)  
test_data['Number_of_trades_normalized'], _, _ = rolling_window_normalization(test_data, 'Original_Number_of_trades', window_size)  
test_data['Original_Taker_buy__base_asset_volume_normalized'], _, _ = rolling_window_normalization(test_data, 'Original_Taker_buy__base_asset_volume', window_size)  
test_data['Original_Taker_buy__quote_asset_volume_normalized'], _, _ = rolling_window_normalization(test_data, 'Original_Taker_buy__quote_asset_volume', window_size)  
  
test_data['EMA50_normalized'] = min_max_normalization(test_data, 'EMA50')  
test_data['EMA100_normalized'] = min_max_normalization(test_data, 'EMA100')  
test_data['MACD_normalized'] = min_max_normalization(test_data, 'MACD')  
  
test_data.dropna(inplace=True)  
stockDataHandler.CleanData(test_data)  
  
filename = stockDataHandler.get_filename_without_extension(__file__)  
  
modelName = stockDataHandler.get_saved_models_path(modelName)  
with custom_object_scope({'root_mean_squared_error': root_mean_squared_error}):  
    model = load_model(modelName)  
  
X_test = test_data[['Close_normalized', 'Open_normalized', 'High_normalized', 'Low_normalized', 'Number_of_trades_normalized', 'Volume_normalized','Original_Taker_buy__base_asset_volume_normalized','Original_Taker_buy__quote_asset_volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']]  
# X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape.shape[1]))  
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))  
  
combined_data_df = pd.DataFrame(columns=['Date', 'Original_Close', 'Predicted_Close'])  
  
rows_list = []  
for index, row in test_data.iterrows():  
    current_input_values = row[['Close_normalized','Open_normalized', 'High_normalized','Low_normalized', 'Number_of_trades_normalized', 'Volume_normalized','Original_Taker_buy__base_asset_volume_normalized','Original_Taker_buy__quote_asset_volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']].values.astype(np.float32)
    current_input = np.reshape(current_input_values, (1, 1, 11))  
    predicted_close_normalized = model.predict(current_input)  
    predicted_close = rolling_window_denormalization(predicted_close_normalized, rolling_mean_close.loc[index], rolling_std_close.loc[index])  
    new_row = {'Date': index, 'Original_Close': row['Original_Close'], 'Predicted_Close': predicted_close[0][0]}  
    rows_list.append(new_row)  
  
combined_data_df = pd.DataFrame(rows_list)  
combined_data_df.set_index('Date', inplace=True)  


# 1. Hent de siste kjente verdiene og lag en ny DataFrame for å forutsi den neste ukjente verdien  
last_known_values = test_data[['Close_normalized', 'Open_normalized', 'High_normalized', 'Low_normalized', 'Number_of_trades_normalized', 'Volume_normalized','Original_Taker_buy__base_asset_volume_normalized','Original_Taker_buy__quote_asset_volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']].iloc[-1]  
next_value_df = pd.DataFrame([last_known_values.values], columns=last_known_values.index)  
  
# 2. Bruk modellen til å forutsi den neste ukjente verdien  
next_value_input = np.reshape(next_value_df.values, (next_value_df.shape[0], 1, next_value_df.shape[1]))  


predicted_close_normalized = model.predict(next_value_input)  
  
# 3. Denormalize den forutsagte verdien  
predicted_close = rolling_window_denormalization(predicted_close_normalized, rolling_mean_close.iloc[-1], rolling_std_close.iloc[-1])  
  
# Legg til den forutsagte verdien til den opprinnelige grafen og vis den  
# Konverter indeksen til en DatetimeIndex  
# test_data.index = pd.to_datetime(test_data.index)  
test_data['Date'] = pd.to_datetime(test_data['Open time'])  
test_data.set_index('Date', inplace=True)  
  
# Beregn den forutsagte datoen ved å legge til en dag til den siste kjente datoen  
predicted_date = test_data.index[-1] + pd.Timedelta(days=1)  

predicted_date = test_data.index[-1] + pd.Timedelta(days=1)  
predicted_data = pd.DataFrame({'Date': [predicted_date], 'Predicted_Close': [predicted_close[0][0]]})  
predicted_data.set_index('Date', inplace=True)  
  
combined_data = pd.concat([test_data, predicted_data])  



# last_known_values = test_data[['Close_normalized','Open_normalized', 'High_normalized','Low_normalized', 'Number_of_trades_normalized', 'Volume_normalized','Original_Taker_buy__base_asset_volume_normalized','Original_Taker_buy__quote_asset_volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']].iloc[-1]  
# current_input = np.reshape(last_known_values.values.astype(np.float32), (1, 1, 11))  
# next_predicted_close_normalized = model.predict(current_input)  
  
# next_predicted_close = rolling_window_denormalization(next_predicted_close_normalized, rolling_mean_close.iloc[-1], rolling_std_close.iloc[-1])  


# next_date = pd.Timestamp(test_data.index[-1]) + pd.Timedelta(hours=1)  
# next_row = pd.Series({'Date': next_date, 'Original_Close': np.nan, 'Predicted_Close': next_predicted_close[0][0]})  
# next_row_df = pd.DataFrame([next_row]).set_index('Date')  


  
fig = go.Figure()  
fig.add_trace(go.Scatter(x=combined_data_df.index, y=combined_data_df['Original_Close'], mode='lines', name='Faktiske priser'))  
fig.add_trace(go.Scatter(x=combined_data_df.index, y=combined_data_df['Predicted_Close'], mode='lines', name='Forutsagte priser'))  
fig.add_trace(go.Scatter(x=[predicted_date], y=[predicted_close[0][0]], mode='markers', marker=dict(size=10, color='red'), name='Neste predikerte verdi'))  

fig.update_xaxes(type='category')  
  
#fig.show()
plot_filename = filename + ".html"  
pyo.plot(fig, filename=plot_filename, auto_open=True)  

