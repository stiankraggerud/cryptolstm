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

import stockDataHandler

import plotly.offline as pyo  
import plotly.io as pio  
pio.renderers.default = "browser"  

import tensorflow.keras.backend as K  
  
def root_mean_squared_error(y_true, y_pred):  
    return K.sqrt(K.mean(K.square(y_pred - y_true)))  


def min_max_normalization(data, column_name):
    """
    Min-Max normalisering.
    """
    scaler = MinMaxScaler()
    normalized_column = scaler.fit_transform(data[[column_name]])
    return normalized_column

def rolling_window_normalization(data, column_name, window_size):  
    """  
    Rullende vindu normalisering.  
    """  
    rolling_mean = data[column_name].rolling(window=window_size).mean()  
    rolling_std = data[column_name].rolling(window=window_size).std()  
      
    normalized_column = (data[column_name] - rolling_mean) / rolling_std  
    return normalized_column, rolling_mean, rolling_std  

def rolling_window_denormalization(normalized_data, rolling_mean, rolling_std):  
    """  
    Reverserer rullende vindu normalisering (denormalisering).  
    """  
    denormalized_data = normalized_data * rolling_std + rolling_mean  
    return denormalized_data  

# Les inn data
test_data = stockDataHandler.LoadData('20august_2023-5.csv')

stockDataHandler.SetEMA(test_data, 50, 'EMA50')
stockDataHandler.SetEMA(test_data, 100, 'EMA100')
stockDataHandler.SetMacd(test_data, 50)

window_size = 20  # Valgfri vindustørrelse
# test_data['Close_normalized'] = rolling_window_normalization(test_data, 'Original_Close', window_size)
# test_data['Volume_normalized'] = rolling_window_normalization(test_data, 'Volume', window_size)

test_data['Close_normalized'], rolling_mean_close, rolling_std_close = rolling_window_normalization(test_data, 'Original_Close', window_size)  
test_data['Volume_normalized'], _, _ = rolling_window_normalization(test_data, 'Volume', window_size)  # Ignorer rolling_mean og rolling_std for 'Volume'  
test_data['Number_of_trades_normalized'], _, _ = rolling_window_normalization(test_data, 'Original_Number_of_trades', window_size)  # Ignorer rolling_mean og rolling_std for 'Volume'  
 

# test_data['Close_normalized'] = min_max_normalization(test_data, 'Original_Close')
# test_data['Volume_normalized'] = min_max_normalization(test_data, 'Volume')

# Min-Max normalisering
test_data['EMA50_normalized'] = min_max_normalization(test_data, 'EMA50')
test_data['EMA100_normalized'] = min_max_normalization(test_data, 'EMA100')
test_data['MACD_normalized'] = min_max_normalization(test_data, 'MACD')

# Fjern NaN-verdier som kan oppstå etter rullende vindu normalisering
test_data.dropna(inplace=True)


stockDataHandler.CleanData(test_data)

  
modelName = stockDataHandler.get_full_path('lstm_model_2018-06_2023_v2.h5')
# Last inn modellen  
 
# Last inn modellen med custom_object_scope  
from tensorflow.keras.utils import custom_object_scope  
  
with custom_object_scope({'root_mean_squared_error': root_mean_squared_error}):  
    model = load_model(modelName)  


# Anta at `test_data` har de samme kolonnene som dine treningsdata  
X_test = test_data[['Close_normalized','Number_of_trades_normalized', 'Volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']]  
  
# Forme data for LSTM (samples, timesteps, features)  
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))  
  
# 1. Hent de siste kjente verdiene og lag en ny DataFrame for å forutsi den neste ukjente verdien  
last_known_values = test_data[['Close_normalized','Number_of_trades_normalized', 'Volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']].iloc[-1]  
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

  
# Kombiner test_data og predicted_data  
combined_data = pd.concat([test_data, predicted_data])  

# Konverter 'Date'-kolonnen til riktig datoformat  


fig = go.Figure()  
fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Original_Close'], mode='lines', name='Faktiske priser'))  
fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Predicted_Close'], mode='lines', connectgaps=True, name='Forutsagte priser'))  
fig.add_trace(go.Scatter(x=[predicted_date], y=[predicted_close[0][0]], mode='markers', marker=dict(size=10, color='red'), name='Neste predikerte verdi'))  
  
fig.update_xaxes(type='category')  
  
# Eksporter plottet som en HTML-fil og åpne den i nettleseren  
plot_filename = "stock_plot_with_next_prediction.html"  
pyo.plot(fig, filename=plot_filename, auto_open=True)  
print("Test data:")  
print(test_data['Original_Close'])  
print("Predicted data:")  
print(predicted_data)  
print("Combined data:")  
print(combined_data[['Original_Close', 'Predicted_Close']])  
