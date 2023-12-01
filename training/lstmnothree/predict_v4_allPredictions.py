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
#test_data = stockDataHandler.LoadData('CSV\\onehour_2023-08-29-now.csv')
#test_data = stockDataHandler.LoadData('CSV\\test.csv')
test_data = stockDataHandler.LoadData('CSV/test.csv')

stockDataHandler.SetEMA(test_data, 50, 'EMA50')
stockDataHandler.SetEMA(test_data, 100, 'EMA100')
stockDataHandler.SetMacd(test_data, 50)

window_size = 20  # Valgfri vindustørrelse
# test_data['Close_normalized'] = rolling_window_normalization(test_data, 'Original_Close', window_size)
# test_data['Volume_normalized'] = rolling_window_normalization(test_data, 'Volume', window_size)

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

# Fjern NaN-verdier som kan oppstå etter rullende vindu normalisering
test_data.dropna(inplace=True)

print('antall rader:')
print(len(test_data))

stockDataHandler.CleanData(test_data)

#modelName = stockDataHandler.get_saved_models_path('lstm_model_2018-06_2023_train_v2_epoch-100.h5')  
modelName = stockDataHandler.get_saved_models_path('lstm_model_2017-06_2023_train_v4_epoch-300_400_400_50_1.h5')  

# Last inn modellen  
predicted_data = pd.DataFrame(columns=['Date', 'Predicted_Close'])  
# Last inn modellen med custom_object_scope  
from tensorflow.keras.utils import custom_object_scope  
  
with custom_object_scope({'root_mean_squared_error': root_mean_squared_error}):  
    model = load_model(modelName)  

test_data['Date'] = pd.to_datetime(test_data['Open time'])  
test_data.set_index('Date', inplace=True)  

for i in range(150, len(test_data)): 
    current_data = test_data.iloc[i:i+1]  
    current_input = np.reshape(current_data[['Close_normalized', 'Open_normalized', 'High_normalized', 'Low_normalized', 'Number_of_trades_normalized', 'Volume_normalized','Original_Taker_buy__base_asset_volume_normalized','Original_Taker_buy__quote_asset_volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']].values, (1, 1, 11))  
    print(current_input)
    
    
    predicted_close_normalized = model.predict(current_input)  

    # Denormalize den forutsagte verdien  
    predicted_close = rolling_window_denormalization(predicted_close_normalized, rolling_mean_close.iloc[i], rolling_std_close.iloc[i])  

    # Lagre den forutsagte prisen i den tomme DataFrame 
    predicted_date = test_data.index[i] + pd.Timedelta(hours=1)   
    # predicted_date = pd.Timestamp(test_data.index[i]) + pd.Timedelta(hours=1)  
    new_row = {'Date': predicted_date, 'Predicted_Close': predicted_close[0][0]}  
    predicted_data = pd.concat([predicted_data, pd.DataFrame([new_row])], ignore_index=True)  
    # predicted_data = predicted_data.append({'Date': predicted_date, 'Predicted_Close': predicted_close[0][0]}, ignore_index=True)    


# Sett Date som indeksen for predicted_data  
predicted_data['Date'] = pd.to_datetime(predicted_data['Date'])  
predicted_data.set_index('Date', inplace=True)  


# 4. Etter at alle prediksjonene er lagret, plotter du linjen med forutsagte priser sammen med candlestick-diagrammet  
combined_data = pd.concat([test_data, predicted_data])  

fig = go.Figure()  

fig.add_trace(go.Scatter(x=combined_data.index, y=combined_data['Predicted_Close'], mode='lines', connectgaps=True, name='Forutsagte priser'))  

fig.add_trace(go.Candlestick(x=combined_data.index,  
                open=combined_data['Original_Open'],  
                high=combined_data['Original_High'],  
                low=combined_data['Original_Low'],  
                close=combined_data['Original_Close'],  
                name='Candlestick'))  
  

print(combined_data) 
print("##############################################")
print(predicted_data)

# Eksporter plottet som en HTML-fil og åpne den i nettleseren  
plot_filename = f"stock_plot_with_all_predictions.html"  
pyo.plot(fig, filename=plot_filename, auto_open=True)  

# fig2 = go.Figure()  
# fig2.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data['Predicted_Close'], mode='lines', name='Predicted Prices'))  
# plot_filename2 = f"predicted_prices_plot.html"  
# pyo.plot(fig2, filename=plot_filename2, auto_open=True)  