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

##Denne versjonen predikerer en time fremover i tid.
modelName = 'lstm_model_2017-06_2023_train_v3_epoch-100_400_400_50_1.h5'

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

test_data['Close_normalized'], rolling_mean_close, rolling_std_close = rolling_window_normalization(test_data, 'Original_Close', window_size)  
test_data['Open_normalized'], rolling_mean_open, rolling_std_open = rolling_window_normalization(test_data, 'Original_Open', window_size)  
test_data['High_normalized'], rolling_mean_high, rolling_std_high = rolling_window_normalization(test_data, 'Original_High', window_size)  
test_data['Low_normalized'], rolling_mean_low, rolling_std_low = rolling_window_normalization(test_data, 'Original_Low', window_size)  
test_data['Volume_normalized'], _, _ = rolling_window_normalization(test_data, 'Volume', window_size)  # Ignorer rolling_mean og rolling_std for 'Volume'  
test_data['Number_of_trades_normalized'], _, _ = rolling_window_normalization(test_data, 'Original_Number_of_trades', window_size)  # Ignorer rolling_mean og rolling_std for 'Volume'  
test_data['Original_Taker_buy__base_asset_volume_normalized'], _,_ = rolling_window_normalization(test_data, 'Original_Taker_buy__base_asset_volume', window_size)
test_data['Original_Taker_buy__quote_asset_volume_normalized'], _,_ = rolling_window_normalization(test_data, 'Original_Taker_buy__quote_asset_volume', window_size)


# Min-Max normalisering
test_data['EMA50_normalized'] = min_max_normalization(test_data, 'EMA50')
test_data['EMA100_normalized'] = min_max_normalization(test_data, 'EMA100')
test_data['MACD_normalized'] = min_max_normalization(test_data, 'MACD')

# Fjern NaN-verdier som kan oppstå etter rullende vindu normalisering
test_data.dropna(inplace=True)
stockDataHandler.CleanData(test_data)

filename = stockDataHandler.get_filename_without_extension(__file__)


  
modelName = stockDataHandler.get_saved_models_path(modelName)
# Last inn modellen  
 
# Last inn modellen med custom_object_scope  
from tensorflow.keras.utils import custom_object_scope  
  
with custom_object_scope({'root_mean_squared_error': root_mean_squared_error}):  
    model = load_model(modelName)  


# Anta at `test_data` har de samme kolonnene som dine treningsdata  
X_test = test_data[['Close_normalized','Open_normalized','High_normalized','Low_normalized','Number_of_trades_normalized', 'Volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']]  
  
# Forme data for LSTM (samples, timesteps, features)  
X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))  
  
# 1. Hent de siste kjente verdiene og lag en ny DataFrame for å forutsi den neste ukjente verdien  
last_known_values = test_data[['Close_normalized','Open_normalized', 'High_normalized','Low_normalized', 'Number_of_trades_normalized', 'Volume_normalized','Original_Taker_buy__base_asset_volume_normalized','Original_Taker_buy__quote_asset_volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']].iloc[-3]  

# Tom DataFrame for å lagre faktiske og predikerte verdier  
combined_data_df = pd.DataFrame(columns=['Date', 'Original_Close', 'Predicted_Close'])  

rows_list = [] 
predicted_closes = []  


rows_list = []   
# Iterer gjennom hver rad i test_data og utfør prediksjon    
for index, row in test_data.iterrows():        
    # 1. Bruk modellen til å forutsi den neste ukjente verdien        
    # Konverter inputverdiene til en NumPy-array med dtype np.float32        
    current_input_values = row[['Close_normalized','Open_normalized', 'High_normalized','Low_normalized', 'Number_of_trades_normalized', 'Volume_normalized','Original_Taker_buy__base_asset_volume_normalized','Original_Taker_buy__quote_asset_volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']].values.astype(np.float32)        
    current_input = np.reshape(current_input_values, (1, 1, 11))        
    predicted_close_normalized = model.predict(current_input)        
          
    # 2. Denormalize den forutsagte verdien      
    # Bruk .loc[index] i stedet for .iloc[-1] for å få riktig rullende gjennomsnitt og standardavvik      
    predicted_close = rolling_window_denormalization(predicted_close_normalized, rolling_mean_close.loc[index], rolling_std_close.loc[index])        
          
    # 3. Legg til faktiske og predikerte verdier i combined_data DataFrame        
    new_row = pd.Series({'Date': index, 'Original_Close': row['Original_Close'], 'Predicted_Close': predicted_close[0][0]})    
    print(type(combined_data_df))  
      # 3. Legg til faktiske og predikerte verdier i combined_data DataFrame        
    new_row = {'Date': index, 'Original_Close': row['Original_Close'], 'Predicted_Close': predicted_close[0][0]}    
    rows_list.append(new_row)    

combined_data_df = pd.DataFrame(rows_list) 
# Sett indeksen til 'Date' kolonnen  
combined_data_df.set_index('Date', inplace=True)  

fig = go.Figure()  
fig.add_trace(go.Scatter(x=combined_data_df.index, y=combined_data_df['Original_Close'], mode='lines', name='Faktiske priser'))  
fig.add_trace(go.Scatter(x=combined_data_df.index, y=combined_data_df['Predicted_Close'], mode='lines', name='Forutsagte priser'))  
  
fig.update_xaxes(type='category')  
  
# Eksporter plottet som en HTML-fil og åpne den i nettleseren  
plot_filename = filename + ".html"  
pyo.plot(fig, filename=plot_filename, auto_open=True)  
print("Combined data:")  
print(combined_data_df[['Original_Close', 'Predicted_Close']])  

# Sjekk om combined_data inneholder data  
print("Combined data:")  
print(combined_data_df[['Original_Close', 'Predicted_Close']])  
  
# Hvis combined_data inneholder data, prøv å plotte hver tidsserie separat  
if not combined_data_df.empty:  
    fig = go.Figure()  
    fig.add_trace(go.Scatter(x=combined_data_df.index, y=combined_data_df['Original_Close'], mode='lines', name='Faktiske priser'))  
    fig.add_trace(go.Scatter(x=combined_data_df.index, y=combined_data_df['Predicted_Close'], mode='lines', name='Forutsagte priser'))  
  
    fig.update_xaxes(type='category')  
  
    pyo.plot(fig, filename=plot_filename, auto_open=True)  
else:  
    print("Ingen data å vise i grafen.")  