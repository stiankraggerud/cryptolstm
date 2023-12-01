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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow.keras.backend as K 
from tensorflow.keras.callbacks import ModelCheckpoint

import stockDataHandler


def rolling_window_normalization(data, column_name, window_size):
    """
    Rullende vindu normalisering.
    """
    rolling_mean = data[column_name].rolling(window=window_size).mean()
    rolling_std = data[column_name].rolling(window=window_size).std()
    
    normalized_column = (data[column_name] - rolling_mean) / rolling_std
    return normalized_column

def min_max_normalization(data, column_name):
    """
    Min-Max normalisering.
    """
    scaler = MinMaxScaler()
    normalized_column = scaler.fit_transform(data[[column_name]])
    return normalized_column


# Les inn data
stock_data = stockDataHandler.LoadData('onehour_2017-06-2023.csv')

stockDataHandler.SetEMA(stock_data, 50, 'EMA50')
stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
stockDataHandler.SetMacd(stock_data, 50)

window_size = 20  # Valgfri vindustørrelse
stock_data['Close_normalized'] = rolling_window_normalization(stock_data, 'Original_Close', window_size)
stock_data['Open_normalized'] = rolling_window_normalization(stock_data, 'Original_Open', window_size)
stock_data['High_normalized'] = rolling_window_normalization(stock_data, 'Original_High', window_size)
stock_data['Low_normalized'] = rolling_window_normalization(stock_data, 'Original_Low', window_size)

stock_data['Original_Taker_buy__base_asset_volume_normalized'] = rolling_window_normalization(stock_data, 'Original_Taker_buy__base_asset_volume', window_size)
stock_data['Original_Taker_buy__quote_asset_volume_normalized'] = rolling_window_normalization(stock_data, 'Original_Taker_buy__quote_asset_volume', window_size)

stock_data['Volume_normalized'] = rolling_window_normalization(stock_data, 'Volume', window_size)
stock_data['Number_of_trades_normalized'] = rolling_window_normalization(stock_data, 'Original_Number_of_trades', window_size) 

# stock_data['Close_normalized'] = min_max_normalization(stock_data, 'Original_Close')
# stock_data['Volume_normalized'] = min_max_normalization(stock_data, 'Volume')

# Min-Max normalisering
stock_data['EMA50_normalized'] = min_max_normalization(stock_data, 'EMA50')
stock_data['EMA100_normalized'] = min_max_normalization(stock_data, 'EMA100')
stock_data['MACD_normalized'] = min_max_normalization(stock_data, 'MACD')

# Fjern NaN-verdier som kan oppstå etter rullende vindu normalisering
stock_data.dropna(inplace=True)


stockDataHandler.CleanData(stock_data)



def generate_target(df, column_name, steps_ahead=1):
    """
    Genererer en 'Target'-kolonne basert på fremtidig pris.
    Prisen 'steps_ahead' punkter frem i tid vil være målverdien.
    """
    df['Target'] = df[column_name].shift(-steps_ahead)
    df.dropna(inplace=True)  # Fjerner NaN-verdier som kan oppstå på grunn av tidsforskyvningen
    return df

# Bruk funksjonen for å generere 'Target'-kolonnen basert på 'Original_Close'-kolonnen
stock_data = generate_target(stock_data, 'Close_normalized',1)

def create_sequences(X, y, sequence_length):  
    X_sequences = []  
    y_sequences = []  
  
    for i in range(len(X) - sequence_length):  
        X_sequences.append(X[i:i + sequence_length])  
        y_sequences.append(y[i + sequence_length])  
  
    X_sequences = np.array(X_sequences)  
    y_sequences = np.array(y_sequences)  
  
    return X_sequences, y_sequences  

# Prepare data for sequences  
X = stock_data[['Close_normalized','Open_normalized','High_normalized', 'Low_normalized', 'Number_of_trades_normalized', 'Volume_normalized','Original_Taker_buy__base_asset_volume_normalized','Original_Taker_buy__quote_asset_volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']]  
y = stock_data['Target']  
  
sequence_length = 1  # Antall tidssteg du ønsker å bruke som input  
  
# Create sequences before reshaping and splitting  
X_sequences, y_sequences = stockDataHandler.create_sequences(X.values, y.values, sequence_length)  
  
# Forme data for LSTM (samples, timesteps, features)  
X_sequences = np.reshape(X_sequences, (X_sequences.shape[0], sequence_length, X_sequences.shape[2]))  
  
# Splitte data i trening og testsett  
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.3, random_state=42)  



model = keras.Sequential()
model.add(layers.LSTM(400, return_sequences=True, input_shape=(X_train.shape[1], 11))) 
model.add(layers.LSTM(400, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(1))
# model.add(layers.Dense(y.shape[1], activation='softmax'))
model.summary()

def root_mean_squared_error(y_true, y_pred):  
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])  # endre til loss='categorical_crossentropy' for klassifisering
model.compile(optimizer='adam', loss='mean_squared_error', metrics=[root_mean_squared_error])  # endre til loss='categorical_crossentropy' for klassifisering


# Trene modellen
#model.fit(X_train, y_train, epochs=1000, batch_size=8)

# Definer antall epoker for lagring av modellen  
save_every_n_epochs = 20  # Endre dette tallet etter ønske  
  
# Opprett en katalog for å lagre modellene  
models_directory = 'saved_models_train_v4'  
if not os.path.exists(models_directory):  
    os.makedirs(models_directory)  
  
# Definer filnavnformatet for å reflektere antall epoker modellen er trent på  
filename_format = os.path.join(models_directory, 'lstm_model_2017-06_2023_train_v4_epoch-{epoch:03d}_400_400_50_1.h5')  
  
# Bruk ModelCheckpoint for å lagre modellen for hver n'te epoke  
checkpoint = ModelCheckpoint(filepath=filename_format, save_freq=save_every_n_epochs * (len(y_train) // 8))  
  
# Trene modellen med ModelCheckpoint som en callback  
model.fit(X_train, y_train, epochs=400, batch_size=8, callbacks=[checkpoint])  



# Lagre modellen
# modelName = stockDataHandler.get_full_path('lstm_model_2018-06_2023_train_v2.h5')
# model.save(modelName)

print('#############################')
# Vurdere modellen på testdata
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f'R2 Score: {r2}')

