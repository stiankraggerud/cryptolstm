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
def generate_target(df, column_name, steps_ahead=1):
    """
    Genererer en 'Target'-kolonne basert på fremtidig pris.
    Prisen 'steps_ahead' punkter frem i tid vil være målverdien.
    """
    df['Target'] = df[column_name].shift(-steps_ahead)
    df.dropna(inplace=True)  # Fjerner NaN-verdier som kan oppstå på grunn av tidsforskyvningen
    return df

def create_sequences(X, y, sequence_length):  
    X_sequences = []  
    y_sequences = []  
  
    for i in range(len(X) - sequence_length):  
        X_sequences.append(X[i:i + sequence_length])  
        y_sequences.append(y[i + sequence_length])  
  
    X_sequences = np.array(X_sequences)  
    y_sequences = np.array(y_sequences)  
  
    return X_sequences, y_sequences  

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
  
stock_data['EMA50_normalized'] = min_max_normalization(stock_data, 'EMA50')  
stock_data['EMA100_normalized'] = min_max_normalization(stock_data, 'EMA100')  
stock_data['MACD_normalized'] = min_max_normalization(stock_data, 'MACD')  
  
stock_data.dropna(inplace=True)  
  
stockDataHandler.CleanData(stock_data)  
  
stock_data = generate_target(stock_data, 'Close_normalized',1)  
  
X = stock_data[['Close_normalized','Open_normalized','High_normalized', 'Low_normalized', 'Number_of_trades_normalized', 'Volume_normalized','Original_Taker_buy__base_asset_volume_normalized','Original_Taker_buy__quote_asset_volume_normalized', 'EMA50_normalized', 'EMA100_normalized', 'MACD_normalized']]  
y = stock_data['Target']  
  
sequence_length = 50 
X_sequences, y_sequences = create_sequences(X.values, y.values, sequence_length)  
X_sequences = np.reshape(X_sequences, (X_sequences.shape[0], sequence_length, X_sequences.shape[2]))  
  
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.3, random_state=42)  
  
model = Sequential()  
model.add(LSTM(400, return_sequences=True, input_shape=(X_train.shape[1], 11)))  
model.add(LSTM(400, return_sequences=False))  
model.add(Dense(50))  
model.add(Dense(1))  
model.summary()  
  
model.compile(optimizer='adam', loss='mean_squared_error')  
  
# Sett antall epoker og intervallet for lagring og beregning av R2 score  
total_epochs = 400  
save_every_n_epochs = 10  

num_iterations = total_epochs // save_every_n_epochs  
models_directory = 'saved_models_train_v5'  
if not os.path.exists(models_directory):  
    os.makedirs(models_directory)  

from datetime import datetime  
  
# Få nåværende dato og tid  
now = datetime.now()  
  
# Formatere til en streng  
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")  
  
# Inkluder i filnavnet  
filename = f'r2_scores_{dt_string}.txt'  
  
with open(os.path.join(models_directory, filename), 'w') as r2_scores_file:    
    for i, epoch in enumerate(range(0, total_epochs, save_every_n_epochs)):  
        print(f'Starting iteration {i+1} of {num_iterations}...')       
        model.fit(X_train, y_train, epochs=save_every_n_epochs, batch_size=8, verbose=1)  
            
        # Lagre modellen  
        model_path = os.path.join(models_directory, f'lstm_modelnothree_train_v4_epoch60_seq50-{epoch+save_every_n_epochs:03d}_400_400_50_1.h5')  
        model.save(model_path)  
            
        # Vurdere modellen på testdata  
        loss = model.evaluate(X_test, y_test)  
        print(f'Test loss for epoch {epoch+save_every_n_epochs}: {loss}')  
            
        # Beregne og lagre R2 Score for denne modellen  
        predictions = model.predict(X_test)  
        r2 = r2_score(y_test, predictions)  
        print(f'R2 Score for epoch {epoch+save_every_n_epochs}: {r2}')  
            
        # Lagre R2-score til tekstfil  
        r2_scores_file.write(f'Test loss for epoch {epoch+save_every_n_epochs}: {loss}')  
        r2_scores_file.write(f'Epoch {epoch+save_every_n_epochs}: R2 Score = {r2}\n')  
        r2_scores_file.flush() 