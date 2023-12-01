import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go

# Last ned data
stock_data = pd.read_csv('./maijuni.csv')
stock_data.set_index('Open Time', inplace=True)
modelname = '1707_1030.h5'

class_weights = {0: 100., 1: 100, 2: 100.}

numberOfBars = 180

stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')
stock_data = stock_data.dropna()

# Beregne EMA
stock_data['EMA50'] = stock_data['Close'].ewm(span=50).mean()
stock_data['EMA100'] = stock_data['Close'].ewm(span=100).mean()

# Generere etiketter for klassifisering basert på toppene og dalene
min_prominence = 130

# Finner indeksene til toppene
peak_indexes, _ = find_peaks(stock_data['Close'].values, prominence=min_prominence)

# Finner indeksene til dalene ved å multiplisere dataene med -1
valley_indexes, _ = find_peaks(-stock_data['Close'].values, prominence=min_prominence)

# Setter signalene for toppene til 'Sell' og for dalene til 'Buy'
stock_data.loc[stock_data.index[peak_indexes], 'Signal'] = 'Sell'
stock_data.loc[stock_data.index[valley_indexes], 'Signal'] = 'Buy'

# Fyller resten med 'Hold'
stock_data['Signal'].fillna('Hold', inplace=True)

# Remove tags if the price change is less than 0.7%
price_change_threshold = 0.007

# Get the indexes of the buys and sells
buy_indexes = stock_data[stock_data['Signal'] == 'Buy'].index
sell_indexes = stock_data[stock_data['Signal'] == 'Sell'].index

for buy_index in buy_indexes:
    # Find the next sell signal after this buy signal
    next_sell_indexes = sell_indexes[sell_indexes > buy_index]
    if next_sell_indexes.empty:
        continue
    next_sell_index = next_sell_indexes[0]
    
    # Calculate the price change
    buy_price = stock_data.loc[buy_index, 'Close']
    sell_price = stock_data.loc[next_sell_index, 'Close']
    price_change = (sell_price - buy_price) / buy_price
    
    # If the price change is less than the threshold, remove the tags
    # if(price_change.empty()):
    #     print('empty')
    if (abs(price_change) < price_change_threshold).all():
        stock_data.loc[buy_index, 'Signal'] = 'Hold'
        stock_data.loc[next_sell_index, 'Signal'] = 'Hold'

# Plott toppene og dalene
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=stock_data.index[peak_indexes], y=stock_data['Close'].values[peak_indexes], mode='markers', name='Sell - Peaks'))
fig.add_trace(go.Scatter(x=stock_data.index[valley_indexes], y=stock_data['Close'].values[valley_indexes], mode='markers', name='Buy - Valleys'))
fig.show()

# Resten av koden forblir den samme


# Splitt data i trening og test sett
training_data_len = math.ceil(len(stock_data)* 0.8)
train_data = stock_data[:training_data_len]

# Resten av koden forblir den samme

# Splitt data i trening og test sett
training_data_len = math.ceil(len(stock_data)* 0.8)
train_data = stock_data[:training_data_len]

# Skaler data
scaler_price = MinMaxScaler(feature_range=(0,1))
scaled_price = scaler_price.fit_transform(train_data['Close'].values.reshape(-1,1))

scaler_volume = MinMaxScaler(feature_range=(0,1))
scaled_volume = scaler_volume.fit_transform(train_data['Volume'].values.reshape(-1,1))

scaler_ema50 = MinMaxScaler(feature_range=(0,1))
scaled_ema50 = scaler_ema50.fit_transform(train_data['EMA50'].values.reshape(-1,1))

scaler_ema100 = MinMaxScaler(feature_range=(0,1))
scaled_ema100 = scaler_ema100.fit_transform(train_data['EMA100'].values.reshape(-1,1))

# Forberede data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    print(str(i) + ' of ' + str(len(train_data)))
    x_train.append(np.column_stack((scaled_price[i-numberOfBars:i, 0], scaled_volume[i-numberOfBars:i, 0], scaled_ema50[i-numberOfBars:i, 0], scaled_ema100[i-numberOfBars:i, 0])))
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)

# One-hot encoding av målvariabel
encoder = LabelEncoder()

# Telle antallet av hver kategori
category_counts = train_data['Signal'].value_counts()
# Skriv ut antallet
print(category_counts)

print(np.unique(train_data['Signal']))  # før encoding
y_train = encoder.fit_transform(y_train)
y_train = keras.utils.to_categorical(y_train)
print(y_train.shape[1])  # etter encoding



# LSTM model
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 4)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(3, activation='softmax'))  # Vi har 3 klasser: 'Buy', 'Sell', 'Hold'
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Check the classes labels
classes = encoder.classes_
print(classes)

# Create a dictionary to set the class weights. Sortert alfabetisk automatisk. (0 = buy, 1 = hold, 2 = sell)

model.fit(x_train, y_train, batch_size=1, epochs=3, class_weight=class_weights)
model.save(modelname)

# Lagrer scaler og encoder til disk
import joblib
joblib.dump(scaler_price, 'scaler_price.pkl')
joblib.dump(scaler_volume, 'scaler_volume.pkl')
joblib.dump(scaler_ema50, 'scaler_ema50.pkl')
joblib.dump(scaler_ema100, 'scaler_ema100.pkl')
joblib.dump(encoder, 'encoder.pkl')
