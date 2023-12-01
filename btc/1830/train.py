import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Last ned data
stock_data = pd.read_csv('./btc/1830/test.csv')
stock_data.set_index('Open Time', inplace=True)
modelname = '1830.h5'
#Sortert alfabetisk automatisk. (0 = buy, 1 = hold, 2 = sell)
class_weights = {0: 1., 1: 0.1, 2: 1.}

numberOfBars = 180

stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')
stock_data = stock_data.dropna()

# Beregne EMA
stock_data['EMA50'] = stock_data['Close'].ewm(span=50).mean()
stock_data['EMA100'] = stock_data['Close'].ewm(span=100).mean()

# Generere etiketter for klassifisering
stock_data['Signal'] = 'Hold'
for i in range(len(stock_data)-1, numberOfBars, -1):
    print(i)
    for j in range(i-numberOfBars, i):
        price_change = (stock_data.iloc[j]['Close'] - stock_data.iloc[i]['Close']) / stock_data.iloc[i]['Close']
        if price_change > 0.006:
            stock_data.loc[stock_data.index[i], 'Signal'] = 'Buy'
            break
        elif price_change < -0.006:
            stock_data.loc[stock_data.index[i], 'Signal'] = 'Sell'
            break

# Fjern de første radene hvor vi ikke har tilstrekkelig data for beregning
stock_data = stock_data.iloc[numberOfBars:]

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



# # LSTM model
# model = keras.Sequential()
# model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 4)))
# model.add(layers.LSTM(100, return_sequences=False))
# model.add(layers.Dense(50))
# model.add(layers.Dense(3, activation='softmax'))  # Vi har 3 klasser: 'Buy', 'Sell', 'Hold'
# model.summary()

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Check the classes labels
# classes = encoder.classes_
# print(classes)

# # Create a dictionary to set the class weights. Sortert alfabetisk automatisk. (0 = buy, 1 = hold, 2 = sell)

# model.fit(x_train, y_train, batch_size=1, epochs=3, class_weight=class_weights)
# model.save(modelname)

# # Lagrer scaler og encoder til disk
# import joblib
# joblib.dump(scaler_price, 'scaler_price.pkl')
# joblib.dump(scaler_volume, 'scaler_volume.pkl')
# joblib.dump(scaler_ema50, 'scaler_ema50.pkl')
# joblib.dump(scaler_ema100, 'scaler_ema100.pkl')
# joblib.dump(encoder, 'encoder.pkl')
