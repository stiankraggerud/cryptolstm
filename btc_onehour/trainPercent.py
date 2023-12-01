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


def get_full_path(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, filename)


# Last ned data
stock_data = pd.read_csv(get_full_path('onehour2022_2003.csv'))
stock_data.set_index('Open time', inplace=True)
modelname = get_full_path('btc_onehour.h5')
numberOfBars = 500

stock_data['Original_Close'] = stock_data['Close'].copy()
# Beregne prosentvis endring
stock_data['Close'] = stock_data['Close'].pct_change() * 100
stock_data['Open'] = stock_data['Open'].pct_change() * 100
stock_data['High'] = stock_data['High'].pct_change() * 100
stock_data['Low'] = stock_data['Low'].pct_change() * 100
stock_data['Volume'] = stock_data['Volume'].pct_change() * 100

# Fjern uendelige og NaN-verdier
stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
stock_data = stock_data.dropna()

# Beregne EMA
stock_data['EMA50'] = stock_data['Close'].ewm(span=50).mean()
stock_data['EMA100'] = stock_data['Close'].ewm(span=100).mean()

# Beregne MACD
macd_line = stock_data['Close'].ewm(span=50).mean() - stock_data['Close'].ewm(span=100).mean()
signal_line = macd_line.ewm(span=9).mean()
stock_data['MACD'] = macd_line
stock_data['Signal_Line'] = signal_line

# Generere etiketter for klassifisering basert på toppene og dalene
min_prominence = 200
peak_indexes, _ = find_peaks(stock_data['Original_Close'].values, prominence=min_prominence) # Topper for salg
valley_indexes, _ = find_peaks(-stock_data['Original_Close'].values, prominence=min_prominence) # Bunner for kjøp

offset = 3


#sell_indexes = peak_indexes - offset
sell_indexes = peak_indexes
sell_indexes = sell_indexes[sell_indexes >= 0]
#buy_indexes = valley_indexes - offset
buy_indexes = valley_indexes

buy_indexes = buy_indexes[buy_indexes >= 0]

# Merk at vi nå bruker 'Sell' for sell_indexes og 'Buy' for buy_indexes
stock_data.loc[stock_data.index[sell_indexes], 'Signal'] = 'Sell'
stock_data.loc[stock_data.index[buy_indexes], 'Signal'] = 'Buy'
stock_data['Signal'].fillna('Hold', inplace=True)

# Splitt data i trening og test sett
training_data_len = math.ceil(len(stock_data) * 0.8)
train_data = stock_data[:training_data_len]

training_data_len = math.ceil(len(stock_data) * 0.8)
train_data = stock_data[:training_data_len]

# Skriv ut antall 'Buy', 'Sell', og 'Hold' signaler i treningsdataene
signal_counts = train_data['Signal'].value_counts()
print("Number of 'Buy', 'Sell', and 'Hold' signals in training data:")
print(signal_counts)

buy_signals = stock_data[stock_data['Signal'] == 'Buy']['Original_Close']
sell_signals = stock_data[stock_data['Signal'] == 'Sell']['Original_Close']

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Original_Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))


fig.update_layout(title='Stock Price with Buy and Sell Signals', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')
#fig.show()

#assert False

# Skaler data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_data[['Open', 'Close', 'High', 'Low', 'Volume', 'EMA50', 'EMA100', 'MACD']])

# Forberede data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(scaled_data[i - numberOfBars:i])
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

# Oversampling før one-hot encoding
ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1),
                                                                y_train_encoded)
x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)

# LSTM model
model = keras.Sequential()
model.add(layers.LSTM(400, return_sequences=True, input_shape=(x_train_resampled.shape[1], 8)))  # 8 features
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(y_train_resampled.shape[1], activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_resampled, y_train_resampled, batch_size=1, epochs=2)
model.save(modelname)

# Lagrer scaler og encoder til disk
joblib.dump(scaler, get_full_path('scaler.pkl'))
joblib.dump(encoder, get_full_path('encoder.pkl'))
