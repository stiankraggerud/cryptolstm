import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objects as go
import joblib

# Last ned data
stock_data = pd.read_csv('./juli.csv')
stock_data.set_index('Open Time', inplace=True)
modelname = '1907_1100.h5'
numberOfBars = 180

stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')
stock_data = stock_data.dropna()

# Beregne EMA
stock_data['EMA50'] = stock_data['Close'].ewm(span=50).mean()
stock_data['EMA100'] = stock_data['Close'].ewm(span=100).mean()

# Beregn RSI
def compute_RSI(data, window):
    delta = data.diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.rolling(window=window).mean()
    roll_down1 = down.abs().rolling(window=window).mean()
    RS = roll_up1 / roll_down1
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI
print(stock_data.index[stock_data.index.duplicated()])
stock_data = stock_data[~stock_data.index.duplicated(keep='first')]
stock_data['RSI'] = compute_RSI(stock_data['Close'], 60)
stock_data['RSI'].fillna(0, inplace=True)

# Generere etiketter for klassifisering basert på toppene og dalene
min_prominence = 120
peak_indexes, _ = find_peaks(stock_data['Close'].values, prominence=min_prominence)
valley_indexes, _ = find_peaks(-stock_data['Close'].values, prominence=min_prominence)
stock_data.loc[stock_data.index[peak_indexes], 'Signal'] = 'Sell'
stock_data.loc[stock_data.index[valley_indexes], 'Signal'] = 'Buy'
stock_data['Signal'].fillna('Hold', inplace=True)

# Splitt data i trening og test sett
training_data_len = math.ceil(len(stock_data) * 0.8)
train_data = stock_data[:training_data_len]

# Skaler data
scaler_price = MinMaxScaler(feature_range=(0, 1))
scaled_price = scaler_price.fit_transform(train_data['Close'].values.reshape(-1, 1))

scaler_volume = MinMaxScaler(feature_range=(0, 1))
scaled_volume = scaler_volume.fit_transform(train_data['Volume'].values.reshape(-1, 1))

scaler_ema50 = MinMaxScaler(feature_range=(0, 1))
scaled_ema50 = scaler_ema50.fit_transform(train_data['EMA50'].values.reshape(-1, 1))

scaler_ema100 = MinMaxScaler(feature_range=(0, 1))
scaled_ema100 = scaler_ema100.fit_transform(train_data['EMA100'].values.reshape(-1, 1))

scaler_rsi = MinMaxScaler(feature_range=(0, 1))
scaled_rsi = scaler_rsi.fit_transform(train_data['RSI'].values.reshape(-1, 1))

# Forberede data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(np.column_stack((scaled_price[i - numberOfBars:i, 0], 
                                    scaled_volume[i - numberOfBars:i, 0], 
                                    scaled_ema50[i - numberOfBars:i, 0], 
                                    scaled_ema100[i - numberOfBars:i, 0],
                                    scaled_rsi[i - numberOfBars:i, 0])))
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)

# One-hot encoding av målvariabel
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

# Oversampling før one-hot encoding
ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train_encoded)
x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)

# Plott toppene, dalene, RSI og SMA100
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close Price'))
fig.add_trace(go.Scatter(x=stock_data.index[peak_indexes], y=stock_data['Close'].values[peak_indexes], mode='markers', name='Sell - Peaks'))
fig.add_trace(go.Scatter(x=stock_data.index[valley_indexes], y=stock_data['Close'].values[valley_indexes], mode='markers', name='Buy - Valleys'))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', yaxis="y2"))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA100'], mode='lines', name='EMA100'))

# Oppdater layout for å legge til en sekundær y-akse for RSI
fig.update_layout(
    yaxis=dict(domain=[0.3, 1]),
    yaxis2=dict(title="RSI", titlefont=dict(color="blue"), tickfont=dict(color="blue"), overlaying='y', side='right', domain=[0, 0.3])
)


fig.show()

# LSTM model
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train_resampled.shape[1], 5)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(y_train_resampled.shape[1], activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_resampled, y_train_resampled, batch_size=1, epochs=3)
model.save(modelname)

# Lagrer scaler og encoder til disk
joblib.dump(scaler_price, 'scaler_price.pkl')
joblib.dump(scaler_volume, 'scaler_volume.pkl')
joblib.dump(scaler_ema50, 'scaler_ema50.pkl')
joblib.dump(scaler_ema100, 'scaler_ema100.pkl')
joblib.dump(scaler_rsi, 'scaler_rsi.pkl')
joblib.dump(encoder, 'encoder.pkl')
