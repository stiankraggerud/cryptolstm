import os
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os


def get_full_path(filename):
    #current_directory = os.getcwd()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, filename)

# Last ned data
stock_data = pd.read_csv(get_full_path('btc_2017_november2023_5min.csv'))
stock_data.set_index('Open time', inplace=True)
modelname = get_full_path('btc_onehour.h5')
numberOfBars = 500

stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data['Volume'] = pd.to_numeric(stock_data['Volume'], errors='coerce')
stock_data = stock_data.dropna()

# Beregne EMA
stock_data['EMA50'] = stock_data['Close'].ewm(span=50).mean()
stock_data['EMA12'] = stock_data['Close'].ewm(span=12).mean()

# Beregne MACD
macd_line = stock_data['Close'].ewm(span=50).mean() - stock_data['Close'].ewm(span=100).mean()
signal_line = macd_line.ewm(span=9).mean()  # Default configuration
stock_data['MACD'] = macd_line
stock_data['Signal_Line'] = signal_line

stock_data = stock_data[~stock_data.index.duplicated(keep='first')]

# Label generation
if 'Signal' in stock_data.columns:
    stock_data = stock_data.drop(columns=['Signal'])

import numpy as np
from scipy.signal import find_peaks

# Definer parametrene for analyse
min_prominence = 15
fortjeneste_terskel = 0.8 # Fortjeneste på 2%
tidsramme = 5  # Antall perioder for å oppnå fortjeneste

# Finn bunner
valley_indexes, _ = find_peaks(-stock_data['Close'].values, prominence=min_prominence)

# Initialiser en tom liste for signaler

signals = ['Hold'] * len(stock_data)
active_buy_index = None  # Holder styr på om det finnes et aktivt 'Buy' signal som venter på et 'Sell'

# Gå gjennom hver bunn og sjekk for fortjeneste innen tidsramme og at lukkeprisen er større enn EMA12
for valley_index in valley_indexes:
    if active_buy_index is not None:
        # Sjekk om vi har passert tidsrammen for det aktive kjøpssignalet uten å finne et salgssignal
        if valley_index > active_buy_index + tidsramme:
            signals[active_buy_index] = 'Hold'
            active_buy_index = None

    # Sjekk om det er et potensielt kjøpssignal og at vi ikke allerede venter på et salgssignal
    if active_buy_index is None and stock_data['Close'].iloc[valley_index] > stock_data['EMA12'].iloc[valley_index]:
        # Vi har et nytt kjøpssignal
        active_buy_index = valley_index
        signals[valley_index] = 'Buy'
        
        # Se etter et salgssignal innen tidsrammen
        for i in range(valley_index + 1, min(len(stock_data), valley_index + tidsramme + 1)):
            if (stock_data['Close'].iloc[i] - stock_data['Close'].iloc[valley_index]) / stock_data['Close'].iloc[valley_index] * 100 >= fortjeneste_terskel:
                # Vi har funnet et salgssignal
                signals[i] = 'Sell'
                active_buy_index = None
                break

# Sjekk etter enden av løkken hvis det siste kjøpssignalet ikke har et salgssignal
if active_buy_index is not None:
    signals[active_buy_index] = 'Hold'

# Oppdater 'Signal' kolonnen i DataFrame
stock_data['Signal'] = signals


import plotly.graph_objects as go

# Opprett figur
fig = go.Figure()

# Legg til linjediagram for lukkekursen
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Close'))

# Finn kjøpssignalene og legg dem til i plottet
buy_signals = stock_data[stock_data['Signal'] == 'Buy']['Close']
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))

# Finn salgssignalene og legg dem til i plottet
sell_signals = stock_data[stock_data['Signal'] == 'Sell']['Close']
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))

# Oppdater layout for å gjøre plottet mer informativt
fig.update_layout(title='Stock Price with Predicted Buy and Sell Signals', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')

# Vis plottet
fig.show()


# Splitt data i trening og test sett
training_data_len = math.ceil(len(stock_data) * 0.7)
train_data = stock_data[:training_data_len]

# Skaler data
scaler_price = MinMaxScaler(feature_range=(0, 1))
scaled_price = scaler_price.fit_transform(train_data['Close'].values.reshape(-1, 1))

scaler_volume = MinMaxScaler(feature_range=(0, 1))
scaled_volume = scaler_volume.fit_transform(train_data['Volume'].values.reshape(-1, 1))

scaler_ema50 = MinMaxScaler(feature_range=(0, 1))
scaled_ema50 = scaler_ema50.fit_transform(train_data['EMA50'].values.reshape(-1, 1))

scaler_ema12 = MinMaxScaler(feature_range=(0, 1))
scaled_ema12 = scaler_ema12.fit_transform(train_data['EMA12'].values.reshape(-1, 1))


scaler_macd = MinMaxScaler(feature_range=(0, 1))
scaled_macd = scaler_macd.fit_transform(train_data['MACD'].values.reshape(-1, 1))

# Forberede data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(np.column_stack((scaled_price[i - numberOfBars:i, 0],
                                    scaled_volume[i - numberOfBars:i, 0],
                                    scaled_ema12[i - numberOfBars:i, 0],
                                    scaled_ema50[i - numberOfBars:i, 0],
                                    scaled_macd[i - numberOfBars:i, 0]
                                    )))
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)

#assert False

# One-hot encoding av målvariabel
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

# Oversampling før one-hot encoding
ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train_encoded)
x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)

# LSTM model
model = keras.Sequential()
model.add(layers.LSTM(400, return_sequences=True, input_shape=(x_train_resampled.shape[1], 5)))  # Changed from 5 to 6 because of an extra feature
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(y_train_resampled.shape[1], activation='softmax'))
model.summary()


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train_resampled, y_train_resampled, batch_size=1, epochs=6)
model.save(modelname)

# # Lagrer scaler og encoder til disk
# joblib.dump(scaler_price, get_full_path('scaler_price.pkl'))
# joblib.dump(scaler_volume, get_full_path('scaler_volume.pkl'))
# joblib.dump(scaler_ema50, get_full_path('scaler_ema50.pkl'))
# joblib.dump(scaler_ema12, get_full_path('scaler_ema100.pkl'))
# #joblib.dump(scaler_ema300, get_full_path('scaler_ema300.pkl'))
# joblib.dump(scaler_macd, get_full_path('scaler_macd.pkl'))
# joblib.dump(encoder, get_full_path('encoder.pkl'))
