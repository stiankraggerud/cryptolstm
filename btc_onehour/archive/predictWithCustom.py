import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib
import tensorflow as tf

def get_full_path(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, filename)

def cost_sensitive_loss(y_true, y_pred):
    cost_matrix = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [1, 1, 0]
    ])
    cost_values = tf.reduce_sum(cost_matrix * y_true, axis=-1)
    loss = cost_values * keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss

# Les inn data
stock_data = pd.read_csv(get_full_path('onehour2022_2003.csv'))
stock_data.set_index('Open time', inplace=True)
model_path = get_full_path('onehour_RMSProp.h5')


stock_data['Original_Close'] = stock_data['Close'].copy()

# Beregne prosentvis endring
stock_data['Close'] = stock_data['Close'].pct_change() * 100
stock_data['Open'] = stock_data['Open'].pct_change() * 100
stock_data['High'] = stock_data['High'].pct_change() * 100
stock_data['Low'] = stock_data['Low'].pct_change() * 100
stock_data['Volume'] = stock_data['Volume'].pct_change() * 100

# Fjern NaN-verdier
stock_data.dropna(inplace=True)

# Last inn tidligere trente objekter (scaler og encoder)
scaler = joblib.load(get_full_path('scaler.pkl'))
encoder = joblib.load(get_full_path('encoder.pkl'))

# Beregne EMA
stock_data['EMA50'] = stock_data['Close'].pct_change().ewm(span=50).mean() * 100
stock_data['EMA100'] = stock_data['Close'].pct_change().ewm(span=100).mean() * 100

# Beregne MACD
macd_line = stock_data['Close'].pct_change().ewm(span=50).mean() * 100 - stock_data['Close'].pct_change().ewm(span=100).mean() * 100
signal_line = macd_line.ewm(span=9).mean()
stock_data['MACD'] = macd_line

# Erstatt uendelige verdier med NaN
stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fjern rader med NaN-verdier (eller du kan fylle dem med en egnet metode)
stock_data.dropna(inplace=True)



# Skaler testdata
scaled_test_data = scaler.transform(stock_data[['Open', 'Close', 'High', 'Low', 'Volume', 'EMA50', 'EMA100', 'MACD']])

# Forberede testdata for prediksjon
x_test = []
for i in range(500, len(stock_data)):
    x_test.append(scaled_test_data[i - 500:i])
x_test = np.array(x_test)

# Last inn tidligere trent modell
model = keras.models.load_model(model_path, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})

# Gjør prediksjoner
predictions = model.predict(x_test)
predicted_labels = encoder.inverse_transform(np.argmax(predictions, axis=1))

print(predicted_labels)


# Legg til predikerte etiketter til DataFrame
stock_data['Predicted_Signal'] = 'Hold'
stock_data['Predicted_Signal'].iloc[500:] = predicted_labels


signal_counts = stock_data['Predicted_Signal'].value_counts()
print("Number of 'Buy', 'Sell', and 'Hold' signals in the data:")
print(signal_counts)

# Visualiser resultater
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Original_Close'], mode='lines', name='Close'))
buy_signals = stock_data[stock_data['Predicted_Signal'] == 'Buy']['Original_Close']
sell_signals = stock_data[stock_data['Predicted_Signal'] == 'Sell']['Original_Close']
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))
fig.update_layout(title='Stock Price with Predicted Buy and Sell Signals', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')
fig.show()
