import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
import joblib
import plotly.graph_objects as go

# Last ned nye data
new_data = pd.read_csv('./maijuni.csv')
new_data.set_index('Open Time', inplace=True)

numberOfBars = 180

new_data['Close'] = pd.to_numeric(new_data['Close'], errors='coerce')
new_data['Volume'] = pd.to_numeric(new_data['Volume'], errors='coerce')
new_data = new_data.dropna()

# Beregne EMA
new_data['EMA50'] = new_data['Close'].ewm(span=50).mean()
new_data['EMA100'] = new_data['Close'].ewm(span=100).mean()

# Generere faktiske signaler basert på toppene og dalene
min_prominence = 90
peak_indexes, _ = find_peaks(new_data['Close'].values, prominence=min_prominence)
valley_indexes, _ = find_peaks(-new_data['Close'].values, prominence=min_prominence)
new_data.loc[new_data.index[peak_indexes], 'Signal'] = 'Sell'
new_data.loc[new_data.index[valley_indexes], 'Signal'] = 'Buy'
new_data['Signal'].fillna('Hold', inplace=True)

# Gjenopprette skaleringene
scaler_price = joblib.load('scaler_price.pkl')
scaler_volume = joblib.load('scaler_volume.pkl')
scaler_ema50 = joblib.load('scaler_ema50.pkl')
scaler_ema100 = joblib.load('scaler_ema100.pkl')

scaled_price = scaler_price.transform(new_data['Close'].values.reshape(-1,1))
scaled_volume = scaler_volume.transform(new_data['Volume'].values.reshape(-1,1))
scaled_ema50 = scaler_ema50.transform(new_data['EMA50'].values.reshape(-1,1))
scaled_ema100 = scaler_ema100.transform(new_data['EMA100'].values.reshape(-1,1))

# Forberede data for prediksjon
x_new = []

for i in range(numberOfBars, len(new_data)):
    x_new.append(np.column_stack((scaled_price[i-numberOfBars:i, 0], scaled_volume[i-numberOfBars:i, 0], scaled_ema50[i-numberOfBars:i, 0], scaled_ema100[i-numberOfBars:i, 0])))

x_new = np.array(x_new)

# Gjenopprette modellen
model = keras.models.load_model('1707_1030.h5')

# Prediksjon
predictions = model.predict(x_new)

# Gjenopprette encoderen
encoder = joblib.load('encoder.pkl')

# Konvertere spådommer tilbake til opprinnelige etiketter
predictions = encoder.inverse_transform(predictions.argmax(axis=1))

new_data = new_data.iloc[numberOfBars:]  # Fjerner de første radene som vi ikke har prediksjoner for
new_data['Prediction'] = predictions

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=new_data.index, y=new_data['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=new_data[new_data['Prediction'] == 'Sell'].index, y=new_data[new_data['Prediction'] == 'Sell']['Close'], mode='markers', name='Sell prediction', marker=dict(color='red', size=8, symbol='circle')))
fig.add_trace(go.Scatter(x=new_data[new_data['Prediction'] == 'Buy'].index, y=new_data[new_data['Prediction'] == 'Buy']['Close'], mode='markers', name='Buy prediction', marker=dict(color='green', size=8, symbol='circle')))
fig.add_trace(go.Scatter(x=new_data[new_data['Signal'] == 'Sell'].index, y=new_data[new_data['Signal'] == 'Sell']['Close'], mode='markers', name='Actual sell', marker=dict(color='red', size=8, symbol='x')))
fig.add_trace(go.Scatter(x=new_data[new_data['Signal'] == 'Buy'].index, y=new_data[new_data['Signal'] == 'Buy']['Close'], mode='markers', name='Actual buy', marker=dict(color='green', size=8, symbol='x')))
fig.update_layout(title='Model', xaxis_title='Date', yaxis_title='Close Price USD ($)', template='plotly_dark')
fig.show()
