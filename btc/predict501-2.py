import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
import joblib
import matplotlib.pyplot as plt

# Last ned nye data
new_data = pd.read_csv('./btc/new_data1.csv')
new_data.set_index('Open Time', inplace=True)

numberOfBars = 180

new_data['Close'] = pd.to_numeric(new_data['Close'], errors='coerce')
new_data['Volume'] = pd.to_numeric(new_data['Volume'], errors='coerce')
new_data = new_data.dropna()

# Beregne EMA
new_data['EMA50'] = new_data['Close'].ewm(span=50).mean()
new_data['EMA100'] = new_data['Close'].ewm(span=100).mean()

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
model = keras.models.load_model('btc_lstm_modelRetrain1607-06.h5')

# Prediksjon
predictions = model.predict(x_new)
print(len(x_new))

# Gjenopprette encoderen
encoder = joblib.load('encoder.pkl')

# Konvertere spådommer tilbake til opprinnelige etiketter
predictions = encoder.inverse_transform(predictions.argmax(axis=1))


sell_indices = np.where(predictions == 'Sell')[0]  # Dette gir oss indeksene der 'Sell' forekommer
buy_indices = np.where(predictions == 'Buy')[0]  # Dette gir oss indeksene der 'Sell' forekommer


# Siden vi har brukt 'numberOfBars' data for hver prediksjon, må vi legge til 'numberOfBars' til våre indekser for å matche med 'new_data'
sell_indices = sell_indices + numberOfBars
buy_indices = buy_indices + numberOfBars

# Bruk disse indeksene for å hente dato og klokkeslett fra 'new_data'
sell_dates = new_data.index[sell_indices]
buy_dates = new_data.index[buy_indices]

new_data = new_data.iloc[numberOfBars:]  # Fjerner de første radene som vi ikke har prediksjoner for
new_data['Prediction'] = predictions

plt.figure(figsize=(20,10))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(new_data['Close'])
plt.plot(new_data[new_data['Prediction'] == 'Sell'].index, new_data[new_data['Prediction'] == 'Sell']['Close'], 'ro') # Markerer steder med "Sell"-prediksjon med røde prikker
plt.plot(new_data[new_data['Prediction'] == 'Buy'].index, new_data[new_data['Prediction'] == 'Buy']['Close'], 'go') # Markerer steder med "Buy"-prediksjon med grønne prikker
plt.legend(['Close', 'Sell prediction', 'Buy prediction'], loc='lower right')
plt.show()



