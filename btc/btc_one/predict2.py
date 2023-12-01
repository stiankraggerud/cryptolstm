import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
import joblib
import plotly.graph_objects as go

# Last ned nye data
new_data = pd.read_csv('./juli.csv')
new_data.set_index('Open Time', inplace=True)

numberOfBars = 180

new_data['Close'] = pd.to_numeric(new_data['Close'], errors='coerce')
new_data['Volume'] = pd.to_numeric(new_data['Volume'], errors='coerce')
new_data = new_data.dropna()

# Beregne EMA
new_data['EMA50'] = new_data['Close'].ewm(span=14).mean()
new_data['EMA100'] = new_data['Close'].ewm(span=50).mean()

# Generere faktiske signaler basert på toppene og dalene
min_prominence = 120
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
model = keras.models.load_model('1807_2000.h5')

# Prediksjon
raw_predictions = model.predict(x_new)

# Gjenopprette encoderen
encoder = joblib.load('encoder.pkl')

# Filtrer basert på tærskel
threshold = 0.99
final_predictions = []

for prob in raw_predictions:
    predicted_label = encoder.inverse_transform([np.argmax(prob)])[0]

    # Dersom sannsynligheten for både 'Buy' og 'Sell' er under terskelverdien, sett til 'Hold'
    if (prob[encoder.transform(['Buy'])[0]] < threshold) and (prob[encoder.transform(['Sell'])[0]] < threshold):
        final_predictions.append('Hold')
    else:
        final_predictions.append(predicted_label)

new_data = new_data.iloc[numberOfBars:]
new_data['Filtered Prediction'] = final_predictions
sell_probabilities = raw_predictions[:, encoder.transform(['Sell'])[0]]
buy_probabilities = raw_predictions[:, encoder.transform(['Buy'])[0]]



# Forbered hovertext for 'Sell predictions'
sell_hovertext = ['Date: {}<br>Close: {:.2f}<br>Sell Probability: {:.2f}'.format(i, c, p) for i, c, p in zip(new_data[new_data['Filtered Prediction'] == 'Sell'].index, new_data[new_data['Filtered Prediction'] == 'Sell']['Close'], sell_probabilities[new_data['Filtered Prediction'] == 'Sell'])]
buy_hovertext = ['Date: {}<br>Close: {:.2f}<br>Buy Probability: {:.2f}'.format(i, c, p) for i, c, p in zip(new_data[new_data['Filtered Prediction'] == 'Buy'].index, new_data[new_data['Filtered Prediction'] == 'Buy']['Close'], buy_probabilities[new_data['Filtered Prediction'] == 'Buy'])]

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=new_data.index, y=new_data['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=new_data[new_data['Filtered Prediction'] == 'Sell'].index, y=new_data[new_data['Filtered Prediction'] == 'Sell']['Close'], mode='markers', name='Sell prediction', marker=dict(color='red', size=8, symbol='circle'), hovertext=sell_hovertext))
fig.add_trace(go.Scatter(x=new_data[new_data['Filtered Prediction'] == 'Buy'].index, y=new_data[new_data['Filtered Prediction'] == 'Buy']['Close'], mode='markers', name='Buy prediction', marker=dict(color='green', size=8, symbol='circle'), hovertext=buy_hovertext))
#fig.add_trace(go.Scatter(x=new_data[new_data['Signal'] == 'Sell'].index, y=new_data[new_data['Signal'] == 'Sell']['Close'], mode='markers', name='Actual sell', marker=dict(color='red', size=8, symbol='x')))
#fig.add_trace(go.Scatter(x=new_data[new_data['Signal'] == 'Buy'].index, y=new_data[new_data['Signal'] == 'Buy']['Close'], mode='markers', name='Actual buy', marker=dict(color='green', size=8, symbol='x')))
fig.update_layout(title='Model', xaxis_title='Date', yaxis_title='Close Price USD ($)', template='plotly_dark')
fig.show()
