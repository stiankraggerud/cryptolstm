import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import joblib
import tensorflow as tf
import stockDataHandler
from binance.client import Client
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from datetime import datetime


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

api_key = "8aonjjjJ0EJt1coxcR335wl1Qsq4eYov9k2CuXUxP7IQhiBwDPOrbp5D6LANdbkW"
api_secret = "Te0MrdyCGhTcNGvmTvWDfSEG4KLPRlaIecaC0PqTtiCAnO2bB5T8miWsfQJUYDSW"
client = Client(api_key, api_secret)

scaler = joblib.load(get_full_path('scaler.pkl'))
encoder = joblib.load(get_full_path('encoder.pkl'))
model_path = get_full_path('onehour_rmsprop_epoch1.h5')
model = keras.models.load_model(model_path, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(
            id='interval-component',
            interval=5*1000,
            n_intervals=0
    )
])

@app.callback(Output('live-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph(n):
    client = Client(api_key, api_secret)

    klines = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, limit=300)
    columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    df = pd.DataFrame(klines, columns=columns)
    stock_data = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    stock_data['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    stock_data.set_index('Open time', inplace=True)
    stock_data['Open'] = stock_data['Open'].astype(float)
    stock_data['High'] = stock_data['High'].astype(float)
    stock_data['Low'] = stock_data['Low'].astype(float)
    stock_data['Close'] = stock_data['Close'].astype(float)
    stock_data['Volume'] = stock_data['Volume'].astype(float)

    stockDataHandler.SetOrignalData(stock_data)
    stockDataHandler.SetEMA(stock_data, 50, 'EMA50')
    stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
    stockDataHandler.SetMacd(stock_data, 50)
    stockDataHandler.CleanData(stock_data)

    scaled_test_data = scaler.transform(stock_data[['Open', 'Close', 'High', 'Low', 'Volume', 'EMA50', 'EMA100', 'MACD']])
    
    x_test = []
    numberOfBars = 100
    for i in range(numberOfBars, len(stock_data)):
        x_test.append(scaled_test_data[i - numberOfBars:i])
    x_test = np.array(x_test)

    predictions = model.predict(x_test)
    buy_probs = predictions[:, encoder.transform(['Buy'])[0]]
    sell_probs = predictions[:, encoder.transform(['Sell'])[0]]

    threshold = 0.9
    predicted_labels = []
    for buy_prob, sell_prob in zip(buy_probs, sell_probs):
        if buy_prob > threshold:
            predicted_labels.append('Buy')
        elif sell_prob > threshold:
            predicted_labels.append('Sell')
        else:
            predicted_labels.append('Hold')

    stock_data['Predicted_Signal'] = 'Hold'
    stock_data['Predicted_Signal'].iloc[numberOfBars:] = predicted_labels
    last_point = stock_data.iloc[-1]
    last_prediction = last_point['Predicted_Signal']

    current_time = datetime.now().strftime('%H:%M:%S')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Original_Close'], mode='lines', name='Close'))
    buy_signals = stock_data[stock_data['Predicted_Signal'] == 'Buy']['Original_Close']
    sell_signals = stock_data[stock_data['Predicted_Signal'] == 'Sell']['Original_Close']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))
    fig.update_layout(title='Stock Price with Predicted Buy and Sell Signals', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')
    
    if last_prediction == "Buy":
        fig.add_trace(go.Scatter(x=[last_point.name], y=[last_point['Original_Close']], mode='markers', name='Last Point - Buy', marker=dict(color='green', size=10, symbol='circle')))
    elif last_prediction == "Sell":
        fig.add_trace(go.Scatter(x=[last_point.name], y=[last_point['Original_Close']], mode='markers', name='Last Point - Sell', marker=dict(color='red', size=10, symbol='circle')))
    else:  # Hold
        fig.add_trace(go.Scatter(x=[last_point.name], y=[last_point['Original_Close']], mode='markers', name='Last Point - Hold', marker=dict(color='blue', size=10, symbol='circle')))


    updated_annotation = {
    'xref': 'paper',
    'yref': 'paper',
    'x': 1,
    'y': 1.05,
    'showarrow': False,
    'text': f'Sist oppdatert {current_time}',
    'font': {
        'size': 15
    }
}

    fig.update_layout(annotations=[updated_annotation])

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
