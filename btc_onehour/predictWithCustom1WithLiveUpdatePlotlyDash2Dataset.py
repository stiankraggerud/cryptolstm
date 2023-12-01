import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import joblib
import tensorflow as tf
import stockDataHandler
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
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ])
    cost_values = tf.reduce_sum(cost_matrix * y_true, axis=-1)
    loss = cost_values * keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss

def percent_return(data, columns):
    for column in columns:
        data[column] = data[column].pct_change()
    return data

def prepare_data(stock_data):
    stock_data = stock_data.copy()  # Skap en ny kopi for å unngå warnings
    
    stock_data['Open time'] = pd.to_datetime(stock_data['Open time'], unit='ms')
    stock_data.set_index('Open time', inplace=True)
    
    # Konverter datakolonner til riktig datatype
    stock_data = stock_data.astype(float)
    
    stockDataHandler.SetOrignalData(stock_data)
    stockDataHandler.SetEMA(stock_data, 50, 'EMA50')
    stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
    stockDataHandler.SetMacd(stock_data, 50)
    
    # Prosentvis avkastning
    columns_to_percent_return = ['Open', 'Close', 'High', 'Low']
    stock_data = percent_return(stock_data, columns_to_percent_return)
    
    stock_data_for_plotting = stock_data.copy()

    # Normalisering med MinMaxScaler
    columns_to_min_max_normalize = ['Volume']
    for column in columns_to_min_max_normalize:
        stock_data[column] = min_max_scalers[column].transform(stock_data[column].values.reshape(-1, 1)).flatten()

    # Fjern uønskede data og rengjør datasettet
    stockDataHandler.CleanData(stock_data)
    

    # Sørg for at kolonner er i en konsistent rekkefølge (legg til kode for å definere 'expected_columns' basert på treningsdataen din)
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA50', 'EMA100', 'MACD']
    stock_data = stock_data[expected_columns]
    if list(stock_data.columns) != expected_columns:
        raise ValueError(f"Columns of stock_data do not match expected columns. Got {list(stock_data.columns)}, expected {expected_columns}")



    return stock_data, stock_data_for_plotting

# (resten av funksjonene og definisjoner forblir uendret)

min_max_scalers = joblib.load(get_full_path('min_max_scalers.pkl'))
encoder = joblib.load(get_full_path('encoder.pkl'))
model_path = get_full_path('onehour_2018_2023_rmsprop_rolling_epoch1.h5')
model = keras.models.load_model(model_path, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='live-graph'),
    dcc.Interval(
            id='interval-component',
            interval=180*1000,
            n_intervals=0
    )
])

@app.callback(Output('live-graph', 'figure'), Input('interval-component', 'n_intervals'))
def update_graph(stock_data_df):
    # Anta at stock_data_df allerede er i riktig format
    stock_data, stock_data_for_plotting = prepare_data(stock_data_df.copy())  # Forbered dataen
    
    x_test = []
    numberOfBars = 100
    for i in range(numberOfBars, len(stock_data)):
        x_test.append(stock_data.iloc[i - numberOfBars:i].values)
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
    
    current_time = datetime.now().strftime('%H:%M:%S')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data_for_plotting.index, y=stock_data_for_plotting['Original_Close'], mode='lines', name='Close'))
    buy_indices = stock_data[stock_data['Predicted_Signal'] == 'Buy'].index
    buy_signals = stock_data_for_plotting.loc[buy_indices]['Original_Close']
    sell_indices = stock_data[stock_data['Predicted_Signal'] == 'Sell'].index
    sell_signals = stock_data_for_plotting.loc[sell_indices]['Original_Close']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))
    fig.update_layout(title='Stock Price with Predicted Buy and Sell Signals (Updated at: {})'.format(current_time))

    return fig

if __name__ == '__main__':
    # Last inn datasettet her og send det til Dash-appen som en parameter
    stock_data_df = pd.read_csv('path_to_your_data.csv')  # Erstatt 'path_to_your_data.csv' med stien til datasettet ditt
    app.run_server(debug=True)
