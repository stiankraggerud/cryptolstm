import pandas as pd
import os
import numpy as np
import plotly.graph_objects as go


def get_full_path(filename):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_directory, filename)


def get_saved_models_path(filename):  
    current_directory = os.path.dirname(os.path.abspath(__file__))  
    project_root = current_directory  
  
    # Gå opp i katalogstrukturen til du når prosjektets rotkatalog  
    while os.path.basename(project_root) != "lstmBtc":  
        if(project_root == 'c:\\'):
            raise Exception('Fant ikke katalog')
        project_root = os.path.dirname(project_root)  
  
    saved_models_directory = os.path.join(project_root, "saved_models_train_v4")  
    return os.path.join(saved_models_directory, filename)  


def create_sequences(X, y, sequence_length):  
    X_sequences = []  
    y_sequences = []  
  
    for i in range(len(X) - sequence_length):  
        X_sequences.append(X[i:i + sequence_length])  
        y_sequences.append(y[i + sequence_length])  
  
    X_sequences = np.array(X_sequences)  
    y_sequences = np.array(y_sequences)  
  
    return X_sequences, y_sequences


def get_filename_without_extension(file_path):  
    current_file = os.path.basename(file_path)  
    filename_without_extension, _ = os.path.splitext(current_file)  
    return filename_without_extension  

#Open time,Open,High,Low,Close,Volume,Close time,Quote asset volume,Number of trades,Taker buy base asset volume,Taker buy quote asset volume,Ignore
def SetOrignalData(stock_data):
   # stock_data.set_index('Open time', inplace=True)

    stock_data['Original_Close'] = stock_data['Close'].copy()
    stock_data['Original_Open'] = stock_data['Open'].copy()
    stock_data['Original_High'] = stock_data['High'].copy()
    stock_data['Original_Low'] = stock_data['Low'].copy()
    stock_data['Original_Number_of_trades'] = stock_data['Number of trades'].copy()
    stock_data['Original_Taker_buy__base_asset_volume'] = stock_data['Taker buy base asset volume'].copy()
    stock_data['Original_Taker_buy__quote_asset_volume'] = stock_data['Taker buy quote asset volume'].copy()

    # Beregne prosentvis endring
    stock_data['Close_Pct'] = stock_data['Close'].pct_change() * 100
    stock_data['Open_Pct'] = stock_data['Open'].pct_change() * 100
    stock_data['High_Pct'] = stock_data['High'].pct_change() * 100
    stock_data['Low_Pct'] = stock_data['Low'].pct_change() * 100
    stock_data['Volume_Pct'] = stock_data['Volume'].pct_change() * 100
    stock_data.dropna(inplace=True)




def LoadData(filename):
    stock_data = pd.read_csv(get_full_path(filename))
    SetOrignalData(stock_data)
    return stock_data

def SetEMA(data, emaPeriod, emaName):
    data[emaName] = data['Close'].pct_change().ewm(span=emaPeriod).mean() * 100


def SetMacd(data, period):
    macd_line = data['Original_Close'].pct_change().ewm(span=period).mean() * 100 - data['Original_Close'].pct_change().ewm(span=100).mean() * 100
    signal_line = macd_line.ewm(span=9).mean()
    data['MACD'] = macd_line


def compute_RSI(data, window=14):
    delta = data['Original_Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    data['RSI'] = rsi
    return data

def compute_ADX(data, window=14):
    TR = np.maximum(data['High'] - data['Low'], 
                   np.maximum(abs(data['High'] - data['Close'].shift(1)), 
                              abs(data['Low'] - data['Original_Close'].shift(1))))
    data['+DM'] = np.where(((data['High'] - data['High'].shift(1)) > (data['Low'].shift(1) - data['Low'])) & 
                          ((data['High'] - data['High'].shift(1)) > 0), data['High'] - data['High'].shift(1), 0)
    data['-DM'] = np.where(((data['Low'].shift(1) - data['Low']) > (data['High'] - data['High'].shift(1))) & 
                          ((data['Low'].shift(1) - data['Low']) > 0), data['Low'].shift(1) - data['Low'], 0)

    data['TR'] = TR
    data['+DM'] = data['+DM'].rolling(window=window).sum()
    data['-DM'] = data['-DM'].rolling(window=window).sum()
    TR = TR.rolling(window=window).sum()

    data['+DI'] = 100 * (data['+DM'] / TR)
    data['-DI'] = 100 * (data['-DM'] / TR)
    DX = 100 * abs((data['+DI'] - data['-DI']) / (data['+DI'] + data['-DI']))

    ADX = DX.rolling(window=window).mean()

    data['ADX'] = ADX
    return data

def CleanData(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Fjern rader med NaN-verdier (eller du kan fylle dem med en egnet metode)
    data.dropna(inplace=True)

def VisualizeData(data):
    # Visualiser resultater
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Original_Close'], mode='lines', name='Close'))
    buy_signals = stock_data[stock_data['Predicted_Signal'] == 'Buy']['Original_Close']
    sell_signals = stock_data[stock_data['Predicted_Signal'] == 'Sell']['Original_Close']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))
    fig.update_layout(title='Stock Price with Predicted Buy and Sell Signals', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')
    fig.show()


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


import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


def showDataInGraph(stock_data):
    # Create a deep copy of the dataset
    stock_data_copy = stock_data.copy(deep=True)

    stock_data_copy.set_index('Open time', inplace=True)

    # Convert the index to datetime format
    stock_data_copy.index = pd.to_datetime(stock_data_copy.index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data_copy.index, y=stock_data_copy['Original_Close'], mode='lines', name='Close'))
    buy_signals = stock_data_copy[stock_data_copy['Signal'] == 'Buy']['Original_Close']
    sell_signals = stock_data_copy[stock_data_copy['Signal'] == 'Sell']['Original_Close']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals, mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='circle')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals, mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='circle')))
    fig.update_layout(title='Stock Price with Predicted Buy and Sell Signals', xaxis_title='Date', yaxis_title='Close Price', template='plotly_dark')

    print(stock_data_copy.columns)

    app = dash.Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id='live-graph', figure=fig)
    ])

    app.run_server(debug=True)

from sklearn.preprocessing import MinMaxScaler
def min_max_normalization(data, columns, scalers=None):
    normalized_data = data.copy()
    if scalers is None:
        scalers = {}
    for column in columns:
        scaler = scalers.get(column, MinMaxScaler())
        normalized_data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1)).flatten()
        scalers[column] = scaler
    return normalized_data, scalers

def percent_return(data, columns):
    for column in columns:
        data[column] = data[column].pct_change()
    return data

