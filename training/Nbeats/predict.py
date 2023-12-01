import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from tensorflow import keras  
import joblib  
import stockDataHandler  
from sklearn.preprocessing import MinMaxScaler  
import math  
  
# Funksjoner fra den opprinnelige koden som er nødvendige for preprocessing  
def min_max_normalization(data, columns):  
    for column in columns:  
        scaler = min_max_scalers[column]  # Bruk lagrede scalers  
        data[column + '_min_max'] = scaler.transform(data[column].values.reshape(-1, 1)).flatten()  
  
def percent_return(data, columns):  
    for column in columns:  
        data[column + '_pct_return'] = data[column].pct_change()  
  
def convert_percentage_to_price(y_percentage, start_value):  
    prices = [start_value]  
    for percent_change in y_percentage:  
        prices.append(prices[-1] * (1 + percent_change))  
    return prices[1:]  
  
def revert_preprocessing(data, percent_columns, min_max_columns, start_value):    
    reverted_data = data.copy()    
    
    # Reverse percent return    
    for column in percent_columns:    
        reverted_data[column] = reverted_data[column].apply(lambda x: (x + 1) * start_value)    
    
    # Reverse min-max normalization    
    for column in min_max_columns:    
        scaler = min_max_scalers[column]    
        reverted_data[column] = scaler.inverse_transform(data[column].values.reshape(-1, 1)).flatten()    
    
    return reverted_data  
  
# Last inn den trente modellen og scalers  
model_path = stockDataHandler.get_full_path('onehour_2018_2023_nbeats.h5')  
scalers_path = stockDataHandler.get_full_path('min_max_scalers_nbeats.pkl')  
  
model = keras.models.load_model(model_path)  
  
  
min_max_scalers = joblib.load(scalers_path)  
  
# Last inn dataen  
stock_data = stockDataHandler.LoadData('onehour2022_2003.csv')  
stockDataHandler.SetEMA(stock_data, 12, 'EMA12')  
stockDataHandler.SetEMA(stock_data, 100, 'EMA100')  
stockDataHandler.SetMacd(stock_data, 50)  
stockDataHandler.compute_ADX(stock_data,14)  
stockDataHandler.compute_RSI(stock_data,14)  
stockDataHandler.CleanData(stock_data)  
  
  
  
stock_data['Open time'] = pd.to_datetime(stock_data['Open time'])  
stock_data.set_index('Open time', inplace=True)  
  
print('stock data length')  
print(len(stock_data))  
  
  
# Husk å splitte dataen  
training_data_len = math.ceil(len(stock_data) * 0.7)  
  
start_date = stock_data.index[training_data_len - 1]    
start_value = stock_data['Close'].iloc[training_data_len - 1]  
  
test_data = stock_data[training_data_len:]  
  
  
  
columns_to_percent_return = ['Open', 'Close', 'High', 'Low', 'RSI', 'ADX']  
columns_to_min_max_normalize = ['Volume', 'EMA12', 'EMA100']  
  
# Preprocessing på test_data  
percent_return(test_data, columns_to_percent_return)  
min_max_normalization(test_data, columns_to_min_max_normalize)  
  
  
x_test = []  
y_test_true = []  
  
numberOfBars = 100  
output_horizon = 1  
  
for i in range(numberOfBars, len(test_data) - output_horizon):  
    x_test.append(test_data[[f'{column}_pct_return' for column in columns_to_percent_return] + [f'{column}_min_max' for column in columns_to_min_max_normalize]].iloc[i - numberOfBars:i].values)  
    y_test_true.append(test_data['Close_pct_return'].iloc[i:i+output_horizon].values)  
  
x_test = np.array(x_test)  
y_test_true = np.array(y_test_true)  
  

first_element_training_data = stock_data.iloc[0]  



# Prediker med modellen  
y_test_pred = model.predict(x_test)  
  
  
y_test_true_price = convert_percentage_to_price(y_test_true.flatten(), start_value)  
y_test_pred_price = convert_percentage_to_price(y_test_pred.flatten(), start_value)  
  
import matplotlib.dates as mdates  
  
dates = test_data.index[numberOfBars:len(test_data) - output_horizon].to_list()  
  
print('Dato')  
print(dates[-1])  
  
print('Faktisk pris---------------------------')  
print(y_test_true_price[-1])  
print('Predikert pris---------------------------')  
print(y_test_pred_price[-1])  
  
# Visualiser prediksjonene sammen med de faktiske verdiene  
plt.figure(figsize=(15, 8))  
plt.plot(dates, y_test_true_price, color='blue', label='True Close Price')  
plt.plot(dates, y_test_pred_price, color='red', linestyle='dashed', label='Predicted Close Price')  
  
# Legg til dette for å endre datofremvisningen på x-aksen:  
ax = plt.gca()  
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  
ax.xaxis.set_major_locator(mdates.AutoDateLocator())  
  
plt.title('NBeats Prediction vs True Data')  
plt.xlabel('Date')  
plt.ylabel('Close Price')  
plt.legend()  
plt.xticks(rotation=45)  # Roter x-aksen etiketter for bedre lesbarhet  
plt.tight_layout()  
plt.show()  
