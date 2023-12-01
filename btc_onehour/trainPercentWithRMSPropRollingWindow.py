import os
import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import stockDataHandler
from sklearn.preprocessing import MinMaxScaler


def min_max_normalization(data, columns):
    normalized_data = data.copy()
    scalers = {}
    for column in columns:
        scaler = MinMaxScaler()
        normalized_data[column] = scaler.fit_transform(data[column].values.reshape(-1, 1)).flatten()
        scalers[column] = scaler
    return normalized_data, scalers

def percent_return(data, columns):
    for column in columns:
        data[column] = data[column].pct_change()
    return data

# Load the data
stock_data = stockDataHandler.LoadData('onehour2018_2023.csv')
stockDataHandler.SetEMA(stock_data, 50, 'EMA50')
stockDataHandler.SetEMA(stock_data, 100, 'EMA100')
stockDataHandler.SetMacd(stock_data, 50)
stockDataHandler.CleanData(stock_data)

numberOfBars = 100

# Label generation
min_prominence = 150
peak_indexes, _ = find_peaks(stock_data['Original_Close'].values, prominence=min_prominence)
valley_indexes, _ = find_peaks(-stock_data['Original_Close'].values, prominence=min_prominence)
sell_indexes = peak_indexes
buy_indexes = valley_indexes

stock_data.loc[stock_data.index[sell_indexes], 'Signal'] = 'Sell'
stock_data.loc[stock_data.index[buy_indexes], 'Signal'] = 'Buy'
stock_data['Signal'].fillna('Hold', inplace=True)

# Split the data
training_data_len = math.ceil(len(stock_data) * 0.7)
train_data = stock_data[:training_data_len]

columns_to_percent_return = ['Open', 'Close', 'High', 'Low']
columns_to_min_max_normalize = ['Volume', 'EMA50', 'EMA100', 'MACD']

# Apply the desired transformations
train_data = percent_return(train_data, columns_to_percent_return)
train_data_min_max, min_max_scalers = min_max_normalization(train_data, columns_to_min_max_normalize)
train_data[columns_to_min_max_normalize] = train_data_min_max[columns_to_min_max_normalize]

# Drop NaN values after transformations
train_data = train_data.dropna()

# Print signal counts
signal_counts = train_data['Signal'].value_counts()
print("Number of 'Buy', 'Sell', and 'Hold' signals in training data:")
print(signal_counts)

# Prepare data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    x_train.append(train_data[columns_to_percent_return + columns_to_min_max_normalize].iloc[i - numberOfBars:i].values)
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)

y_train_encoded = keras.utils.to_categorical(y_train_encoded) 

# Oversampling
ros = RandomOverSampler(random_state=0)
x_train_resampled, y_train_encoded_resampled = ros.fit_resample(x_train.reshape(x_train.shape[0], -1), y_train_encoded)
x_train_resampled = x_train_resampled.reshape(-1, x_train.shape[1], x_train.shape[2])
y_train_resampled = keras.utils.to_categorical(y_train_encoded_resampled)

cost_matrix = np.array([
    [0, 1, 1],  # Kostnad fra Buy til [Buy, Sell, Hold]
    [1, 0, 1],  # Kostnad fra Sell til [Buy, Sell, Hold]
    [1, 1, 0]   # Kostnad fra Hold til [Buy, Sell, Hold]
])

import tensorflow as tf
def cost_sensitive_loss(y_true, y_pred):
    cost_values = tf.reduce_sum(cost_matrix * y_true, axis=-1)
    cost_values = tf.cast(cost_values, tf.float32)
    loss = tf.reduce_sum(cost_values * tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred), axis=-1)
    return loss

# LSTM model
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train_resampled.shape[1], len(columns_to_percent_return + columns_to_min_max_normalize))))  
model.add(layers.LSTM(100, return_sequences=True))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(y_train_encoded_resampled.shape[1], activation='softmax'))
model.summary()

# Laster inn tidligere modell:
modelname = stockDataHandler.get_full_path('onehour_2018_2023_rmsprop_rolling_epoch1.h5')
model = keras.models.load_model(modelname, custom_objects={'cost_sensitive_loss': cost_sensitive_loss})

model.compile(optimizer='RMSProp', loss=cost_sensitive_loss, metrics=['accuracy'])

model.fit(x_train_resampled, y_train_encoded_resampled, batch_size=3, epochs=5)

newModelname = stockDataHandler.get_full_path('onehour_2018_2023_rmsprop_rolling_epoch3.h5')
model.save(newModelname)

joblib.dump(min_max_scalers, stockDataHandler.get_full_path('min_max_scalers.pkl'))
joblib.dump(encoder, stockDataHandler.get_full_path('encoder.pkl'))


#TODO: Første jeg skal gjøre er å splitte ut alle felles ting, slik at jeg ikke trenger å gjøre endringer i alle tre filene når jeg skal kjøre nye tester...

# Når modellen stort sett predikerer "Kjøp" i en real-time evaluering, er det et tegn på at den kanskje har blitt overjustert til opplæringsdataene, eller at det har vært et skifte i dataene siden modellen ble trent.

# Her er noen trinn du kan ta for å forbedre modellens prestasjon:

# Data Shift: Forsikre deg om at distribusjonen av de sanntidsdataene du evaluerer på er lik distribusjonen av treningsdataene. Hvis de er vesentlig forskjellige, kan det være en forklaring på hvorfor modellen ikke presterer bra.

# Oppdater Datasettet: For aksjehandel kan eldre data være mindre relevant enn nyere data. Vurder å oppdatere treningssettet ditt regelmessig med de nyeste dataene, og kanskje fjerne noen av de eldste dataene.

# Oversampling: Selv om oversampling kan bidra til å bekjempe ubalansen i treningsdataene, kan det også føre til overjustering, spesielt hvis antallet "Kjøp" -eksempler er blitt kunstig oppblåst for mye. Vurder å redusere mengden oversampling eller prøve andre teknikker som SMOTE.

# Parameterjustering: Prøv å justere modellparametrene, som antall epoker, batchstørrelse, etc. Noen ganger kan en enkel endring i disse parametrene forbedre ytelsen betydelig.

# Kompleksitet: Hvis nettverket ditt er for komplekst, kan det føre til overjustering. Vurder å redusere antall lag eller noder for å se om det forbedrer resultatene på sanntidsdataene.

# Valideringssett: I stedet for bare å splitte dataene i trenings- og testsett, vurder å bruke et valideringssett for å finjustere modellen. Dette kan hjelpe deg med å identifisere overjustering tidligere.

# Alternative Modeller: Du kan også vurdere å prøve alternative modeller, som Random Forest, Gradient Boosting eller LSTM-nettverk, som kan være bedre egnet for denne typen tidsserieprediksjon.

# Feilanalyse: Se nærmere på de eksemplene der modellen gjør feil. Er det en trend? Er feilene konsentrert rundt bestemte tidsperioder eller hendelser? En slik analyse kan gi deg innsikt i hvorfor modellen ikke presterer bra og hva som kan gjøres for å forbedre den.

# Regelmessig Retrening: Aksjemarkedsdata endres kontinuerlig basert på globale hendelser, markedssentiment og mange andre faktorer. Det er viktig å trene modellen regelmessig med nye data for å holde den aktuell.