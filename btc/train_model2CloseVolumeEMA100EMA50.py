import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

# Download data
stock_data = pd.read_csv('./btc/test.csv')
stock_data.set_index('Open Time', inplace=True)

numberOfBars = 180

stock_data['Close'] = pd.to_numeric(stock_data['Close'], errors='coerce')
stock_data = stock_data.dropna()

# Calculate future return
future_return = stock_data['Close'].pct_change(numberOfBars)

# Generate labels for classification
stock_data['Signal'] = np.nan
stock_data.loc[future_return > 0.007, 'Signal'] = 'Buy'
stock_data.loc[future_return < -0.007, 'Signal'] = 'Sell'

# Shift the 'Signal' column numberOfBars periods into the past
stock_data['Signal'] = stock_data['Signal'].shift(-numberOfBars)

# Remove instances with NaN in 'Signal'
stock_data.dropna(subset=['Signal'], inplace=True)

# Split data into training and test set
training_data_len = math.ceil(len(stock_data)* 0.8)
train_data = stock_data[:training_data_len]

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(train_data['Close'].values.reshape(-1,1))

# Prepare data for LSTM
x_train = []
y_train = []

for i in range(numberOfBars, len(train_data)):
    print(str(i) + ' of ' + str(len(train_data)))
    x_train.append(scaled_data[i-numberOfBars:i, 0])
    y_train.append(train_data.iloc[i]['Signal'])

x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# One-hot encode target variable
encoder = LabelEncoder()

# Count the number of each category
category_counts = train_data['Signal'].value_counts()
# Print the counts
print(category_counts)

print(np.unique(train_data['Signal']))  # before encoding
y_train = encoder.fit_transform(y_train)
y_train = keras.utils.to_categorical(y_train)
print(y_train.shape[1])  # after encoding, this should now be 2

# LSTM model
model = keras.Sequential()
model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(layers.LSTM(100, return_sequences=False))
model.add(layers.Dense(50))
model.add(layers.Dense(2, activation='softmax'))  # We now have 2 classes: 'Buy', 'Sell'
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=1, epochs=3)
model.save('btc_lstm_modelHoldTrade.h5')

# Save scaler and encoder to disk
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')
