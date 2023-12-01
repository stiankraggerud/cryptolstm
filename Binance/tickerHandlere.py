from binance.client import Client
import pandas as pd

class BinanceData:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)

    def get_historical_klines(self, symbol, interval, start, end):
        klines = self.client.get_historical_klines(symbol, interval, start, end)

        #lager en pandas dataframe
        df = pd.DataFrame(klines, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time','Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        df.set_index('Open time', inplace=True)
        return df
