from Binance.tickerHandlere import BinanceData
from binance.client import Client
from datetime import datetime, timedelta
import pytz



apikey = "8aonjjjJ0EJt1coxcR335wl1Qsq4eYov9k2CuXUxP7IQhiBwDPOrbp5D6LANdbkW"
apisecret ="Te0MrdyCGhTcNGvmTvWDfSEG4KLPRlaIecaC0PqTtiCAnO2bB5T8miWsfQJUYDSW"
binanceData = BinanceData(apikey, apisecret)

# Få nåværende dato og klokkeslett i UTC


end =  datetime.now(pytz.timezone('UTC'))
start = end - timedelta(minutes=10)

# Formater datoen og klokkeslettet
formatted_end = end.strftime("%d %b, %Y %H:%M UTC")
formatted_start = start.strftime("%d %b, %Y %H:%M UTC")


df = binanceData.get_historical_klines("BTCUSDT",Client.KLINE_INTERVAL_1MINUTE, formatted_start, formatted_end)

print(df)