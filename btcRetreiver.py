from Binance.tickerHandlere import BinanceData
from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import pytz
from pytz import UTC
from tqdm import tqdm

apikey = "8aonjjjJ0EJt1coxcR335wl1Qsq4eYov9k2CuXUxP7IQhiBwDPOrbp5D6LANdbkW"
apisecret = "Te0MrdyCGhTcNGvmTvWDfSEG4KLPRlaIecaC0PqTtiCAnO2bB5T8miWsfQJUYDSW"
binanceData = BinanceData(apikey, apisecret)

def generate_csv(symbol, fromdate, todate, filename, klineinterval = Client.KLINE_INTERVAL_1HOUR):
    # Start- og sluttdato
    #start = datetime(2023, 8, 29, tzinfo=UTC)
    start = fromdate
    #end = datetime(2023, 6, 1, tzinfo=UTC)
    end = todate
    #end = datetime.now(pytz.UTC)
    end = end - timedelta(hours=1)  

    all_data_list = []
    interval = timedelta(days=30)

    # For å beregne totalt antall dager for tqdm
    total_days = (end - start).days

    with tqdm(total=total_days) as pbar:
        current_start = start
        while current_start < end:
            current_end = min(current_start + interval, end)
            print(current_end)
            # Hente data for det nåværende intervallet
            try:
                data = binanceData.get_historical_klines(
                    #"BTCUSDT",
                    symbol,
                    klineinterval,
                    current_start.strftime("%d %b, %Y %H:%M UTC"),
                    current_end.strftime("%d %b, %Y %H:%M UTC")
                )
                if not data.empty:
                    all_data_list.append(data)
                    new_start = data.index[-1].replace(tzinfo=UTC)
                    
                    # Sjekk for å unngå en endeløs løkke
                    if new_start == current_start:
                        current_start += interval
                    else:
                        current_start = new_start
                else:
                    current_start = current_end
            except Exception as e:
                print(f"Feil under henting av data: {e}")
                break

            # Oppdaterer tqdm fremdriftslinjen
            days_retrieved = (current_end - current_start).days
            pbar.update(days_retrieved)

    all_data = pd.concat(all_data_list, ignore_index=False)
    all_data.to_csv(filename, mode='w', index=True)